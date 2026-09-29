"""
Before/after card art comparison (spec §6).

Renders 8 fixed prompts twice:
  old: stabilityai/sdxl-turbo, 1 step, guidance 0, 408x336, the old prompt string
  new: image_generation.generate_art (config.IMAGE_MODEL_ID, SDXL Lightning)
and writes data/compare/index.html with the images side by side, the prompts and timings.

Usage (from proxy-server/):  <PY> tools/compare_art.py
"""
import base64
import gc
import html
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
import image_generation as ig  # noqa: E402

OUT_DIR = Path(config.DATA_DIR) / "compare"

# (label, subject, card_data): W, U, B, R, G, multicolor, artifact, land
PROMPTS = [
    ("White", "an archangel of justice descending on a battlefield",
     {"colors": ["W"], "type": "Creature"}),
    ("Blue", "a sphinx posing riddles to a lost scholar in a sunken library",
     {"colors": ["U"], "type": "Creature"}),
    ("Black", "a necromancer's ritual raising skeletons from a moonlit graveyard",
     {"colors": ["B"], "type": "Sorcery"}),
    ("Red", "a goblin shaman hurling a fireball",
     {"colors": ["R"], "type": "Creature"}),
    ("Green", "an ancient treefolk guardian in a misty primeval forest",
     {"colors": ["G"], "type": "Creature"}),
    ("Multicolor (Grixis)", "a dragon overlord commanding a storm above a burning city",
     {"colors": ["U", "B", "R"], "type": "Legendary Creature"}),
    ("Artifact", "an ornate brass compass that points to hidden treasure",
     {"colors": [], "type": "Artifact"}),
    ("Land", "a crystal cavern glowing with raw mana",
     {"colors": [], "type": "Land"}),
]

# ----- Old prompt construction, frozen verbatim from the pre-AI-Night createCardImage -----
_OLD_MONO_BLUE = {
    "creature": ", detailed creature portrait, living being, aquatic creature, flying creature, sea monster, elemental being, sphinx, merfolk, bird, octopus, dragon, character focus",
    "instant": ", water magic, ice effects, wind storm, lightning, teleportation, illusion magic, time distortion, crystal energy, arcane symbols, spell energy",
    "sorcery": ", tidal wave, storm clouds, ice formation, mystical library, ancient knowledge, arcane research, spell scrolls, crystal formations, time magic",
    "enchantment": ", shimmering water, floating islands, aurora effects, crystalline structures, frozen landscape, misty atmosphere, magical academy, ancient library, time distortion",
    "planeswalker": ", powerful planeswalker character, scholar, artificer, elemental master, sea witch, storm caller, ancient being, magical portrait, character focus",
}
_OLD_TYPES = [
    ("creature", ", detailed creature portrait, living being, character focus"),
    ("instant", ", magical effect in progress, spell energy, dynamic action, casting magic"),
    ("sorcery", ", grand magical ritual, powerful spell effect, mystical ceremony, magical transformation"),
    ("artifact", ", detailed artifact object, magical device, ancient relic, crafted item focus"),
    ("enchantment", ", magical aura, enchanted environment, mystical atmosphere, ongoing magic effect"),
    ("land", ", landscape view, terrain, natural environment, geographical location"),
    ("planeswalker", ", powerful planeswalker character, magical being, character focus"),
    ("battle", ", epic battle scene, conflict, warfare, dramatic confrontation"),
]


def old_prompt(prompt: str, card_data: dict) -> str:
    colors = card_data.get("colors", [])
    palette = ""
    if colors:  # the old code gave colorless cards no palette
        found = ig.COLOR_PALETTES.get(frozenset(colors)) or ig._PALETTE_FALLBACK_BY_COUNT.get(len(colors), "")
        palette = f", color palette: {found}" if found else ""
    card_type = card_data.get("type", "").lower()
    context = ", magical fantasy scene"
    for keyword, text in _OLD_TYPES:
        if keyword in card_type:
            mono_blue = colors == ["U"] and keyword in _OLD_MONO_BLUE
            context = _OLD_MONO_BLUE[keyword] if mono_blue else text
            break
    art_prompt = f"{prompt}{context}{palette}, fantasy art, Magic: The Gathering style, detailed illustration, dramatic lighting"
    return ig.truncate_prompt_smartly(art_prompt, max_tokens=75)


# ----- Rendering -----
def render_old() -> tuple[list[dict], float]:
    import torch
    from diffusers import AutoPipelineForText2Image

    use_cuda = config.USE_CUDA and torch.cuda.is_available()
    start = time.perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        variant="fp16" if use_cuda else None,
    ).to("cuda" if use_cuda else "cpu")
    if use_cuda:  # same memory settings as the old get_image_pipeline
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - start

    results = []
    for i, (_, subject, card_data) in enumerate(PROMPTS, 1):
        text = old_prompt(subject, card_data)
        start = time.perf_counter()
        image = pipe(prompt=text, num_inference_steps=1, guidance_scale=0.0, width=408, height=336).images[0]
        seconds = time.perf_counter() - start
        image.save(OUT_DIR / f"old_{i}.png")
        results.append({"prompt": text, "seconds": seconds})
        print(f"old {i}/{len(PROMPTS)}: {seconds:.2f}s")

    del pipe
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return results, load_seconds


def render_new() -> tuple[list[dict], float]:
    start = time.perf_counter()
    ig._get_pipeline()
    load_seconds = time.perf_counter() - start

    results = []
    for i, (_, subject, card_data) in enumerate(PROMPTS, 1):
        positive, negative = ig.build_art_prompt(subject, card_data)
        start = time.perf_counter()
        art_b64 = ig.generate_art(subject, card_data)
        seconds = time.perf_counter() - start
        (OUT_DIR / f"new_{i}.png").write_bytes(base64.b64decode(art_b64))
        results.append({"prompt": positive, "negative": negative, "seconds": seconds})
        print(f"new {i}/{len(PROMPTS)}: {seconds:.2f}s")
    return results, load_seconds


def ollama_resident() -> str:
    """Names of Ollama models resident in memory (they share the GPU), or 'none'/'unknown'."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=2) as resp:
            models = json.load(resp).get("models", [])
        return ", ".join(m.get("name", "?") for m in models) or "none"
    except Exception:
        return "unknown (Ollama not reachable)"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_html(old: list[dict], old_load: float, new: list[dict], new_load: float, llm: str) -> Path:
    rows = []
    for i, ((label, subject, card_data), o, n) in enumerate(zip(PROMPTS, old, new), 1):
        colors = "".join(card_data["colors"]) or "colorless"
        rows.append(f"""
    <section class="row">
      <h2>{i}. {html.escape(label)} <span class="meta">{html.escape(card_data['type'])} · {colors}</span></h2>
      <p class="subject">“{html.escape(subject)}”</p>
      <div class="pair">
        <figure>
          <img src="old_{i}.png" width="408" height="336" alt="Old art for {html.escape(subject)}">
          <figcaption><b>Before</b> · SDXL-Turbo · {o['seconds']:.2f}s
            <code>{html.escape(o['prompt'])}</code></figcaption>
        </figure>
        <figure>
          <img src="new_{i}.png" width="408" height="336" alt="New art for {html.escape(subject)}">
          <figcaption><b>After</b> · {html.escape(config.IMAGE_MODEL_ID)} · {n['seconds']:.2f}s
            <code>{html.escape(n['prompt'])}</code></figcaption>
        </figure>
      </div>
    </section>""")

    old_warm = _mean([r["seconds"] for r in old[1:]])
    new_warm = _mean([r["seconds"] for r in new[1:]])
    gen_w, gen_h = config.IMAGE_GEN_SIZE
    art_w, art_h = config.ART_BOX_SIZE
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Art Comparison</title>
<style>
  :root {{ --bg:#f6f5f2; --fg:#1d1d1f; --muted:#6b6b70; --card:#ffffff; --line:#e2e0da; --code:#f0eee9; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#16161a; --fg:#ececef; --muted:#9a9aa3; --card:#1f1f25; --line:#2e2e36; --code:#26262d; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 900px; margin: 0 auto; padding: 24px 16px 64px; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 4px; }}
  .lede {{ color: var(--muted); margin: 0 0 20px; }}
  table {{ border-collapse: collapse; margin: 0 0 28px; width: 100%; background: var(--card); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }}
  tr:last-child td {{ border-bottom: 0; }}
  .row {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin: 0 0 20px; }}
  h2 {{ font-size: 1.1rem; margin: 0; }}
  .meta {{ color: var(--muted); font-weight: 400; font-size: .9rem; }}
  .subject {{ margin: 4px 0 12px; font-style: italic; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 720px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  figure {{ margin: 0; }}
  img {{ width: 100%; height: auto; display: block; border-radius: 6px; background: #000; }}
  figcaption {{ font-size: .85rem; margin-top: 6px; }}
  code {{ display: block; margin-top: 4px; padding: 6px 8px; background: var(--code); border-radius: 4px; font-size: .75rem; color: var(--muted); white-space: normal; word-break: break-word; }}
</style>
</head>
<body>
<main>
  <h1>Card art: before and after</h1>
  <p class="lede">Same 8 subjects rendered with the old setup and the new one. Both are shown at the {art_w}×{art_h} art-box size used on the card.</p>
  <table>
    <tr><th></th><th>Before</th><th>After</th></tr>
    <tr><td>Model</td><td>stabilityai/sdxl-turbo</td><td>{html.escape(config.IMAGE_MODEL_ID)}</td></tr>
    <tr><td>Steps · guidance</td><td>1 · 0.0</td><td>{config.IMAGE_STEPS} · {config.IMAGE_GUIDANCE}</td></tr>
    <tr><td>Render size</td><td>408×336</td><td>{gen_w}×{gen_h}, Lanczos to {art_w}×{art_h}</td></tr>
    <tr><td>Negative prompt</td><td>none</td><td>{html.escape(ig.NEGATIVE_PROMPT)}</td></tr>
    <tr><td>Model load</td><td>{old_load:.1f}s</td><td>{new_load:.1f}s</td></tr>
    <tr><td>Mean per image (images 2–8)</td><td>{old_warm:.2f}s</td><td>{new_warm:.2f}s</td></tr>
    <tr><td>First image</td><td>{old[0]['seconds']:.2f}s</td><td>{new[0]['seconds']:.2f}s</td></tr>
    <tr><td>GPU memory</td><td>whole pipeline on GPU</td><td>{'model CPU offload, ' if config.IMAGE_CPU_OFFLOAD else ''}VAE {html.escape(str(config.IMAGE_VAE_ID))}</td></tr>
  </table>
  <p class="lede">LLM resident in GPU memory during this run: {html.escape(llm)}.</p>
  {''.join(rows)}
</main>
</body>
</html>
"""
    out = OUT_DIR / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    llm = ollama_resident()
    print(f"Ollama models resident: {llm}")
    print("== old setup: stabilityai/sdxl-turbo ==")
    old, old_load = render_old()
    print(f"== new setup: {config.IMAGE_MODEL_ID} ==")
    new, new_load = render_new()
    out = write_html(old, old_load, new, new_load, llm)
    print(f"old: load {old_load:.1f}s, warm mean {_mean([r['seconds'] for r in old[1:]]):.2f}s")
    print(f"new: load {new_load:.1f}s, warm mean {_mean([r['seconds'] for r in new[1:]]):.2f}s")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
