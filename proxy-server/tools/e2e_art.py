"""
End-to-end check of generated card art.

Runs a fixed matrix of card requests through the real gallery pipeline (rules text via
app.createCardContent and art via image_generation.generate_art in parallel, like
GenerationQueue, then app.finalize_card) with fixed seeds, and writes to data/e2e/<label>/:

    report.md    per-card prompt, seconds and exposure stats, plus totals
    report.json  the same, for diffing runs
    art.png      contact sheet: one row per card, one column per seed
    cards.png    contact sheet of the rendered cards (first seed only)
    NN-S-name-art.png / NN-S-name-card.png

Exposure stats: white = % of pixels with every channel >= 240 (blown highlights),
lstd = luminance standard deviation (contrast), sat = mean HSV saturation.

Needs a GPU (and Ollama unless --art-only). Loads its own SDXL pipeline, so don't run it
while people are generating cards on the live site. Usage (from proxy-server/):

    python tools/e2e_art.py --label baseline
    python tools/e2e_art.py --label quick --art-only --seeds 2 --only 2,6,9
    python tools/e2e_art.py --label paint --art-only --set ART_STYLE="oil painting, ..."
    python tools/e2e_art.py --label lora --art-only --lora ntc-ai/SDXL-LoRA-slider.oil-painting "oil painting.safetensors" 2
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (card params as the frontend sends them, art prompt as card-form generateArtPromptText builds it).
# Weighted toward people: humans drawn as horned spirits and bare chests were the reported bugs.
SPECS = [
    ({"name": "Dawnwatch Sentry", "colors": ["W"], "type": "Creature", "subtype": "Human Soldier",
      "rarity": "common", "cmc": 2}, "Dawnwatch Sentry, a human soldier, modest size"),
    ({"name": "Ligma", "colors": ["B", "G"], "type": "Creature", "supertype": "Legendary",
      "subtype": "Human", "rarity": "rare", "cmc": 3}, "Ligma, a legendary human, medium scale"),
    ({"name": "Jerry", "colors": ["W"], "type": "Creature", "supertype": "Legendary",
      "subtype": "Human", "rarity": "mythic", "cmc": 4}, "Jerry, a legendary human, medium scale"),
    ({"name": "Sera the Wise", "colors": ["U"], "type": "Creature", "supertype": "Legendary",
      "subtype": "Human Wizard", "rarity": "rare", "cmc": 3}, "Sera the Wise, a legendary human wizard, medium scale"),
    ({"name": "Grimsby", "colors": ["B"], "type": "Creature", "subtype": "Human Rogue",
      "rarity": "common", "cmc": 2}, "Grimsby, a human rogue, modest size"),
    ({"name": "Kara Flamecrest", "colors": ["R"], "type": "Creature", "subtype": "Human Warrior",
      "rarity": "uncommon", "cmc": 3}, "Kara Flamecrest, a human warrior, medium scale"),
    ({"name": "Ser Aldric", "colors": ["R", "W"], "type": "Creature", "subtype": "Knight",
      "rarity": "uncommon", "cmc": 4}, "Ser Aldric, a knight, medium scale"),
    ({"name": "Brother Oak", "colors": ["G"], "type": "Creature", "subtype": "Human Monk",
      "rarity": "common", "cmc": 2}, "Brother Oak, a human monk, modest size"),
    ({"name": "Grom the Unbroken", "colors": ["R"], "type": "Creature", "supertype": "Legendary",
      "subtype": "Human Berserker", "rarity": "rare", "cmc": 5}, "Grom the Unbroken, a legendary human berserker, large and imposing"),
    ({"name": "Mossgrave the Undying", "colors": ["B", "G"], "type": "Creature", "supertype": "Legendary",
      "subtype": "Elf Druid", "rarity": "rare", "cmc": 4}, "Mossgrave the Undying, a legendary elf druid, medium scale"),
    ({"name": "Countess Vael", "colors": ["B"], "type": "Creature", "supertype": "Legendary",
      "subtype": "Vampire Noble", "rarity": "mythic", "cmc": 4}, "Countess Vael, a legendary vampire noble, medium scale"),
    ({"name": "Vyraxa, Ember Sovereign", "colors": ["R"], "type": "Creature", "supertype": "Legendary",
      "subtype": "Dragon", "rarity": "mythic", "cmc": 6}, "Vyraxa, Ember Sovereign, a legendary dragon, large and imposing"),
    ({"name": "Suma deez", "colors": ["R"], "type": "Creature", "supertype": "Legendary",
      "subtype": "Beast", "rarity": "rare", "cmc": 4}, "Suma deez, a legendary beast, medium scale"),
    ({"name": "Kaelis, Stormweaver", "colors": ["U", "R"], "type": "Planeswalker", "supertype": "Legendary",
      "subtype": "Kaelis", "rarity": "mythic", "cmc": 4}, "Kaelis, Stormweaver, a legendary planeswalker kaelis"),
    ({"name": "Cinder Snap", "colors": ["R"], "type": "Instant", "rarity": "common", "cmc": 1},
     "Cinder Snap, an instant"),
    ({"name": "Oath of the Bright Shield", "colors": ["W"], "type": "Enchantment", "subtype": "Aura",
      "rarity": "uncommon", "cmc": 2}, "Oath of the Bright Shield, an aura enchantment"),
    ({"name": "Blade of the Last Ember", "colors": [], "type": "Artifact", "subtype": "Equipment",
      "rarity": "rare", "cmc": 2}, "Blade of the Last Ember, an equipment artifact"),
    ({"name": "Shattered Observatory", "colors": [], "type": "Land", "rarity": "rare", "cmc": 0},
     "Shattered Observatory, a land"),
]
SEED_BASE = 1000


def slug(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:32]


def exposure(png: bytes) -> dict:
    import numpy as np
    from PIL import Image
    img = Image.open(io.BytesIO(png)).convert("RGB")
    a = np.asarray(img).astype(float)
    hsv = np.asarray(img.convert("HSV")).astype(float)
    return {"white": round(100 * float((a.min(2) >= 240).mean()), 2),
            "lstd": round(float(a.mean(2).std()), 1),
            "sat": round(float(hsv[..., 1].mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--seeds', type=int, default=2, help='renders per card (seeds SEED_BASE+n*100+s)')
    ap.add_argument('--only', help='comma-separated 1-based spec numbers')
    ap.add_argument('--art-only', action='store_true', help='skip rules text (no Ollama) and card rendering')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                    help='override a string constant in image_generation (e.g. ART_STYLE, HUMAN_CONTEXT)')
    ap.add_argument('--lora', action='append', nargs=3, default=[], metavar=('REPO', 'FILE', 'SCALE'),
                    help='fuse a style LoRA instead of config.IMAGE_LORAS (repeatable; --lora none none 0 for none)')
    args = ap.parse_args()

    import torch
    import config
    import image_generation as ig
    overrides = {}
    if args.lora:
        config.IMAGE_LORAS = [(r, f, float(s)) for r, f, s in args.lora if r != 'none']
        overrides['IMAGE_LORAS'] = config.IMAGE_LORAS
    for item in args.set:
        name, _, value = item.partition('=')
        if not isinstance(getattr(ig, name, None), str):
            ap.error(f'--set: image_generation.{name} is not a string constant')
        setattr(ig, name, value)
        overrides[name] = value
    if not args.art_only:
        import app  # heavy import; after the overrides so nothing re-reads them

    only = {int(x) for x in args.only.split(',')} if args.only else None
    out = ROOT / 'data' / 'e2e' / args.label
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for n, (card, prompt) in enumerate(SPECS, 1):
        if only and n not in only:
            continue
        for s in range(args.seeds):
            seed = SEED_BASE + n * 100 + s
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                text_future = None if args.art_only else pool.submit(app.createCardContent, prompt, dict(card))

                def paint():
                    torch.manual_seed(seed)  # generate_art uses the global RNG
                    return ig.generate_art(prompt, dict(card))
                art_future = pool.submit(paint)
                art_b64 = art_future.result()
                art_s = time.time() - t0
                text = text_future.result() if text_future else None
            total_s = time.time() - t0

            count = lambda t: len(ig._pipeline.tokenizer(t).input_ids) - 2  # noqa: E731
            positive, negative = ig.build_art_prompt(prompt, card, count)
            base = f'{n:02d}-{s}-{slug(card["name"])}'
            art_png = base64.b64decode(art_b64)
            (out / f'{base}-art.png').write_bytes(art_png)
            card_file = None
            if not args.art_only:
                final, card_b64 = app.finalize_card(dict(card), text, art_b64)
                if card_b64:
                    card_file = f'{base}-card.png'
                    (out / card_file).write_bytes(base64.b64decode(card_b64))
            r = {'n': n, 'seed': seed, 'name': card['name'], 'subtype': card.get('subtype', ''),
                 'prompt': positive, 'prompt_tokens': count(positive), 'art': f'{base}-art.png',
                 'card': card_file, 'text_ok': bool(text) if not args.art_only else None,
                 'art_seconds': round(art_s, 1), 'total_seconds': round(total_s, 1), **exposure(art_png)}
            results.append(r)
            print(f"[{args.label}] {n:02d}.{s} {card['name']}: white {r['white']}% lstd {r['lstd']} "
                  f"sat {r['sat']} art {art_s:.1f}s total {total_s:.1f}s"
                  + ('' if args.art_only else f" text {'ok' if text else 'FAILED'}"), flush=True)

    write_report(out, args, overrides, negative, results)


def contact_sheet(out, results, key, name, cols, cell):
    from PIL import Image, ImageDraw
    rows = sorted({r['n'] for r in results if r[key]})
    if not rows:
        return
    w, h = cell
    sheet = Image.new('RGB', (w * cols + 170, h * len(rows)), 'white')
    draw = ImageDraw.Draw(sheet)
    for y, n in enumerate(rows):
        items = [r for r in results if r['n'] == n and r[key]][:cols]
        draw.text((6, y * h + 6), f"{n:02d} {items[0]['name'][:22]}\n{items[0]['subtype'][:24]}", fill='black')
        for x, r in enumerate(items):
            sheet.paste(Image.open(out / r[key]).convert('RGB').resize((w, h)), (170 + x * w, y * h))
    sheet.save(out / name)


def write_report(out, args, overrides, negative, results):
    def avg(k):
        return round(sum(r[k] for r in results) / max(1, len(results)), 2)
    summary = {'label': args.label, 'renders': len(results), 'overrides': overrides,
               'negative_prompt': negative, 'avg_white': avg('white'), 'avg_lstd': avg('lstd'),
               'avg_sat': avg('sat'), 'max_white': max((r['white'] for r in results), default=0),
               'avg_art_seconds': avg('art_seconds'), 'avg_total_seconds': avg('total_seconds'),
               'text_failures': sum(1 for r in results if r['text_ok'] is False)}
    (out / 'report.json').write_text(json.dumps({'summary': summary, 'results': results}, indent=2,
                                                ensure_ascii=False), encoding='utf-8')
    contact_sheet(out, results, 'art', 'art.png', args.seeds, (272, 224))
    contact_sheet(out, results, 'card', 'cards.png', 1, (250, 349))

    md = [f"# Art e2e: {args.label}", '',
          f"Renders: {summary['renders']} · avg white {summary['avg_white']}% (max {summary['max_white']}%) · "
          f"avg lstd {summary['avg_lstd']} · avg sat {summary['avg_sat']} · "
          f"art {summary['avg_art_seconds']}s · total {summary['avg_total_seconds']}s · "
          f"text failures {summary['text_failures']}", '',
          f"Overrides: `{overrides or 'none'}`", '', f"Negative prompt: `{negative}`", '',
          '![art contact sheet](art.png)', '']
    for r in results:
        md += [f"## {r['n']:02d} seed {r['seed']} {r['name']} ({r['subtype']})", '',
               f"white {r['white']}% · lstd {r['lstd']} · sat {r['sat']} · art {r['art_seconds']}s · "
               f"{r['prompt_tokens']} tokens", '', f"`{r['prompt']}`", '', f"![art]({r['art']})", '']
    (out / 'report.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
