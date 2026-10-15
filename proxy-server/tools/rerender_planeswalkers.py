"""
Re-render finished planeswalkers on the planeswalker frame: each loyalty cost in its badge,
the starting loyalty in the shield. Cards made before planeswalker support used the M15 frame.

Usage (from proxy-server/):
    <PY> tools/rerender_planeswalkers.py            # previews in data/e2e/planeswalkers/
    <PY> tools/rerender_planeswalkers.py --write    # update the stored cards

--write stores each card's starting loyalty (when it has none) and swaps the new image in.
The old image stays beside it as <id>.pre-planeswalker.png. Browsers may show the old image
for up to an hour (the media route's cache time). Uses the CPU only; safe while the site runs.
"""
import argparse
import base64
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
import power_level  # noqa: E402
from storage import Storage  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):  # the renderer is chatty on import
    from card_renderer import card_renderer  # noqa: E402


def planeswalkers(storage: Storage) -> list[dict]:
    rows = storage._all("SELECT id FROM cards WHERE status = 'done' AND card_json LIKE '%laneswalker%'")
    cards = [storage.get_card(r["id"]) for r in rows]
    return [c for c in cards if card_renderer.is_planeswalker((c["card"] or {}).get("type"))]


def render(card: dict, art_path: str | None) -> bytes:
    art = Path(art_path) if art_path else None
    art_b64 = base64.b64encode(art.read_bytes()).decode() if art and art.is_file() else None
    with contextlib.redirect_stdout(io.StringIO()):
        rendered = card_renderer.generate_card_image(card, art_b64)
    if not rendered:
        raise RuntimeError("the card did not render")
    return base64.b64decode(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="update the stored cards (default: previews only)")
    args = parser.parse_args()

    data_dir = Path(config.DATA_DIR)
    storage = Storage(data_dir / "mtgenesis.db")
    preview_dir = data_dir / "e2e" / "planeswalkers"
    found = planeswalkers(storage)
    print(f"{len(found)} finished planeswalker(s)")

    for row in found:
        card = dict(row["card"])
        if not card.get("loyalty"):
            card["loyalty"] = str(power_level.starting_loyalty(card))
        png = render(card, row.get("art_path"))
        label = f"{row['id']}  {card.get('name')!r}  loyalty {card['loyalty']}"
        if not args.write:
            preview_dir.mkdir(parents=True, exist_ok=True)
            (preview_dir / f"{row['id']}.png").write_bytes(png)
            print(f"preview  {label}")
            continue
        path = Path(row["card_path"]) if row.get("card_path") else data_dir / "cards" / f"{row['id']}.png"
        backup = path.with_name(f"{path.stem}.pre-planeswalker.png")
        if path.is_file() and not backup.exists():
            backup.write_bytes(path.read_bytes())
        tmp = path.with_suffix(".rerender.tmp")
        tmp.write_bytes(png)
        storage.update_card(row["id"], card=card, card_path=path)
        tmp.replace(path)
        print(f"updated  {label}")

    if not args.write and found:
        print(f"Previews in {preview_dir}. Run with --write to update the cards.")


if __name__ == "__main__":
    main()
