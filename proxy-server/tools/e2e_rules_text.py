"""
End-to-end check of generated rules text.

Runs a fixed matrix of realistic card requests through the real pipeline
(app.createCardContent -> app.finalize_card -> card_renderer), with no artwork, and
writes the rendered cards plus a report to data/e2e/<label>/:

    report.md    per-card raw model reply, final text and lint findings, plus totals
    report.json  the same, for diffing runs
    NN-name.png  the rendered card, to eyeball legibility

Needs Ollama running. Usage (from proxy-server/):

    python tools/e2e_rules_text.py --label baseline
    python tools/e2e_rules_text.py --label qwen35 --model qwen3.5:4b --repeat 2
    python tools/e2e_rules_text.py --label quick --only 3,5,11
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _art_prompt(name, desc):
    return f"Fantasy art of {name}, {desc}, detailed digital art, Magic: The Gathering style"


# (card params as the frontend sends them, art prompt)
SPECS = [
    ({"name": "Dawnwatch Sentry", "manaCost": "{1}{W}", "colors": ["W"], "type": "Creature",
      "subtype": "Human Soldier", "rarity": "common", "cmc": 2},
     _art_prompt("Dawnwatch Sentry", "a creature human soldier, radiant and ordered, modest size")),
    ({"name": "Tidecaller Adept", "manaCost": "{2}{U}", "colors": ["U"], "type": "Creature",
      "subtype": "Merfolk Wizard", "rarity": "uncommon", "cmc": 3},
     _art_prompt("Tidecaller Adept", "a creature merfolk wizard, arcane and oceanic")),
    ({"name": "Mossgrave the Undying", "manaCost": "{2}{B}{G}", "colors": ["B", "G"], "type": "Creature",
      "supertype": "Legendary", "subtype": "Elf Druid", "rarity": "rare", "cmc": 4},
     _art_prompt("Mossgrave the Undying", "a legendary creature elf druid, infused with shadow and decay and primal growth")),
    ({"name": "Vyraxa, Ember Sovereign", "manaCost": "{4}{R}{R}", "colors": ["R"], "type": "Creature",
      "supertype": "Legendary", "subtype": "Dragon", "rarity": "mythic", "cmc": 6},
     "an ancient red dragon queen coiled atop a volcano, molten gold hoard"),
    ({"name": "Cinder Snap", "manaCost": "{R}", "colors": ["R"], "type": "Instant",
      "rarity": "common", "cmc": 1},
     _art_prompt("Cinder Snap", "an instant, a crackling burst of flame")),
    ({"name": "Tidal Unmaking", "manaCost": "{X}{U}{U}", "colors": ["U"], "type": "Sorcery",
      "rarity": "rare", "cmc": 2},
     _art_prompt("Tidal Unmaking", "a sorcery, a wave that erases a city")),
    ({"name": "Oath of the Bright Shield", "manaCost": "{1}{W}", "colors": ["W"], "type": "Enchantment",
      "subtype": "Aura", "rarity": "uncommon", "cmc": 2},
     _art_prompt("Oath of the Bright Shield", "an enchantment aura, a knight wreathed in holy light")),
    ({"name": "Blade of the Last Ember", "manaCost": "{2}", "colors": [], "type": "Artifact",
      "subtype": "Equipment", "rarity": "rare", "cmc": 2},
     _art_prompt("Blade of the Last Ember", "an artifact equipment, a sword with a dying coal in its hilt")),
    ({"name": "Rustwing Glider", "manaCost": "{3}", "colors": [], "type": "Artifact",
      "subtype": "Vehicle", "rarity": "uncommon", "cmc": 3},
     _art_prompt("Rustwing Glider", "an artifact vehicle, a patched-together flying machine")),
    ({"name": "Shattered Observatory", "manaCost": "", "colors": [], "type": "Land",
      "rarity": "rare", "cmc": 0},
     _art_prompt("Shattered Observatory", "a land, a ruined tower open to the stars")),
    ({"name": "Kaelis, Stormweaver", "manaCost": "{2}{U}{R}", "colors": ["U", "R"], "type": "Planeswalker",
      "supertype": "Legendary", "subtype": "Kaelis", "rarity": "mythic", "cmc": 4},
     _art_prompt("Kaelis, Stormweaver", "a legendary planeswalker, a mage bending lightning over the sea")),
    ({"name": "Rootmass Colossus", "manaCost": "{2}{G}{G}", "colors": ["G"], "type": "Creature",
      "subtype": "Elemental", "rarity": "rare", "cmc": 4, "power": "*", "toughness": "*"},
     _art_prompt("Rootmass Colossus", "a creature elemental, a titan of roots and soil")),
    ({"name": "Grave Bargain", "manaCost": "{2}{B}", "colors": ["B"], "type": "Sorcery",
      "rarity": "common", "cmc": 3},
     _art_prompt("Grave Bargain", "a sorcery, a hooded figure trading with the dead")),
    ({"name": "Banner of the Charging Host", "manaCost": "{2}{R}{W}", "colors": ["R", "W"], "type": "Enchantment",
      "rarity": "uncommon", "cmc": 4},
     _art_prompt("Banner of the Charging Host", "an enchantment, a war banner over a cavalry charge")),
]


def slug(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--model', help='sets MTG_TEXT_MODEL for this run')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--only', help='comma-separated 1-based spec numbers')
    ap.add_argument('--from-db', action='store_true',
                    help='use the distinct card requests stored in data/mtgenesis.db instead of SPECS')
    args = ap.parse_args()

    if args.from_db:
        import sqlite3
        db = sqlite3.connect(f"file:{ROOT / 'data' / 'mtgenesis.db'}?mode=ro", uri=True)
        seen, specs = set(), []
        for prompt, params in db.execute('select prompt, card_params_json from cards order by created_at'):
            card = json.loads(params)
            key = (card.get('name'), card.get('type'), card.get('rarity'))
            if key not in seen:
                seen.add(key)
                specs.append((card, prompt))
        SPECS[:] = specs

    if args.model:
        os.environ['MTG_TEXT_MODEL'] = args.model

    import app  # heavy import (torch, diffusers); after env is set
    from rules_text import lint_rules_text

    # Record every raw model reply, whichever Ollama call the pipeline uses.
    raw_log = []
    for fn_name in ('generate', 'chat'):
        original = getattr(app.ollama_client, fn_name)

        def wrapped(*a, _orig=original, _name=fn_name, **k):
            resp = _orig(*a, **k)
            try:
                raw = resp['response'] if _name == 'generate' else resp['message']['content']
            except Exception:
                raw = repr(resp)
            raw_log.append(raw)
            return resp

        setattr(app.ollama_client, fn_name, wrapped)

    only = {int(x) for x in args.only.split(',')} if args.only else None
    out = ROOT / 'data' / 'e2e' / args.label
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for rep in range(args.repeat):
        for i, (card, prompt) in enumerate(SPECS, 1):
            if only and i not in only:
                continue
            raw_log.clear()
            t0 = time.time()
            text = app.createCardContent(prompt, dict(card))
            gen_s = time.time() - t0
            final, png = app.finalize_card(dict(card), text, None)
            desc = final.get('description') or ''
            issues = lint_rules_text(desc, final) if text else [('error', 'generation failed (no text)')]
            fname = f'{i:02d}-{rep}-{slug(card["name"])}.png'
            if png:
                (out / fname).write_bytes(base64.b64decode(png))
            results.append({
                'n': i, 'rep': rep, 'name': card['name'], 'type': app_type(card),
                'rarity': card['rarity'], 'raw': list(raw_log), 'final': desc,
                'pt': f"{final.get('power', '')}/{final.get('toughness', '')}" if 'creature' in card['type'].lower() else '',
                'issues': issues, 'seconds': round(gen_s, 1), 'png': fname,
            })
            e = sum(1 for s, _ in issues if s == 'error')
            w = len(issues) - e
            print(f'[{args.label}] {i:02d}.{rep} {card["name"]}: {e} errors, {w} warnings, {gen_s:.1f}s')

    write_report(out, args, results)


def app_type(card):
    parts = [card.get('supertype') or '', card.get('type') or '']
    head = ' '.join(p for p in parts if p)
    return f"{head} — {card['subtype']}" if card.get('subtype') else head


def write_report(out, args, results):
    errors = sum(1 for r in results for s, _ in r['issues'] if s == 'error')
    warns = sum(1 for r in results for s, _ in r['issues'] if s == 'warn')
    clean = sum(1 for r in results if not any(s == 'error' for s, _ in r['issues']))
    lines = [l for r in results for l in r['final'].split('\n') if l.strip()]
    openings = Counter(' '.join(l.split()[:3]).lower() for l in lines)
    chars = [len(r['final']) for r in results]
    secs = [r['seconds'] for r in results]
    summary = {
        'label': args.label, 'model': os.environ.get('MTG_TEXT_MODEL', '(pipeline default)'),
        'cards': len(results), 'clean_cards': clean, 'errors': errors, 'warnings': warns,
        'avg_chars': round(sum(chars) / max(1, len(chars))),
        'avg_seconds': round(sum(secs) / max(1, len(secs)), 1),
        'most_repeated_openings': openings.most_common(6),
    }
    (out / 'report.json').write_text(json.dumps({'summary': summary, 'results': results}, indent=2, ensure_ascii=False), encoding='utf-8')

    md = [f"# Rules text e2e: {args.label}", '',
          f"Model: `{summary['model']}` · cards: {summary['cards']} · clean (no errors): {clean} · "
          f"errors: {errors} · warnings: {warns} · avg {summary['avg_chars']} chars · avg {summary['avg_seconds']}s",
          '', f"Most repeated openings: {summary['most_repeated_openings']}", '']
    for r in results:
        md += [f"## {r['n']:02d}.{r['rep']} {r['name']} ({r['rarity']} {r['type']}) {r['pt']}", '',
               f"![card]({r['png']})", '', '**Final text**', '', '```', r['final'], '```', '']
        if r['issues']:
            md += ['**Lint**', ''] + [f"- {s}: {m}" for s, m in r['issues']] + ['']
        md += ['<details><summary>Raw model replies</summary>', '', '```']
        md += [x for x in r['raw']] + ['```', '</details>', '']
    (out / 'report.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
