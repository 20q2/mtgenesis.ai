# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MTGenesis.AI generates custom Magic: The Gathering cards. An Angular 16 + Angular Material frontend collects card properties; a Flask backend generates rules text with Ollama (`qwen3:8b`, set by `TEXT_MODEL` in `proxy-server/config.py`) and artwork with a local Stable Diffusion pipeline (diffusers), then composites a finished card PNG with Pillow.

## Commands

Local dev needs three processes:

```bash
ollama serve                         # Ollama on :11434 (needs `ollama pull qwen3:8b`)
cd proxy-server && python app.py     # Flask on :5000
npm run start                        # Angular dev server on :4200
```

- Backend deps: `pip install -r proxy-server/requirements.txt` (pins CPU `torch==2.8.0+cpu`; a CUDA build of torch must be installed separately to use GPU mode).
- Production build: `npm run build` (uses `environment.prod.ts`).
- Frontend tests (Karma/Jasmine, Chrome): `npm test`; single spec: `npx ng test --include src/app/components/card-form/card-form.component.spec.ts`. Existing specs are CLI scaffolds.
- Renderer smoke test (no Flask/Ollama needed): `python test_colored_artifact.py` from the repo root. It imports `proxy-server/card_renderer.py` directly and renders a card.
- Backend tests: `python -m pytest tests` from `proxy-server/` (the `slow` ones import app.py and torch).
- Rules-text e2e (needs Ollama): `python tools/e2e_rules_text.py --label <name> [--model M] [--repeat N] [--from-db]` from `proxy-server/`. It runs a fixed matrix of card requests (or the stored ones) through the real pipeline and writes rendered PNGs plus `report.md` (raw replies, final text, lint findings) to `data/e2e/<name>/`. Use it after any prompt or cleanup change.
- No linter or Python test framework is configured.

## Known issue: missing frontend models

Every service and component imports from `src/app/models/` (`card.model.ts`, `api.model.ts`), but that directory is not in the repo. The `models/` rule in `.gitignore` (meant for downloaded AI weights) matches it. A fresh checkout won't compile until those files are recreated and the ignore rule is narrowed (e.g. to `/models/`).

## Architecture

### Request flow

1. `CardFormComponent` → `AppComponent` → `CardService` POSTs `{prompt, width, height, cardData}` to `POST /api/v1/create_card`. `HealthService` polls `GET /health` every 3s, and `CardService` picks the cold-start or warm-run timeout from `environment.*TimeoutMs`.
2. `create_card` in `proxy-server/app.py` enqueues the job on the global `RequestQueue` (worker thread, max 2 concurrent, periodic cleanup), then blocks and polls it until done. `create_card_async` + `card_status/<id>` are the same flow for clients that poll instead.
3. `process_card_generation` runs `createCardImage` (diffusers) and `createCardContent` (Ollama) in parallel in a `ThreadPoolExecutor`. An image timeout cancels the whole request. Other failures degrade to a partial result with a `warning`.
4. Post-processing merges the LLM output into `cardData`:
   - replaces `~` with the card name
   - cleans formatting
   - generates creature P/T (`generate_creature_stats`) and Vehicle crew cost if missing
5. `card_renderer.generate_card_image(card_data, artwork_base64)` composites the final card.
6. The response contains `cardData` (rules text), `imageData` (base64 art), `card_image` (base64 rendered card) and `generation_time`.

### Backend (`proxy-server/app.py`, ~1.3k lines, plus `rules_text.py`, `image_generation.py`, `api_routes.py`)

- **Config constants live in `proxy-server/config.py`** (imported with `from config import *`):
  - `USE_CUDA` selects GPU or CPU.
  - `MODEL_SIZE`: `placeholder` = no image generation (gray art; handy for fast iteration); anything else uses `IMAGE_MODEL_ID` (an SDXL fine-tune).
  - `TEXT_MODEL` (Ollama rules-text model, `MTG_TEXT_MODEL` env override), `TEXT_ATTEMPTS`, `TEXT_THINK`.
  - Timeout constants (`COLD_START_TIMEOUT`, etc.).
  - xformers is disabled through env vars that must be set before `diffusers` is imported.
- **The image pipeline loads lazily** in `get_image_pipeline()`. `_models_loaded` tracks whether the cold-start or warm timeout applies.
- **Rules text lives in `proxy-server/rules_text.py`**; `createCardContent` in app.py just calls `generate_rules_text`:
  - `build_messages` sends a system prompt (Oracle templating rules, card-type rules, few-shot examples) plus the card's facts, an ability budget by rarity, type-specific requirements, and one random color "design hook" for variety.
  - The model answers in JSON (`{"abilities": [...]}` via Ollama's `format` schema), one ability per item, with `think=False` and `num_ctx` 2560. **Keep the context small**: at 4096 the 8B model grows enough that SDXL spills out of the 12 GB GPU and art goes from ~5 s to ~120 s.
  - `clean_abilities` turns that list into legal, legible templating (symbols, modern wording, self-references, keyword line in rules order, card-type rules, */* definitions, rarity cap). `lint_rules_text` scores the result; up to `TEXT_ATTEMPTS` replies are tried and the one with the fewest lint errors wins.
  - Most "bad card text" bugs are a new fixer in `_fix_templating`/`clean_abilities` or a new lint rule. Add a case to `tests/test_rules_text.py` (they're real model outputs) and re-run `tools/e2e_rules_text.py`.
  - **Power level (`proxy-server/power_level.py`)** keeps cheap cards from being mega strong:
    - `creature_stats` is the stat curve. The prompt states the body up front ("2/2 (fixed)") and `finalize_card` prints the same stats.
    - `budget` is mana value plus a rarity bonus, minus the body's worth. `describe_budget` turns it into a concrete prompt line ("one small bonus, like ...").
    - `estimate` prices rules text in mana using Limited rates. Repeating triggers and free {T} abilities cost 2–3× a one-shot effect.
    - Generation retries anything more than `WARN_OVER` above its budget and keeps the weakest valid attempt. As a last resort, `trim_to_budget` drops the priciest ability. `FORBIDDEN` patterns in rules_text.py (repeating removal, free spells, filler mana) are never printed.
    - To make cards stronger or weaker overall, tune `RARITY_BONUS`, `STAT_TOTAL` or the rates in `ability_value`, then compare `over_budget_*` in e2e reports.
  - `finalize_card` still replaces `~`, fixes bullets and periods (keyword lines get no period), and generates missing creature/Vehicle P/T and Vehicle crew. Stats are a deterministic curve by mana value and rarity; `*` P/T only comes from the request.
- **CORS/ngrok headers:** every response goes through `add_ngrok_headers`. New routes should do the same and handle `OPTIONS`.

### Knowledge Pool

Shared custom cards for the paper event: `/pool` in the app. The spec is `docs/superpowers/specs/2026-09-29-knowledge-pool-design.md`.

- The host opens a pool from `/admin` (`PoolAdminComponent`). It has a per-player submission cap and a list of slots, each with a color rule and a type rule; the default is 16 slots for 8 players.
- Players submit finished gallery cards into slots and vote once per slot (never for their own card). Submitters stay anonymous until close. The top card in each slot becomes legal.
- Storage: the `pools` / `pool_slots` / `pool_entries` / `pool_votes` tables and methods in `storage.py`. `card_fits_slot` is mirrored by `cardFitsSlot` in `pool.service.ts`, so keep the two in sync.
- Routes: `/pools/*` and `/admin/pools*` in `api_routes.py`. Each entry carries an advisory `power` check from `power_level.assess`.
- Tests: `tests/test_pool.py`.

### Card renderer (`proxy-server/card_renderer.py`)

`MagicCardRenderer` is a singleton (`card_renderer`) that layers PNG assets from `proxy-server/assets/` (Card Conjurer–style frames: `m15/`, `frames/`, `manaSymbols/`, fonts, etc.):

- selects the frame from color identity and type (`load_base_frame`)
- handles special cases: colored artifacts (masked blend), two-color pinlines and crowns, legendary crowns, P/T boxes
- draws the name, type line and mana cost, plus rules text with inline mana symbols (`draw_text_with_mana_symbols`) and font auto-sizing
- adds a rarity-colored site logo

Asset images are cached in memory. `cairosvg`/`wand` are optional imports, and `convert_svg_to_png.py` is a one-off tool for converting mana-symbol SVGs.

### Deployment

- Production runs the Flask server on a local GPU machine, exposed through ngrok. The frontend is served from GitHub Pages (`gh-pages` branch, https://20q2.github.io/mtgenesis.ai/).
- `Start MTGenesis.cmd` → `deploy/start-site.ps1` does the whole launch:
  - starts Ollama, Flask and ngrok (reusing any already running)
  - pushes `api-config.json` with the live tunnel URL to `gh-pages`
  - rebuilds the app with `--base-href /mtgenesis.ai/` only when the frontend source differs from the commit recorded in `gh-pages/build-info.json`
  - uses `.deploy/` (gitignored) as its scratch space
- `src/main.ts` fetches `api-config.json` before bootstrapping in production and overrides `environment.apiUrl`. `src/environments/config.ts` is only the build-time fallback.
- Because the site lives under a subpath, asset URLs must be relative (`assets/...`, not `/assets/...`).
- `lambda-proxy/` is an optional AWS Lambda + API Gateway CORS proxy in front of ngrok (see `lambda-proxy/deploy.md`). The launcher doesn't use or update it; its `NGROK_URL` is hard-coded.
