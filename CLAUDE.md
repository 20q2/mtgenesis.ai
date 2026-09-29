# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MTGenesis.AI generates custom Magic: The Gathering cards. An Angular 16 + Angular Material frontend collects card properties; a Flask backend generates rules text with Ollama (`mistral:latest`) and artwork with a local Stable Diffusion pipeline (diffusers), then composites a finished card PNG with Pillow.

## Commands

Local dev needs three processes:

```bash
ollama serve                         # Ollama on :11434 (needs `ollama pull mistral:latest`)
cd proxy-server && python app.py     # Flask on :5000
npm run start                        # Angular dev server on :4200
```

- Backend deps: `pip install -r proxy-server/requirements.txt` (pins CPU `torch==2.8.0+cpu`; a CUDA build of torch must be installed separately to use GPU mode).
- Production build: `npm run build` (uses `environment.prod.ts`).
- Frontend tests (Karma/Jasmine, Chrome): `npm test`; single spec: `npx ng test --include src/app/components/card-form/card-form.component.spec.ts`. Existing specs are CLI scaffolds.
- Renderer smoke test (no Flask/Ollama needed): `python test_colored_artifact.py` from the repo root. It imports `proxy-server/card_renderer.py` directly and renders a card.
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

### Backend (`proxy-server/app.py`, one ~3.5k-line file)

- **Config constants at the top of the file**, not env vars:
  - `USE_CUDA` selects GPU or CPU.
  - `MODEL_SIZE` picks the image model: `heavy` = sdxl-turbo, `medium` = SD 1.5, `light` = SD 1.4, `placeholder` = no image generation (handy for fast iteration).
  - Timeout constants (`COLD_START_TIMEOUT`, etc.).
  - xformers is disabled through env vars that must be set before `diffusers` is imported.
- **The image pipeline loads lazily** in `get_image_pipeline()`. `_models_loaded` tracks whether the cold-start or warm timeout applies.
- **Rules-text generation (`createCardContent`)** is mostly prompt engineering plus defensive cleanup. It builds a long constrained prompt from `cardData` (CMC, colors, type, rarity, asterisk P/T), asks the LLM for abilities as quoted strings, then:
  - retries up to 3 times through `strip_non_rules_text` / `validate_rules_text`
  - runs the output through a chain of sanitizers: `sanitize_*_abilities`, `apply_universal_complexity_limits`, `reorder_abilities_properly` (keywords → triggered → activated), `limit_creature_active_abilities`, etc.

  Most "bad card text" bugs are fixed in one of these helpers.
- **CORS/ngrok headers:** every response goes through `add_ngrok_headers`. New routes should do the same and handle `OPTIONS`.

### Card renderer (`proxy-server/card_renderer.py`)

`MagicCardRenderer` is a singleton (`card_renderer`) that layers PNG assets from `proxy-server/assets/` (Card Conjurer–style frames: `m15/`, `frames/`, `manaSymbols/`, fonts, etc.):

- selects the frame from color identity and type (`load_base_frame`)
- handles special cases: colored artifacts (masked blend), two-color pinlines and crowns, legendary crowns, P/T boxes
- draws the name, type line and mana cost, plus rules text with inline mana symbols (`draw_text_with_mana_symbols`) and font auto-sizing
- adds a rarity-colored site logo

Asset images are cached in memory. `cairosvg`/`wand` are optional imports, and `convert_svg_to_png.py` is a one-off tool for converting mana-symbol SVGs.

### Deployment

- Production runs the Flask server on a local GPU machine, exposed through ngrok.
- The frontend's prod `apiUrl` comes from `src/environments/config.ts`, currently a hard-coded ngrok URL. `lambda-proxy/lambda_function.py` holds a second hard-coded copy.
- **When the ngrok URL changes, update both.**
- `lambda-proxy/` is an optional AWS Lambda + API Gateway CORS proxy in front of ngrok (see `lambda-proxy/deploy.md`).
