# AI Night: Users, Commander Sets, Voting, Queue Visibility & Image Quality

**Status:** Approved in brainstorming 2026-09-28; awaiting written-spec review.

## 1. Purpose

MTGenesis.AI is used at a recurring "AI Night". Each player brings 3 AI-generated versions of the same commander, and the group votes on which version becomes legal at the table.

This work adds:
- lightweight identity
- 3-card commander sets with per-slot rerolls
- lock-in and a shared voting page
- persistent storage
- visible queue status
- a real improvement in art quality on the new host machine

Free-play generation must keep working and never touches the lineup.

### Success criteria

- A player logs in with only a username from their own device.
- They generate 3 versions of a commander from one prompt, reroll any slot, and lock in the set under a commander name.
- Everyone sees every locked set on `/vote` and votes once per set, own set included. They can change their vote until the host closes the event.
- The leading version of each set is unmistakable.
- After close, the winners (one per set) are displayed and preserved as history.
- Users can always see whether the server is busy and where their cards are in the queue.
- Card art is visibly better than SDXL-Turbo at 512px, judged by the owner on a before/after comparison page.

### Decisions made

| Topic | Decision |
|---|---|
| Devices | Everyone uses their own device against one host server. Shared state lives server-side. Browser localStorage only remembers who is logged in. |
| Vote model | Best version per player: each set votes independently, and the top card in each set wins. |
| Vote rules | One vote per voter per set. The vote can be changed while the event is open. Self-votes are allowed. Live counts are visible. |
| Lifecycle | Named events with host controls. One event open at a time. Closing freezes votes. Past events stay viewable. |
| Admin auth | A PIN in a config constant, sent as the `X-Admin-Pin` header. |
| Image model | Fantasy-tuned SDXL (`Lykon/dreamshaper-xl-lightning`) at high resolution. |
| Architecture | Approach A: SQLite plus PNG files on disk, a job queue the frontend polls, and new backend code in separate modules. |

### Non-goals

- Passwords or real authentication. Impersonation is acceptable.
- Multiple concurrent events.
- FLUX or model switching.
- Changes to rules-text generation quality, which the owner is happy with.

## 2. Host environment

- **Hardware:** RTX 5070 (12 GB VRAM, Blackwell sm_120), i7-14700F, 32 GB RAM, Windows 11.
- **Not yet installed:** Python and Ollama. These must be installed before backend work can run.
- **PyTorch:** the 5070 requires a CUDA 12.8+ build of torch (≥ 2.7, `cu128` wheels). `proxy-server/requirements.txt` currently pins `torch==2.8.0+cpu` and must change.
- **xformers:** drop it. Use torch's built-in SDPA attention.
- **VRAM budget:** Mistral 7B (about 4.5 GB) and the SDXL fp16 pipeline (about 7 GB) must coexist.
  - Ollama is called with `keep_alive` so Mistral stays resident.
  - The pipeline enables VAE slicing.

## 3. Storage

All storage lives under `proxy-server/data/`, which is gitignored and created on startup:

- `mtgenesis.db` is a SQLite database using the stdlib `sqlite3` module, with WAL mode and one connection per request/thread.
- `cards/<card_id>.png` holds the fully rendered card.
- `art/<card_id>.png` holds the raw artwork.

IDs are UUID4 strings. Timestamps are ISO-8601 UTC strings.

### Tables

**`users`**
- `id`
- `username`: unique, compared case-insensitively. Trimmed, 1–24 characters, `[A-Za-z0-9 _-]`.
- `created_at`

**`events`**
- `id`
- `name`
- `status`: `open` | `closed`
- `created_at`, `closed_at`

**`sets`**
- `id`
- `user_id`
- `event_id`: NULL while a draft; assigned at lock
- `commander_name`, `prompt`
- `card_params_json`: the form's card data used for all 3 slots
- `status`: `draft` | `locked` | `abandoned`
- `created_at`, `locked_at`

**`cards`**
- `id`, `user_id`
- `set_id`: nullable. NULL means free play.
- `slot`: 1–3 or NULL
- `replaced`: bool. True once a reroll superseded it.
- `prompt`
- `card_json`: final card data after post-processing
- `art_path`, `card_path`
- `status`: `queued` | `generating` | `rendering` | `done` | `failed`
- `text_ready`, `art_ready`: bools. Text and art run concurrently while `generating`.
- `error`
- `created_at`, `finished_at`

**`votes`**
- `voter_id`, `set_id`, `card_id`, `created_at`
- Unique on `(voter_id, set_id)`. Voting again overwrites.

### Rules

- **Events:**
  - At most one event is `open`.
  - Creating an event while one is open is rejected with 409.
  - Closing sets `closed_at` and freezes the event: no votes, no lock or unlock changes.
- **Sets:**
  - A set's current cards are its rows with `replaced = false`, one per slot.
  - A user has at most one active draft. Starting a new set marks any existing draft `abandoned`; its cards remain in the gallery.
  - A user has at most one locked set per event. Locking a second one returns 409; the first must be unlocked.
  - A user's **current set** is their `draft`, or else their set `locked` in the open event.
  - **Lock** requires:
    - an open event
    - all 3 current cards with `done` status
    - the caller is the set's owner
    - a non-empty `commander_name`

    Locking sets `event_id` to the open event.
  - **Unlock** is allowed only while the event is open. It deletes all votes on that set and returns it to `draft` with `event_id` NULL.
- **Rerolls:**
  - A reroll is allowed on draft sets only, for the owner only.
  - It marks the old card `replaced = true` (it stays in the gallery) and creates a new queued card in the same slot.
- **Votes:**
  - The target card must belong to the set, be current, and the set must be locked in an open event.
- **Leader:**
  - The leader of a set is the card with the most votes.
  - If several cards share the top count, all of them are marked `tied`.
  - With zero votes there is no leader.
  - At close, the winner is the leader. If still tied, all tied cards are reported as tied winners and the host decides at the table.

## 4. Backend API

All new routes are in a Flask Blueprint in `proxy-server/api_routes.py`, registered by `app.py` under `/api/v1`. The API follows these rules:

- Every response goes through `add_ngrok_headers`, and every route answers `OPTIONS`.
- JSON is camelCase.
- User-scoped routes require the `X-User-Id` header. A missing or unknown ID returns 401.
- Errors return `{ "error": string }` with an appropriate 4xx/5xx status.
- The existing `/api/v1/create_card`, `/create_card_async`, `/card_status`, `/create_card_sync` and `/health` routes stay unchanged.

### Shared shapes

**`CardView`**

```jsonc
{
  "id", "userId", "setId", "slot",
  "status", "error", "textReady", "artReady",
  "queuePosition", "etaSeconds",       // null once the art is ready or the card has finished
  "card",                              // CardData after post-processing, or null
  "cardImageUrl", "artImageUrl",       // "/api/v1/media/cards/<id>.png" etc., or null
  "createdAt"
}
```

**`SetView`**

```jsonc
{
  "id", "userId", "username", "eventId",
  "commanderName", "prompt", "status", "lockedAt",
  "cards": [CardView x3 ordered by slot, each with "votes": number, "leader": bool, "tied": bool],
  "myVoteCardId"                       // present when the X-User-Id header is sent
}
```

**`EventView`**

```jsonc
{ "id", "name", "status", "createdAt", "closedAt", "sets": [SetView, locked only] }
```

### Endpoints

| Method and path | Body | Returns |
|---|---|---|
| `POST /users/login` | `{username}` | `{id, username}`. Creates the user if new. |
| `GET /me/cards` | | `CardView[]`, all of my cards newest first, replaced ones included |
| `GET /me/sets/current` | | My current set (see section 3) as a `SetView`, or `null` |
| `POST /generations` | `{prompt, cardData, count: 1\|3, commanderName?}` | `{setId?, cards: CardView[]}` (see notes) |
| `POST /cards/<id>/reroll` | | New `CardView` |
| `GET /cards/<id>` | | `CardView` |
| `POST /sets/<id>/lock` | `{commanderName?}` | `SetView` |
| `POST /sets/<id>/unlock` | | `SetView` |
| `GET /events/current` | | `EventView` or `null` |
| `GET /events` | | `[{id, name, status, createdAt, closedAt}]` newest first |
| `GET /events/<id>` | | `EventView` |
| `POST /votes` | `{setId, cardId}` | Updated `SetView` |
| `GET /queue_status` | | `{busy, cardsAhead, generatingNow, avgImageSeconds, etaSeconds}` (see notes) |
| `GET /media/cards/<id>.png`, `GET /media/art/<id>.png` | | PNG file |
| `POST /admin/events` | `{name}` | `EventView`. Requires `X-Admin-Pin`. |
| `POST /admin/events/<id>/close` | | `EventView` with winners. Requires `X-Admin-Pin`. |

Notes:

- **`POST /generations`:**
  - `count: 1` is free play: no set, no event needed.
  - `count: 3` requires `commanderName`. It creates a new draft set, abandoning any existing draft. No open event is needed to generate.
  - Every card is created `queued` and enqueued.
  - Returns 429 if the caller would exceed 3 pending cards.
- **`GET /queue_status`:** `cardsAhead` is the total number of queued cards and `etaSeconds` is the ETA for a newly submitted card. It is extended in section 5.
- **Admin PIN:** the `ADMIN_PIN` constant sits at the top of `app.py` next to the other config constants.

## 5. Generation pipeline

This lives in the new module `proxy-server/generation_queue.py`. It replaces `RequestQueue` for new routes; the old routes keep `RequestQueue`.

### Interface (the contract used by `api_routes.py`)

```python
enqueue(card_id: str) -> None
position(card_id: str) -> tuple[int | None, float | None]   # (1-based queue position, eta seconds)
status() -> dict                                            # payload for GET /queue_status
pending_count(user_id: str) -> int
recover_on_startup() -> None      # marks queued/generating/rendering cards failed ("Server restarted - please reroll")
```

### Workers

- **Text worker thread:** takes one card at a time and calls the existing `createCardContent(prompt, card_data)`.
- **Image worker thread:** owns the GPU, takes one card at a time and calls `image_generation.generate_art(prompt, card_data)`.
- A card enters both queues when enqueued. Its status moves through:
  - `queued`
  - `generating`, once either worker picks it up. Each worker sets `text_ready` or `art_ready` when its half finishes.
  - `rendering`, once both halves are done
  - `done`
- **Rendering** reuses today's post-processing in `process_card_generation`, factored into a shared function in `app.py`, then `card_renderer.generate_card_image`:
  - `~` replacement, bullet fixes, periods
  - creature P/T and Vehicle crew generation
- **Failures:** if either half fails, the card becomes `failed` with a user-readable `error`.
- **Set cards:** `card.name` is forced to the set's `commander_name` before rendering, and the LLM's name is discarded.
- **Output:** results are written to `data/art/` and `data/cards/`, and the `cards` row is updated via `storage.update_card(...)`.

### Queue positions and ETA

- The position is the card's index in the image queue, since images are the bottleneck.
- `etaSeconds = avg_last_10_image_durations × position`, plus the remaining estimate for the card currently painting. Before any history exists, assume 10 seconds per image.

### Per-user cap

- At most 3 pending (not `done`/`failed`) cards per user, enforced in `POST /generations` and reroll.

## 6. Image quality

Image generation moves out of `app.py` into `proxy-server/image_generation.py`, which exposes `generate_art(prompt, card_data) -> str` (base64 PNG at art-box size). The lazy pipeline load and cold/warm tracking move with it.

### Config constants (top of `app.py`)

| Constant | Value |
|---|---|
| `USE_CUDA` | existing |
| `MODEL_SIZE` | existing; `placeholder` is kept for development |
| `IMAGE_MODEL_ID` | `"Lykon/dreamshaper-xl-lightning"` |
| `IMAGE_STEPS` | 6 |
| `IMAGE_GUIDANCE` | 2.0 |
| `IMAGE_GEN_SIZE` | `(1088, 896)` |

### Pipeline

- `StableDiffusionXLPipeline` in fp16 on CUDA, with the DPM++ SDE Karras scheduler as recommended for Lightning.
- VAE slicing enabled.
- Output downscaled with Lanczos to the art box. 1088×896 matches the 408×336 aspect ratio of 1.214.

### Prompt construction

The prompt combines:
- a style prefix: painterly Magic: The Gathering-style fantasy illustration, dramatic lighting, highly detailed
- the user's subject
- mood hints from colors (e.g. R → fiery, embers; U → arcane, ocean; B → shadow, decay; G → verdant, primal; W → radiant, ordered) and from the type (creature → character focus; land → landscape; artifact → object focus)
- a negative prompt: text, letters, watermark, signature, border, frame, card, UI, blurry, lowres, deformed, extra limbs

CLIP's 77-token limit is still respected by the existing `truncate_prompt_smartly`, and the subject is prioritized over the style.

### Comparison page

`proxy-server/tools/compare_art.py` renders about 8 fixed prompts with the old SDXL-Turbo setup and the new one, writing a side-by-side HTML page for the owner to judge.

### Requirements

Update `proxy-server/requirements.txt` for the CUDA 12.8 torch build: document the extra index URL in a comment and drop xformers.

## 7. Frontend

### Shell and routing

`AppComponent` becomes a shell containing:
- the header with nav (Create · Commander Set · Gallery · Vote)
- the username with a Log out link
- the queue badge
- `<router-outlet>`

An `authGuard` sends users without a stored user to `/login`.

| Route | Component | Notes |
|---|---|---|
| `/login` | `LoginPageComponent` | Username input and Enter |
| `/create` (default) | `CreatePageComponent` | Today's page moved here, including the rotating messages. Submits `count: 1` via polling. Results are saved automatically. |
| `/set` | `SetBuilderPageComponent` | See below |
| `/gallery` | `GalleryPageComponent` | Grid of `/me/cards` with downloads |
| `/vote` | `VotePageComponent` | See below |
| `/events/:id` | `EventHistoryPageComponent` | Read-only final results |
| `/admin` | `AdminPageComponent` | PIN (kept in sessionStorage), create event, close event, history |

### Set builder (`/set`)

- The existing `CardFormComponent` plus a Commander name field, then **Generate set of 3**.
- A `CardSlotComponent` for each of the 3 slots shows the status line:
  - "Queued #4 · ~35s"
  - While generating, two checklist lines, "Rules text" and "Artwork", each ticked when its flag is ready
  - "Rendering…"
  - the finished card (tap to enlarge)
  - or "Failed: <error>"
- Each slot has a **Reroll** button.
- The **Lock in** button:
  - It is enabled only when all 3 slots are done and an event is open.
  - Otherwise it's disabled with the reason, e.g. "No event open — ask the host".
- When locked, the page shows a **Locked** badge and an **Unlock** button. Unlock asks for confirmation: "This clears votes on your set".
- The page loads existing state from `GET /me/sets/current`, so a refresh loses nothing.

### Vote page (`/vote`)

- One row per locked set: the commander name, "by <username>", and the 3 cards (a horizontal scroller on narrow screens).
- Each card has a **Vote** button and a live count.
- My pick is outlined.
- The leader gets a gold glow and a "Leading" crown. Tied cards show "Tied".
- The page polls every 5s.
- When the event is closed:
  - a winners banner lists each set's winning version, or tied versions
  - vote buttons are disabled
- With no open event, the page shows "No event open" and links to past events.

### Queue badge

- It polls `GET /queue_status` every 5s, replacing today's `/health` polling in `HealthService`.
- It shows **"Server idle"** or **"Server busy · N cards ahead · ~M min"**.

### Services

| Service | Responsibility |
|---|---|
| `UserService` | Login and the localStorage key `mtgenesis.user` |
| `GenerationService` | Submit, reroll, and poll `GET /cards/<id>` every 2s until done or failed |
| `EventService` | Events, sets, votes, admin |
| `QueueService` | The badge polling |

An HTTP interceptor adds `X-User-Id` to every request.

### Models

- Recreate `src/app/models/card.model.ts` and `api.model.ts`. Their earlier contents were never committed, so they are reconstructed from how they're used in the existing code.
- Add the view types from section 4.
- Change `.gitignore` `models/` → `/models/`.

## 8. Testing

- **Backend (new `pytest` suite in `proxy-server/tests/`):**
  - Storage rules:
    - login case-insensitivity and validation
    - one set per user per event, and replacing a draft
    - lock preconditions, and unlock clearing votes
    - one vote per voter per set, and vote overwrites
    - votes rejected on closed events
    - leader, tie and no-vote cases
    - one open event
  - API tests use the Flask test client with a temporary data dir and a fake queue.
  - Queue tests use fake text and image functions:
    - status transitions, ordering, the per-user cap
    - ETA math, forced commander name, restart recovery
- **Frontend (Karma):**
  - services with `HttpTestingController`
  - the guard redirect
  - Lock-button enablement
  - leader and tie display
- **Image quality:** the comparison page from section 6, reviewed by the owner.
- **End-to-end:**
  - The backend runs with `MODEL_SIZE="placeholder"` and a stubbed text model.
  - Playwright drives two users through: login → set → reroll → lock → cross-votes → admin closes → winners shown.
  - Then one real-GPU run of a full 3-card set.

## 9. Delivery plan (agent split)

Work happens on a feature branch. Phase 1 agents each work in their own git worktree.

### Phase 0 (lead, done first)

- The environment on this machine: Python, torch cu128, Ollama with `mistral:latest`. Every system-wide install is approved by the owner.
- The model files and the `.gitignore` fix.
- The contract:
  - the section 4 TypeScript types in `api.model.ts`
  - stub `storage.py` and `generation_queue.py` modules with final signatures and docstrings
  - the `image_generation.generate_art` signature

### Phase 1 (4 Opus agents in parallel)

| Agent | Owns |
|---|---|
| **A: Storage and API** | `storage.py`, `api_routes.py`, their tests |
| **B: Generation queue** | `generation_queue.py`, shared post-processing factored out of `process_card_generation`, blueprint registration and startup recovery in `app.py`, and its tests |
| **C: Image quality** | `image_generation.py` (extracted from `app.py`), config constants, requirements, `tools/compare_art.py` |
| **D: Frontend** | everything under `src/`, built against the contract |

### Phase 2 (lead)

- Merge the branches and resolve the `app.py` overlaps between B and C.
- Run the full backend and frontend tests.
- Run the end-to-end check.
- Review the code, then hand off.
