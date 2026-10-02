# Card Director Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A director LLM call writes a hidden brief per card (three distinct ones per commander set) that steers both the rules-text prompt and the art prompt, so cards and set versions stop looking and playing alike.

**Architecture:** A new pure module `proxy-server/director.py` (Ollama client injected) writes and validates briefs. `GenerationQueue` gets a brief stage served first by its Ollama worker; finished briefs go to `cards.brief_json` and are merged into the card params as `brief`, which `rules_text.build_messages` and `image_generation.build_art_prompt` read. No brief means today's prompts, byte for byte.

**Tech Stack:** Python 3.12, Flask, SQLite, Ollama (`qwen3:8b`), diffusers SDXL; pytest via `proxy-server/.venv/Scripts/python.exe -m pytest tests -q` from `proxy-server/`.

**Spec:** `docs/superpowers/specs/2026-10-02-card-director-design.md`

## Global Constraints

- Brief shape: `{identity ≤ 20 words, mechanic ≤ 15 words, art: {subject, action, setting, framing, light} each ≤ 12 words}`.
- Mechanics differ when content-word Jaccard overlap < 0.5 (`MECHANIC_MAX_OVERLAP = 0.5`).
- Director call: `format` JSON schema, `think=False`, `num_ctx` 2560 (keep the context small so SDXL fits beside the 8B model on the 12 GB GPU), one retry, then `None`. Never raises to callers.
- `DIRECTOR_ENABLED = os.environ.get("MTG_DIRECTOR", "1") != "0"`; `DIRECTOR_MODEL = os.environ.get("MTG_DIRECTOR_MODEL", TEXT_MODEL)`.
- Without a brief, `build_messages` and `build_art_prompt` output must be identical to today.
- The brief is never in `CardView`.
- Files are CRLF; keep their endings. Do not start Flask, Ollama, or touch `proxy-server/data` (the live site uses them). GPU e2e runs need the user's go-ahead (Task 6).

## Review Focus

1. A commander set's first card reaches the brief stage while its siblings are still waiting: exactly one 3-brief call, and each slot gets *its* brief (slot 1 ↔ brief 0), not the order the cards were taken (Task 4 test).
2. A reroll when the set's original director call had failed (siblings have no brief): it must make a `count=1` call with `avoid=[]`, not a 3-brief call (Task 4 test).
3. The director returns briefs whose art fields contain filtered words or no subtype: the art prompt must not carry the words and must name the subtype (Tasks 1, 3 tests).
4. Ollama is down: the brief stage must not fail or stall the card; text then fails as it does today and art still paints (Task 4 test).
5. Queue position while a card waits for its brief must be a number (cards ahead), not `None` — the UI reads `None` as "art ready" (Task 4 test).

---

### Task 1: `director.py` and config

**Files:**
- Create: `proxy-server/director.py`
- Modify: `proxy-server/config.py` (after `TEXT_THINK`)
- Test: `proxy-server/tests/test_director.py`

**Interfaces:**
- Produces:
  - `MECHANIC_MAX_OVERLAP = 0.5`, `ART_FIELDS = ("subject", "action", "setting", "framing", "light")`
  - `content_words(text: str) -> set[str]` — lowercase words of 3+ letters minus a small stopword set (`the, and, for, with, your, you, each, that, this, from, into, are, its, when, whenever, card, cards, creature, creatures`)
  - `jaccard(a: set[str], b: set[str]) -> float` (0.0 when both empty)
  - `write_briefs(card: dict, count: int, avoid: list[dict] | None, client, model: str) -> list[dict] | None`
  - `config.DIRECTOR_ENABLED: bool`, `config.DIRECTOR_MODEL: str`

- [ ] **Step 1: Write the failing tests** with a stub client whose `chat(**kwargs)` returns queued replies `{"message": {"content": json.dumps(...)}}` (or raises) and records kwargs.

```python
CARD = {"name": "Zur'ka, Élan of Ash", "type": "Creature", "supertype": "Legendary",
        "subtype": "Human Cleric", "colors": ["B"], "manaCost": "{2}{B}", "cmc": 3, "rarity": "mythic"}

def test_one_brief_parsed_and_trimmed():
    # identity of 30 words comes back trimmed to 20; art fields to 12; result is a list of 1

def test_three_briefs_with_distinct_mechanics(): ...   # 3 briefs returned in order
def test_overlapping_mechanics_retry_once_then_none():
    # first reply: two briefs share a mechanic -> second call made; second reply also bad -> None; 2 calls total
def test_avoid_is_sent_and_enforced():
    # avoid=[{"mechanic": "sacrifice tokens to drain"}]; reply mechanic "sacrifice tokens to drain life" -> retry
    # the user message of the call contains "sacrifice tokens to drain"
def test_subject_gets_the_subtype_when_missing():
    # art.subject "a robed priest at an altar" -> "a human cleric, a robed priest at an altar"
def test_art_word_filter():
    # "a shirtless nude priest" -> no "shirtless"/"nude" in any art field
@pytest.mark.parametrize("reply", ["not json", '{"briefs": []}', '{"briefs": [{"identity": "x"}]}'])
def test_garbage_returns_none(reply): ...
def test_client_error_returns_none():  # chat raises -> None, no exception
def test_call_options():  # format schema present, think False, options num_ctx 2560, model passed through
def test_content_words_and_jaccard():
    assert jaccard(content_words("Sacrifice tokens to drain"), content_words("drain by sacrificing tokens")) > 0
    assert jaccard(set(), set()) == 0.0
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_director.py -q` — Expected: ImportError.

- [ ] **Step 3: Implement** `director.py`:
  - Response schema: `{"type":"object","properties":{"briefs":{"type":"array","minItems":count,"maxItems":count,"items":{brief object, all fields required}}},"required":["briefs"]}`.
  - Messages: system = the prompt below; user = the card facts (reuse `rules_text.type_line(card)` and `power.describe_budget(card)`), then `Write {count} brief(s).`, then when `count > 1` "Each brief must use a different mechanic.", then for `avoid` "Use a mechanic different from: " + each avoided mechanic.
  - Call: `client.chat(model=model, messages=..., format=schema, think=False, options={"temperature": 0.9, "top_p": 0.95, "num_predict": 200 * count + 100, "num_ctx": 2560}, keep_alive="30m")`.
  - Validate: all fields strings and non-empty → trim → filter words (`nude, naked, topless, shirtless, bare-chested, bare chest, cleavage, nsfw`, case-insensitive, whole words, collapse spaces) → subtype prefix (`a <subtype lowercased>` + ", " when no subtype word appears in `art.subject`) → pairwise and avoid overlap `< MECHANIC_MAX_OVERLAP`. Any failure: one retry; then `None`.
  - System prompt (exact copy):

```text
You are the creative director for a custom Magic: The Gathering card. From the card's name, type line, colors, cost and rarity, invent who or what this card is and what its rules text should revolve around, then describe its painting.
Return JSON only: {"briefs": [{"identity": "...", "mechanic": "...", "art": {"subject": "...", "action": "...", "setting": "...", "framing": "...", "light": "..."}}]}.
- identity: who or what the card is, in at most 20 words. Draw on the name and subtype.
- mechanic: what its abilities revolve around, in at most 15 words, in Magic terms (for example "sacrifice tokens to drain each opponent"). Fit the power budget: cheap or common cards get small, simple mechanics.
- art: a painting brief. subject names the creature type and what it looks like; action is what it is doing; setting is where; framing is the camera (for example "low angle, close"); light is the light source and mood. Each at most 12 words. People are always fully clothed.
Make each brief specific to this card. Avoid generic fantasy filler.
```

- Add the two config constants with a comment that they are the director's switch and model.

- [ ] **Step 4: Run** the test file — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "Director: write and validate card briefs"`

### Task 2: Storage — `brief_json`

**Files:**
- Modify: `proxy-server/storage.py` (cards schema, `_migrate`, `_CARD_COLS`, `_decode_card`, new `set_card_brief`)
- Test: `proxy-server/tests/test_storage_sets.py` (row-shape key set gains `"brief"`), `proxy-server/tests/test_director_storage.py`

**Interfaces:**
- Produces: card rows carry `brief: dict | None`; `Storage.set_card_brief(card_id: str, brief: dict | None) -> None`.

- [ ] **Step 1: Write failing tests:** a new card's `brief` is `None`; `set_card_brief(id, {...})` round-trips; an old database without the column gains it on open (build the old `cards` table like `tests/test_sharing.py::test_old_database_gains_the_shared_column`, include `shared_at`); `CardView` keys in `tests/test_api.py` stay unchanged (no `brief`).
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_director_storage.py tests/test_storage_sets.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.** Column `brief_json TEXT`, migrated like `shared_at`; decoded to `brief` alongside `card_params`/`card`.
- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest tests -q` — Expected: all PASS.
- [ ] **Step 5: Commit** — `git commit -m "Storage: cards.brief_json"`

### Task 3: Prompts read the brief

**Files:**
- Modify: `proxy-server/rules_text.py` (`build_messages`), `proxy-server/image_generation.py` (`build_art_prompt`)
- Test: `proxy-server/tests/test_rules_text.py`, `proxy-server/tests/test_image_prompt.py`

**Interfaces:**
- Consumes: Task 1's brief dict shape, read from `card["brief"]` / `card_data["brief"]`.
- Produces: no new public names.

- [ ] **Step 1: Write failing tests:**
  - `build_messages` with a brief: user message contains `Card idea: <identity>. Build the abilities around this mechanic, scaled to the power budget: <mechanic>.`; contains no `Design hook` and no `Concept:` line.
  - `build_messages` without a brief, same `random.Random(7)`: equal to the output computed before this change (snapshot a fixed card's messages in the test from `HEAD` code first, then assert equality).
  - `build_art_prompt("Zur'ka, a legendary cleric", card_with_brief)`: positive starts with the brief's `subject, action, setting, framing`; does not contain `Zur'ka`; still contains `ART_STYLE`'s first words and the human context; negative unchanged.
  - Trim order with a brief: with a 6-token budget stub `count_tokens` that forces trimming, `light` is cut before `setting`, and `subject` survives.
  - `build_art_prompt` without a brief: identical to `HEAD` output for three cards (creature, instant, colorless artifact).
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_rules_text.py tests/test_image_prompt.py -q` — Expected: FAIL on the brief cases only.
- [ ] **Step 3: Implement.** In `build_messages`, when `card.get("brief")`: skip the `Concept:` fact and replace the hook line. Still call `rng.choice(hooks)` in both paths, so seeded tests and the `--repeat` e2e runs draw the same random numbers either way. In `build_art_prompt`, when `card_data.get("brief")`: subject = `", ".join(art[f] for f in ART_FIELDS if art.get(f))` and trimming of the subject drops whole trailing fields first (`light`, then `framing`, then `setting`, then `action`) before word-trimming `subject` down to `SUBJECT_MIN_TOKENS`.
- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest tests -q` — Expected: all PASS.
- [ ] **Step 5: Commit** — `git commit -m "Rules and art prompts follow the card brief"`

### Task 4: Queue brief stage and app wiring

**Files:**
- Modify: `proxy-server/generation_queue.py`, `proxy-server/app.py` (queue construction, `brief_fn` wiring)
- Test: `proxy-server/tests/test_generation_queue.py`
- Modify: `docs/superpowers/specs/2026-10-02-card-director-design.md` §3 "Restart" line (see ruling below)

**Interfaces:**
- Consumes: `Storage.set_card_brief`, card `brief`, `Storage.set_cards(set_id)`.
- Produces:
  - `BriefFn = Callable[[dict, int, list[dict]], "list[dict] | None"]` — `(card_params, count, avoid) -> briefs | None`
  - `GenerationQueue(..., brief_fn: BriefFn | None = None)`; `None` = director off (enqueue exactly as today)
  - `GenerationQueue.process_next_brief() -> bool`
  - app: `brief_fn = (lambda params, n, avoid: director.write_briefs(params, n, avoid, ollama_client, DIRECTOR_MODEL)) if DIRECTOR_ENABLED else None`

Ruling to carry: spec §3 says a card waiting for its brief is "re-enqueued" on restart, but `recover_on_startup` marks unfinished cards failed ("Server restarted - please reroll"). Keep today's behavior and correct the spec line to say so.

- [ ] **Step 1: Write failing tests** (`start_workers=False`, step the workers by hand, stub `text_fn`/`art_fn`/`render_fn` that record the `card_params` they get, stub `brief_fn` that records calls):
  - `test_brief_runs_before_text_and_art`: one free-play card; before `process_next_brief`, `process_next_text()` and `process_next_image()` both return False; after it, both run and both see `params["brief"]`.
  - `test_text_worker_serves_briefs_first`: card A past its brief, card B enqueued; the Ollama worker step (`process_next_ollama`, the new combined step the text thread runs) does B's brief before A's text.
  - `test_set_makes_one_call_and_maps_slots`: three set cards (slots 1–3) enqueued; take slot 2 first; exactly one call with `count=3, avoid=[]`; slot *i* card stored brief *i-1* (Review Focus 1).
  - `test_reroll_avoids_siblings`: set whose three cards have briefs; reroll slot 2 (storage `reroll_card`) and enqueue; call is `count=1, avoid=[brief of slot 1, brief of slot 3]`.
  - `test_reroll_after_failed_set_brief`: siblings have no brief; reroll → `count=1, avoid=[]` (Review Focus 2).
  - `test_none_brief_continues`: `brief_fn` returns None → text and art run without `brief`; card finishes `done`.
  - `test_brief_fn_exception_continues`: `brief_fn` raises → same as None (Review Focus 4).
  - `test_director_off_is_todays_queue`: `brief_fn=None` → `process_next_text` works immediately after `enqueue`.
  - `test_position_counts_brief_stage`: two cards waiting for briefs, none in image list → second card's `position()` is `(2, number)`; `status()["cardsAhead"] == 2` (Review Focus 5).
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_generation_queue.py -q` — Expected: new tests FAIL.
- [ ] **Step 3: Implement.**
  - `_brief_waiting: deque[str]`. `enqueue` with `brief_fn`: add to `_partials` and `_brief_waiting` only.
  - `process_next_brief`: pop a card; if it already has `brief` in storage, release it. Else find current set siblings (`storage.set_cards`) still in `_brief_waiting`: the group = this card + those siblings, ordered by slot. If the group has more than one card → `brief_fn(params, len(group), [])`; else → `brief_fn(params, 1, [s["brief"] for s in other current set cards if s["brief"]])`. Free play → `(params, 1, [])`. Exceptions → `None`. Store briefs by slot order with `set_card_brief`, remove the group from `_brief_waiting`, and release each: append to `_text_waiting` and `_image_waiting`.
  - The text thread runs `process_next_ollama`: `return self.process_next_brief() or self.process_next_text()`.
  - `text_fn`/`art_fn` get `params = {**card_params, "brief": card["brief"]}` when the card has one, else `card_params` unchanged.
  - `position()`: if the card is in `_brief_waiting`, return `(len(_image_waiting) + index_in_brief + 1, eta)` using the same eta formula; `status()["cardsAhead"]` adds `len(_brief_waiting)`. `_drop` also removes from `_brief_waiting`.
  - Update the module docstring's lifecycle paragraph.
  - app.py: build `brief_fn` per Interfaces and pass it; print `DIRECTOR_ENABLED`/`DIRECTOR_MODEL` in the startup banner.
  - Spec: fix the Restart line per the ruling above.
- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest tests -q` — Expected: all PASS (slow tests import app.py and must still pass).
- [ ] **Step 5: Commit** — `git commit -m "Queue: brief stage before text and art"`

### Task 5: e2e set mode and variety metrics

**Files:**
- Modify: `proxy-server/tools/e2e_rules_text.py`, `proxy-server/tools/e2e_art.py`
- Test: `proxy-server/tests/test_director.py` (metric helper only)

**Interfaces:**
- Consumes: `director.write_briefs`, `director.content_words`, `director.jaccard`, `storage.commander_slot_params`.
- Produces: `director.set_overlap(texts: list[str]) -> float` — mean pairwise `jaccard(content_words(a), content_words(b))`.
  - CLI: `e2e_rules_text.py --sets N [--no-director]`, `e2e_art.py --sets N [--no-director]`.
  - `SET_SPECS`: 4 fixed commander requests (mono-B Human Cleric mythic, mono-G Elf Druid rare, U/R Wizard mythic, colorless Golem rare — name, colors, pips, subtype, rarity).

- [ ] **Step 1: Write the failing test:** `set_overlap(["Flying. Draw a card.", "Flying. Draw a card.", "Trample"])` is between 0 and 1 and greater than `set_overlap(["Flying", "Trample", "Deathtouch"])`; `set_overlap(["x"])` is 0.0.
- [ ] **Step 2: Run** — Expected: FAIL (no `set_overlap`).
- [ ] **Step 3: Implement** `set_overlap`, then the CLI modes:
  - rules: for each of the first N `SET_SPECS`: build slot params with `commander_slot_params`; unless `--no-director`, `write_briefs(params_slot1, 3, [], client, DIRECTOR_MODEL)` and attach brief *i* to slot *i*; run `createCardContent` + `finalize_card` per slot as the tool already does. Report per set: briefs (JSON), the three final texts, `set_overlap` of the three texts, lint findings, `over_budget`; totals: mean `set_overlap`, lint errors, over-budget count, mean seconds per card (director call included).
  - art: same sets and briefs (seed `SEED_BASE + set_index * 100 + slot`), `generate_art` per slot; CLIP similarity with `transformers` `CLIPModel`/`CLIPProcessor` `openai/clip-vit-base-patch32` (downloaded on first use, loaded only in `--sets` mode); report per set the mean pairwise cosine of image embeddings and existing exposure stats; a contact sheet row per set.
- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest tests -q` and `.venv/Scripts/python.exe tools/e2e_rules_text.py --help` / `tools/e2e_art.py --help` — Expected: tests PASS; both help texts list `--sets` and `--no-director`.
- [ ] **Step 5: Commit** — `git commit -m "e2e: commander set mode with variety metrics"`

### Task 6: Docs, and the measured comparison (needs the user)

**Files:**
- Modify: `CLAUDE.md` (Backend section: a Director bullet group; Commands: the new e2e flags)

- [ ] **Step 1: Edit CLAUDE.md** — director module, brief stage, `MTG_DIRECTOR`/`MTG_DIRECTOR_MODEL`, hidden `brief_json`, the `--sets/--no-director` e2e flags.
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests -q` — Expected: all PASS. Commit — `git commit -m "Docs: card director"`.
- [ ] **Step 3: Ask the user for a window when the site isn't live** (memory: no GPU experiments while live). Then run, from `proxy-server/`:
  `tools/e2e_rules_text.py --label director-off --sets 4 --no-director --repeat 2`, `--label director-on --sets 4 --repeat 2`, `tools/e2e_art.py --label art-director-off --sets 4 --no-director`, `--label art-director-on --sets 4`.
  Expected (spec §7): director-on mean text overlap and mean CLIP similarity each ≥ 25% lower than director-off; lint errors and over-budget no worse; mean seconds per card ≤ director-off + 4. Report the numbers to the user; if a criterion fails, stop and discuss before merging.
