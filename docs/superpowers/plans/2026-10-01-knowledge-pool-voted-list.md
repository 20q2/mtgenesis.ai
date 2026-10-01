# Knowledge Pool Voted List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the slot/ban Knowledge Pool with one open, medal-voted list where the top ⌊submitters / 2⌋ colorless or mono-colored cards make the pool, submittable from the create screen and gallery.

**Architecture:** Rework in place. `storage.py` loses slots and bans, gains a startup migration and a pure `pool_ranking`; `api_routes.py` serves a flat `PoolView` and adds `poolEntryId` to `CardView`; the Angular pool page, admin form and a new shared submit button consume it.

**Tech Stack:** Flask + SQLite (`proxy-server/`, pytest via `.venv/Scripts/python.exe -m pytest`), Angular 16 (Karma/Jasmine, `npx ng test --watch=false --browsers=ChromeHeadless`).

**Spec:** `docs/superpowers/specs/2026-09-29-knowledge-pool-design.md`

## Global Constraints

- Medal points: gold 3, silver 2, bronze 1 (`MEDAL_POINTS`, unchanged). One of each per player **per pool**.
- Cutoff N = ⌊submitters / 2⌋, submitters = distinct users with ≥ 1 entry. Rank by (points, gold, silver) descending; everything level with the Nth card on all three is also in; 0 points is never in.
- Eligible cards: owner's own, status `done`, `len(card_colors(card)) <= 1`. Error text: "Only colorless or mono-colored cards can enter the pool".
- Entry cap per pool: 1–10, default 3.
- Everything is visible while open (usernames always filled in).
- Files are CRLF; keep their line endings. Run Python with `proxy-server/.venv/Scripts/python.exe`.
- Asset URLs stay relative; new routes need no OPTIONS code (app.py handles it).

## Review Focus

1. A card with no `colors` list and a hybrid cost like `{W/U}` counts as two colors and must be rejected (Task 1 test).
2. Withdrawing an entry can drop a player out of "submitters" and move the cutoff; the ranking must be recomputed, and the withdrawn card's medals must be freed so voters can reuse them (Task 1 test).
3. `colors: ['C']` (or empty colors with `{3}` cost) is colorless and allowed (Task 1 test).
4. The pool closes while a player is on the create page: the Submit button's request gets a 409 and must show the server's message, not stay "In the pool" (Task 5 test).
5. A database with the old slot tables *and data* must not be silently wiped: startup raises (Task 1 test).

---

### Task 1: Storage — flat pool, ranking, migration

**Files:**
- Modify: `proxy-server/storage.py` (schema `_SCHEMA` pool tables; pool constants; `card_fits_slot`, `slot_rule_text`, `clean_pool_slots` removed; `Storage._migrate`; pool methods; `pool_standings` → `pool_ranking`)
- Test: `proxy-server/tests/test_pool.py` (rewrite storage tests; delete slot/ban tests)

**Interfaces:**
- Produces:
  - `POOL_ENTRY_CAP_MAX = 10`, `POOL_DEFAULT_ENTRIES = 3`
  - `pool_card_eligible(card: dict) -> bool` — `len(card_colors(card)) <= 1`
  - `pool_ranking(entries: list[dict], medals: dict[str, dict[str, int]]) -> dict[str, dict]` — `entries` rows have `id`, `user_id`, `created_at`; returns per entry id `{gold, silver, bronze, points, rank, in, tiedAtCutoff}`; also `pool_cutoff(entries) -> int`
  - `Storage.create_pool(name: str, max_entries_per_user: int) -> dict`
  - `Storage.submit_pool_entry(user_id: str, card_id: str) -> dict` (the entry row)
  - `Storage.withdraw_pool_entry(entry_id, user_id) -> str` (pool id), `award_pool_medal(voter_id, entry_id, medal) -> str`, `clear_pool_medal(voter_id, entry_id) -> str` (same signatures as today)
  - `Storage.pool_entries`, `pool_medal_counts`, `my_pool_medals`, `current_pool`, `get_pool`, `list_pools`, `close_pool` (unchanged signatures)
  - `Storage.open_pool_entry_ids() -> dict[str, str]` — card id → entry id for the open pool (`{}` when none)
  - Removed: `pool_slots`, `pool_ban_counts`, `my_pool_bans`, `ban_pool_entry`, `unban_pool_entry`, `_open_pool_slot`, `pool_standings`, `POOL_BANS_PER_PLAYER`, `POOL_BAN_THRESHOLD`, `POOL_MAX_SLOTS`, `POOL_SLOT_LABEL_MAX`, `COLOR_RULES`, `TYPE_RULES`

- [ ] **Step 1: Write the failing tests** in `tests/test_pool.py` (keep the file's existing `pool` fixture idea: `tmp_storage.create_pool("Night", 3)`; helper `done_card(storage, user_id, colors, cost)` creating a `done` card whose `card` dict has those `colors`/`manaCost`).

```python
@pytest.mark.parametrize("colors,cost,ok", [
    (["W"], "{1}{W}", True), ([], "{3}", True), (["C"], "{2}", True), ([], "{G}{G}", True),
    (["W", "U"], "{W}{U}", False), ([], "{W/U}", False), ([], "{1}{B}{R}", False)])
def test_pool_card_eligible(colors, cost, ok):
    assert pool_card_eligible({"colors": colors, "manaCost": cost}) is ok

def test_submit_rules(tmp_storage, pool):
    # own done mono card ok; 403 other's card; 400 unfinished; 400 multicolor with the spec's text;
    # 409 same card twice; 409 at cap 3; 404 when no pool is open; 409 after close

def test_withdraw_frees_medals_and_moves_cutoff(tmp_storage, pool):
    # 4 submitters -> cutoff 2; one withdraws their only entry -> cutoff 1;
    # the gold that was on it is gone and the voter can give gold to another card

def test_medals_one_each_per_pool_and_never_own(tmp_storage, pool):
    # gold on A then gold on B moves it (A has 0 gold); silver on B replaces gold on B;
    # 403 on own card; 400 unknown medal

@pytest.mark.parametrize(...)  # cases below
def test_pool_ranking(...):
```

`test_pool_ranking` cases (entry ids → (gold, silver, bronze) counts, submitters from distinct `user_id`s):
- 8 submitters, points 9, 7, 6(1g), 6(0g, 2s), 6(0g, 1s), 2 → `in` for the first four, not the 6(1s) or the 2; `tiedAtCutoff` false for all.
- Same but the last two sixes both 0g 2s → both `in` and both `tiedAtCutoff`; 5 in.
- 3 submitters, one entry with points, one with 0 → cutoff 1, the 0-point one never `in`.
- 1 submitter → cutoff 0, nothing `in`.
- Equal keys share `rank` (e.g. ranks 1, 2, 3, 3).

```python
def test_migration_drops_empty_old_tables(tmp_path):
    # build the old schema (pool_slots/pool_bans + slot_id columns) empty, open Storage, then
    # PRAGMA table_info(pool_entries) has no slot_id and pool_slots is gone; reopening is a no-op

def test_migration_refuses_old_tables_with_rows(tmp_path):
    # same old schema with one pools row -> Storage(db) raises RuntimeError mentioning pool_slots

def test_open_pool_entry_ids(tmp_storage, pool):
    # {} before any entry; {card_id: entry_id} after; {} after close_pool
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd proxy-server && .venv/Scripts/python.exe -m pytest tests/test_pool.py -q`
Expected: ImportError on `pool_card_eligible` / `pool_ranking`.

- [ ] **Step 3: Implement** in `storage.py`:
  - New pool tables per spec §3 (`pool_entries` without `slot_id`, unique `(pool_id, card_id)`, index on `(pool_id, user_id)`; `pool_medals` unique `(voter_id, pool_id, medal)` and `(voter_id, entry_id)`).
  - `_migrate`: if `pool_slots` exists, count rows in `pools`, `pool_slots`, `pool_entries`, `pool_medals`, `pool_bans`; any rows → `raise RuntimeError("Old Knowledge Pool tables (pool_slots, pool_bans) hold data; migrate them by hand")`; else `DROP` those five and re-run the pool part of the schema. Runs before anything indexes the new columns.
  - `pool_ranking`: sort entry ids by `(-points, -gold, -silver, created_at)`; `rank` = 1 + number of entries with a strictly better key; cutoff key = key of the entry at index `cutoff - 1` (if `cutoff > 0` and it exists); `in` = `points > 0 and key >= cutoff key`; `tiedAtCutoff` = `in and key == cutoff key and` more than one entry has that key.
  - `create_pool` validates the cap with `1 <= cap <= POOL_ENTRY_CAP_MAX` (message "Entries per player must be a whole number from 1 to 10").
  - `submit_pool_entry` order of checks: open pool (404 "No Knowledge Pool is open"), card exists (404), owner (403), done (400), eligible (400, spec text), duplicate (409), cap (409, existing wording).

- [ ] **Step 4: Run backend tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pool.py -q -k "not api"`
Expected: PASS (API tests are rewritten in Task 2).

- [ ] **Step 5: Commit** — `git commit -m "Pool storage: one voted list, colorless/mono only, no slots or bans"`

### Task 2: API — flat PoolView and poolEntryId

**Files:**
- Modify: `proxy-server/api_routes.py` (imports; `card_view`; `pool_view`; pool routes; admin create; remove ban routes; module docstring line about bans)
- Test: `proxy-server/tests/test_pool.py` (API tests), `proxy-server/tests/test_api.py` (`CARD_VIEW_KEYS` gains `"poolEntryId"`)

**Interfaces:**
- Consumes: Task 1's `pool_ranking`, `pool_cutoff`, `open_pool_entry_ids`, `submit_pool_entry(user_id, card_id)`, `create_pool(name, cap)`.
- Produces (JSON, consumed by Task 3): `PoolView` and `PoolEntryView` exactly as spec §4; `CardView.poolEntryId: str | None`.
  - `card_view(row, pool_entry_ids: dict[str, str] | None = None)` — fetches `storage.open_pool_entry_ids()` when not given; list routes (`/me/cards`, `/cards/shared`, set views, pool views) fetch it once and pass it.

- [ ] **Step 1: Write the failing tests** (replace `test_api_full_flow` and `test_api_withdraw_and_auth`; keep `test_api_admin_create_requires_pin`, `test_api_pool_current_etag`, `test_power_check_flags_pot_of_green`):

```python
def test_api_full_flow(client, tmp_storage):
    # admin creates {"name": "Night", "maxEntriesPerUser": 3}; "slots" in the body is ignored
    # 4 users each submit one mono card via {"cardId"}; a multicolor card -> 400 with spec text
    # PoolView: submitters 4, cutoff 2, entries sorted by rank, every entry has username,
    #   keys == {"id","cardId","card","username","mine","power","gold","silver","bronze",
    #            "points","rank","in","tiedAtCutoff","myMedal","createdAt"}
    # medals from three voters -> top two have in True; myMedals reflects the viewer
    # GET /me/cards shows poolEntryId on the entered card and None on others
    # /pools/bans -> 404 (route gone)
    # close -> medal/submit/withdraw 409

def test_api_admin_cap_default_and_range(client):
    # missing maxEntriesPerUser -> 3; 0 and 11 -> 400
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pool.py tests/test_api.py -q`
Expected: FAIL (old slot shape, missing `poolEntryId`).

- [ ] **Step 3: Implement** `pool_view(row, viewer_id)` per spec §4 (one `get_user` cache as today; entries sorted by `rank` then `created_at`); `POST /pools/entries` reads only `cardId` (400 "cardId is required"); `admin_create_pool` passes `data.get("maxEntriesPerUser", POOL_DEFAULT_ENTRIES)`; delete the two ban routes.

- [ ] **Step 4: Run all backend tests**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: all PASS.

- [ ] **Step 5: Commit** — `git commit -m "Pool API: flat PoolView, cardId-only entries, poolEntryId on cards"`

### Task 3: Frontend models and PoolService

**Files:**
- Modify: `src/app/models/api.model.ts` (`PoolView`, `PoolEntryView` per spec; delete `PoolSlotView`, `PoolSlotSpec`, `PoolColorRule`, `PoolTypeRule`; `CardView.poolEntryId: string | null`)
- Modify: `src/app/services/pool.service.ts`, `src/app/testing/fixtures.ts` (`poolEntryId: null`; pool fixtures to the flat shape)
- Test: `src/app/services/pool.service.spec.ts`

**Interfaces:**
- Produces:
  - `PoolService.submit(cardId: string): Observable<PoolView>`
  - `PoolService.createPool(name: string, maxEntriesPerUser: number, pin: string): Observable<PoolView>`
  - `current()`, `get(id)`, `list()`, `withdraw(entryId)`, `medal(entryId, medal)`, `clearMedal(entryId)`, `closePool(id, pin)` unchanged
  - `export function poolColorOk(card: Partial<CardParams> | null | undefined): boolean` — `cardColors(card).length <= 1`
  - `export const POOL_DEFAULT_ENTRIES = 3`
  - Removed: `ban`, `unban`, `defaultPoolSlots`, `cardFitsSlot`, `POOL_COLOR_RULES`, `POOL_TYPE_RULES`

- [ ] **Step 1: Write failing specs:** `submit('c1')` POSTs `/pools/entries` with `{cardId: 'c1'}`; `createPool('Night', 3, '9999')` POSTs `{name, maxEntriesPerUser: 3}` with the PIN header; `poolColorOk` true for `{colors:['R']}`, `{colors:[], manaCost:'{4}'}`, `{colors:['C']}`; false for `{colors:['W','U']}`, `{colors:[], manaCost:'{W/U}'}`.
- [ ] **Step 2: Run** `npx ng test --watch=false --browsers=ChromeHeadless --include src/app/services/pool.service.spec.ts` — Expected: compile errors / FAIL.
- [ ] **Step 3: Implement.** The pool page and admin will not compile until Tasks 4–5; do Tasks 3–5 before running the whole suite.
- [ ] **Step 4: Run the same command** — Expected: PASS.
- [ ] **Step 5: Commit together with Tasks 4–5** (the app only compiles as a whole).

### Task 4: Pool page

**Files:**
- Modify: `src/app/pages/pool-page/pool-page.component.{ts,html,scss}`
- Test: `src/app/pages/pool-page/pool-page.component.spec.ts` (rewrite)

**Interfaces:**
- Consumes: Task 3's `PoolView`, `PoolService`.
- Produces (for the template and spec): `pool`, `isOpen`, `isPast`, `inEntries: PoolEntryView[]` (entries with `in`), `lineIndex: number` (index after which the pool line is drawn = last `in` entry's index, −1 if none), `award(entry, medal)`, `clearMedal(entry)`, `withdraw(entry)`, `canVote(entry): boolean` (open and not mine), `medalsGiven: Medal[]`. Remove the slot overview, picker, bans, `results`, `bannedOut`.

- [ ] **Step 1: Write failing specs:**
  - header shows "4 players → top 2 make the pool" and "your entries 1 / 3", and the rules line text from spec §5
  - tiles render in the server's order; exactly one `.pool-line` element sits after the last `in` tile; `in` tiles carry "In", `tiedAtCutoff` ones "Tied · in"
  - medal buttons call `PoolService.medal(entry.id, 'gold')`; pressed state from `myMedal`; own card shows Withdraw and no medal buttons
  - closed pool: no medal/withdraw buttons; a "The pool" section lists only `in` cards
  - empty pool: a link to `/create`
- [ ] **Step 2: Run** `--include src/app/pages/pool-page/pool-page.component.spec.ts` — Expected: FAIL.
- [ ] **Step 3: Implement.** Reuse the existing tile, medal-button and power-badge styles in the scss; delete slot/ban/picker styles. Keep the light panel look (light panels, dark chrome, slim controls).
- [ ] **Step 4: Run** — Expected: PASS.

### Task 5: Submit-to-pool button, admin form

**Files:**
- Create: `src/app/components/pool-submit/pool-submit.component.{ts,html,scss,spec.ts}` (declare in `app.module.ts`)
- Modify: `src/app/pages/create-page/create-page.component.{ts,html}`, `src/app/pages/gallery-page/gallery-page.component.{ts,html}`, their specs
- Modify: `src/app/components/pool-admin/pool-admin.component.{ts,html,scss,spec.ts}` (name + cap only, cap default `POOL_DEFAULT_ENTRIES`, min 1 max 10)

**Interfaces:**
- Consumes: `PoolService.submit`, `poolColorOk`, `CardView.poolEntryId`.
- Produces: `<app-pool-submit [card]="view" [pool]="pool" (submitted)="onPoolSubmitted($event)">` where `card: CardView`, `pool: PoolView | null`, `submitted: EventEmitter<PoolView>`. Host pages load `pool` once with `PoolService.current()` (errors → `null`) and, on `submitted`, store the returned pool and set that card's `poolEntryId` from the returned entries.
  - `disabledReason(): string | null` — in order: `'No pool is open'`, `"Multicolor cards can't enter the pool"`, `` `You've used all ${cap} entries` `` (when `myEntryCount >= maxEntriesPerUser`); null otherwise. Hidden entirely unless `card.status === 'done'`.
  - Label: `In the pool ✓` when `card.poolEntryId`, else `Submit to pool (${myEntryCount}/${cap})`.

- [ ] **Step 1: Write failing specs** for `PoolSubmitComponent`: each disabled reason; label with count; click calls `submit(card.id)` and emits; **a 409 response shows the server's error text and the button is not marked "In the pool"** (Review Focus 4). Create-page spec: button present once the job is done. Gallery spec: present on done My cards tiles only, absent on Community tiles. Pool-admin spec: create sends name and cap 3 by default, no slot controls rendered.
- [ ] **Step 2: Run** the five specs with `--include` — Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the whole frontend suite and a production build**

Run: `npx ng test --watch=false --browsers=ChromeHeadless` then `npx ng build --configuration production --output-path "$TEMP/mtg-build-check"`
Expected: all SUCCESS; build has no errors.

- [ ] **Step 5: Commit Tasks 3–5** — `git commit -m "Pool UI: ranked list with the pool line, submit from create and gallery, simple admin form"`

### Task 6: Docs and final verification

**Files:**
- Modify: `CLAUDE.md` (Knowledge Pool section: rewrite the bullets to the new rules; drop the `card_fits_slot`/`cardFitsSlot` sync note and replace it with `card_colors` ↔ `cardColors`/`poolColorOk`; mention submitting from create/gallery)

- [ ] **Step 1: Edit CLAUDE.md.**
- [ ] **Step 2: Run everything**

Run: `cd proxy-server && .venv/Scripts/python.exe -m pytest tests -q`, then the full frontend suite and build from Task 5.
Expected: all PASS.

- [ ] **Step 3: grep for leftovers:** `grep -rn "slotId\|ban\b\|bans\|cardFitsSlot\|pool_slots" src proxy-server --include=*.ts --include=*.html --include=*.py` — Expected: only the migration code and its test.
- [ ] **Step 4: Commit** — `git commit -m "Docs: Knowledge Pool voted list"`
