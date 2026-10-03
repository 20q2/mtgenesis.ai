# Commander Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commander sets follow the AI Night rule sheet. Each player builds three independent commanders (3, 4 and 5 CMC), each with 3 versions. P/T is point-buy, a player's three rarities are Uncommon, Rare and Mythic, voting is per commander grouped by CMC, and the owner's own vote counts double.

**Architecture:** A `sets` row becomes one commander. It gains `cmc` and `rarity` columns, and the rules are checked across a user's sets in `storage.py`. A new pure module, `commander_rules.py`, turns a request into the params all 3 versions share. The frontend gets a `commander-rules.ts` mirror for instant feedback, plus a per-CMC `CommanderPanelComponent` (today's set builder logic) hosted three times by the page.

**Tech Stack:** Flask + SQLite (`storage.py`), pytest; Angular 16, Karma/Jasmine.

**Spec:** `docs/superpowers/specs/2026-10-03-commander-rules-design.md`

## Global Constraints

- Only the commander set flow changes. `POST /generations` with `count: 1` and `/create` behave exactly as today.
- `COMMANDER_CMCS = (3, 4, 5)`; `COMMANDER_RARITIES = ("uncommon", "rare", "mythic")`; `VEHICLE_BONUS = 2`; points = `cmc + 1` (+ 2 for a Vehicle).
- Colored pips are worth at most 3 mana (`{2/W}` counts 2).
- P/T must be whole numbers written as digits (surrounding whitespace is trimmed), with power ≥ 0, toughness ≥ 1 and power + toughness ≤ points. X, `*` and `1+*` are rejected.
- A Vehicle commander is `supertype "Legendary"`, `type "Artifact"`, and its subtype contains `Vehicle`.
- Legacy sets (`cmc IS NULL`) are shown but never count toward the rules and can't be locked.
- Server error copy (exact):
  - "Commanders are Uncommon, Rare or Mythic"
  - "P/T must be whole numbers — X and * aren't allowed"
  - "A {cmc}-mana commander has {points} points; {p}/{t} uses {p+t}"
  - "Your {cmc} CMC commander is locked in — unlock it to start over"
  - "You already have a {Rarity} commander ({cmc} CMC)"
  - "This set was made under the old rules — start a new commander"
- Reroll confirm copy: "Rerolls are only for a version that doesn't function under any circumstance. Reroll it?"
- Vote header copy: "Vote for one version of each commander. A vote on your own commander counts as two — owners, vote first."
- Asset URLs stay relative. New routes (none planned) would need `add_ngrok_headers`/`OPTIONS`.
- Backend tests: `python -m pytest tests -m "not slow"` from `proxy-server/`. Frontend: `npx ng test --watch=false --include <spec>` from the repo root.

## Review Focus

1. **Regenerating the same CMC with the same rarity** (a player redoes their 4 CMC Rare) must succeed. The draft being replaced must not conflict with itself. → Task 2, `test_regenerate_same_cmc_same_rarity_ok`.
2. **An old-rules draft left over from before the change** must not block or appear in the new builder, and must not take up a rarity. → Task 2, `test_legacy_draft_ignored_by_current_sets_and_rules`.
3. **P/T typed with spaces or leading zeros** (`" 2 "`, `"02"`) is accepted and stored as `"2"`. A blank pair falls back to auto, but `"" / "3"` is a 400. → Task 1, `test_pt_whitespace_and_leading_zero`.
4. **The owner changes their vote** between versions: the ×2 weight moves with it, and the history of a closed event still shows weighted counts. → Task 2, `test_self_vote_weight_moves_with_vote_change`.
5. **A bad P/T or rarity on Generate** must not abandon the existing draft at that CMC or queue cards. → Task 3, `test_invalid_commander_request_keeps_draft`.

---

### Task 1: Commander rules module

**Files:**
- Create: `proxy-server/commander_rules.py`
- Modify: `proxy-server/power_level.py` (extract `lean`)
- Create: `proxy-server/tests/test_commander_rules.py`

**Interfaces:**
- Produces:
  - `COMMANDER_CMCS: tuple[int, ...]`, `COMMANDER_RARITIES: tuple[str, ...]`, `VEHICLE_BONUS: int`
  - `stat_points(cmc: int, is_vehicle: bool) -> int`
  - `auto_stats(cmc: int, is_vehicle: bool, subtype: str) -> tuple[int, int]`
  - `commander_params(card_data: dict, cmc: int) -> dict`. It raises `storage.StorageError(400, ...)`. Import `StorageError` from `storage`. `storage` must not import `commander_rules`, so there is no cycle.
  - `power_level.lean(power: int, toughness: int, subtype: str) -> tuple[int, int]`: the creature-type shift now inside `creature_stats` (wall/treefolk/golem/construct/turtle → toughness, goblin/berserker/warrior/dragon/demon/cat/rogue → power). `creature_stats` calls it, and its behavior is unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/test_commander_rules.py`. Use `BASE = {"name": "Zur", "manaCost": "{W}{U}", "colors": ["W","U"], "subtype": "Human Wizard", "rarity": "Rare"}`. Port the pip cases from `tests/test_commander_slots.py`, then delete that file in Task 3.
  - `test_cost_padded_to_each_cmc`: `{W}{U}` gives `{1}{W}{U}`, `{2}{W}{U}` and `{3}{W}{U}` for 3, 4 and 5, with `cmc` set. `{X}{4}{B}{B}` at 5 → `{3}{B}{B}`. `""` at 3 → `{3}`. `{W/U}{B/P}` at 3 → `{1}{W/U}{B/P}`. `{2/W}{G}` at 3 → `{2/W}{G}`. The input dict is not mutated.
  - `test_too_many_pips_is_400`: `{W}{W}{U}{U}` → status 400.
  - `test_bad_cmc_is_400`: cmc 2 and cmc 6 → 400.
  - `test_creature_kind`: default kind → `supertype "Legendary"`, `type "Creature"`, subtype unchanged, and no `commanderKind` key.
  - `test_vehicle_kind`: `commanderKind "vehicle"`, subtype `""` → `type "Artifact"`, subtype `"Vehicle"`. Subtype `"Construct"` → `"Construct Vehicle"`. Subtype `"Vehicle"` stays `"Vehicle"`. Kind `"planeswalker"` → 400.
  - `test_rarity`: `"Uncommon"`, `"rare"` and `"MYTHIC"` are stored lowercased. `"common"`, `""` and a missing rarity → 400 with message `"Commanders are Uncommon, Rare or Mythic"`.
  - `test_stat_points`: `stat_points(3, False) == 4`, `stat_points(5, False) == 6`, `stat_points(3, True) == 6`.
  - `test_pt_within_budget`: at 3 CMC, `3/1`, `1/3`, `0/1` and `1/1` are accepted and stored as strings. A 3 CMC Vehicle accepts `4/2`.
  - `test_pt_rejections`: at 3 CMC, `3/2` → 400 whose message equals `"A 3-mana commander has 4 points; 3/2 uses 5"`. `*`/`*`, `X`/`2`, `1+*`/`2`, `-1`/`3`, `2/0` and `1.5`/`1` → 400. The non-number cases' message equals `"P/T must be whole numbers — X and * aren't allowed"`.
  - `test_pt_whitespace_and_leading_zero`: `" 2 "`/`"02"` → `"2"`/`"2"`. `""`/`"3"` → 400. `""`/`""` and missing → auto.
  - `test_auto_stats`: `auto_stats(3, False, "Human")` = (2, 2); `(4, False, "Human")` = (2, 3); `(5, False, "Goblin")` = (4, 2); `(5, True, "Vehicle")` = (4, 4); `(4, False, "Wall")` = (1, 4). Sums equal `stat_points`.
  - `test_power_level_lean_unchanged`: `power_level.creature_stats` gives the same results as before for `{"cmc": 4, "rarity": "rare", "subtype": "Goblin"}` and `{"cmc": 3, "subtype": "Wall"}`. Record the current values first by running the function before the refactor.

- [ ] **Step 2: Run** `python -m pytest tests/test_commander_rules.py -v`. Expected: FAIL (`ModuleNotFoundError: commander_rules`).

- [ ] **Step 3: Implement** `power_level.lean` (move the regex block out of `creature_stats`) and `commander_rules.py` per the interfaces and the spec §2. Pip parsing reuses the regex and pip-value logic from `storage.commander_slot_params` (copy it here; Task 3 deletes the original). The auto split is `power = points // 2`, `toughness = points - power`, then `lean`.

- [ ] **Step 4: Run** `python -m pytest tests/test_commander_rules.py tests/test_power_level.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** with the message `Commander rules: point-buy P/T, Creature or Vehicle, U/R/M rarity`.

---

### Task 2: Storage: one set per commander

**Files:**
- Modify: `proxy-server/storage.py` (migration, `_SET_COLS`, `create_set`, `current_set` → `current_sets`, `lock_set`, `unlock_set`, `locked_sets` order, `vote_tally`, new `owner_vote`)
- Modify: `proxy-server/tests/test_storage_sets.py`, `proxy-server/tests/test_storage_votes.py`

**Interfaces:**
- Consumes: nothing from Task 1. `storage` must not import `commander_rules` (that module imports `StorageError` from `storage`). `current_sets` takes the newest row per non-NULL `cmc`.
- Produces:
  - `Storage.create_set(user_id, commander_name, prompt, card_params, *, cmc: int, rarity: str) -> dict`, with the set dict now carrying `cmc` and `rarity` keys
  - `Storage.current_sets(user_id) -> list[dict]`, in CMC order (`current_set` is removed)
  - `Storage.vote_tally(set_id) -> dict[str, int]`, weighted
  - `Storage.owner_vote(set_id) -> str | None`: the card id the set's owner voted for

- [ ] **Step 1: Update the test helpers and write failing tests.** `make_set(storage, user_id, name=..., status="done", cmc=4, rarity="mythic")` passes `cmc`/`rarity` and uses params with that cmc. Fix the existing tests that relied on one-draft-per-user or one-lock-per-event semantics: `test_create_set_abandons_previous_draft` (now same CMC), `test_current_set_*` (→ `current_sets`), `test_lock_preconditions` (the "already locked" case is now same CMC) and `test_unlock_abandons_other_draft` (same CMC). New tests:
  - `test_drafts_at_different_cmcs_coexist`: create 3/uncommon, 4/rare and 5/mythic → `current_sets` returns 3 sets with cmc `[3, 4, 5]`, all drafts.
  - `test_regenerate_same_cmc_same_rarity_ok`: create 4/rare twice → the first is `abandoned`, the second is the draft, and no error.
  - `test_rarity_conflict_on_create`: a 3/rare draft exists → creating 4/rare raises 409 with message `"You already have a Rare commander (3 CMC)"`. Same when the 3/rare is locked in the open event. No conflict when the 3/rare is locked in a **closed** event, or `abandoned`.
  - `test_locked_cmc_blocks_new_draft`: 4 CMC locked in the open event → `create_set(cmc=4)` raises 409 `"Your 4 CMC commander is locked in — unlock it to start over"`.
  - `test_lock_rules_per_cmc`: one locked 3 CMC; locking a 4 CMC works; a second 3 CMC draft can't exist (blocked on create), so test the lock-time rarity check by inserting a conflicting draft directly through SQL (`UPDATE sets SET rarity = ...`) and asserting lock → 409.
  - `test_unlock_abandons_only_same_cmc_draft`: drafts at 3 and 5 plus a locked 4. Unlocking the 4 leaves the 3 and 5 drafts alone.
  - `test_legacy_draft_ignored_by_current_sets_and_rules`: create a set, then `UPDATE sets SET cmc = NULL, rarity = NULL`. `current_sets` is `[]`. Creating 4/mythic works and abandons the legacy draft. Locking a legacy draft raises 409 `"This set was made under the old rules — start a new commander"`.
  - `test_reroll_keeps_params`: the rerolled card's `card_params` equal the original's (cmc, rarity, type, power, toughness).
  - `test_locked_sets_ordered_by_cmc`: lock 5, then 3, then a legacy set (lock it with SQL) → `locked_sets` cmc order `[3, 5, None]`.
  - Votes (`test_storage_votes.py`), `test_self_vote_counts_two`: the owner votes for card A and another voter for card B → tally `{A: 2, B: 1}`, and `leader_flags` makes A the leader. `owner_vote(set_id) == A`.
  - `test_self_vote_weight_moves_with_vote_change`: the owner moves the vote to B → `{B: 3}`. After `close_event`, `vote_tally` is still `{B: 3}`.

- [ ] **Step 2: Run** `python -m pytest tests/test_storage_sets.py tests/test_storage_votes.py -v`. Expected: the new tests FAIL (unexpected keyword `cmc`, missing `current_sets`).

- [ ] **Step 3: Implement** per spec §3:
  - Migration: add `cmc INTEGER` and `rarity TEXT` to `sets` in `_migrate`, and add both to `_SET_COLS`.
  - `create_set`: checks and abandon in one `_tx()`; "live" means draft, or locked with `event_id` = the open event; only `cmc IS NOT NULL` rows count. The abandon covers same-CMC drafts **and** legacy drafts.
  - `lock_set`: legacy check first, then the per-CMC and rarity checks against sets locked in the open event.
  - `unlock_set`: abandon `draft` rows with the same `cmc`.
  - `locked_sets`: `ORDER BY cmc IS NULL, cmc, locked_at, rowid`.
  - `vote_tally`: `SUM(CASE WHEN v.voter_id = s.user_id THEN 2 ELSE 1 END)` joined to `sets`.
  - Rarity label in messages: `rarity.capitalize()`.

- [ ] **Step 4: Run** `python -m pytest tests/test_storage_sets.py tests/test_storage_votes.py tests/test_pool.py tests/test_director_storage.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** with the message `Storage: a set is one commander, rules checked per CMC and rarity, self-vote x2`.

---

### Task 3: API, and dropping the 3/4/5 slot params

**Files:**
- Modify: `proxy-server/api_routes.py` (`create_generations`, `my_current_set`, `set_view`)
- Modify: `proxy-server/storage.py` (delete `commander_slot_params`, `COMMANDER_SLOT_CMC`, `_MANA_SYMBOL_RE` if unused)
- Modify: `proxy-server/tools/e2e_sets.py` (each spec → one commander at a fixed CMC)
- Delete: `proxy-server/tests/test_commander_slots.py`
- Modify: `proxy-server/tests/test_api.py`

**Interfaces:**
- Consumes: `commander_rules.commander_params`, `Storage.create_set(..., cmc=, rarity=)`, `Storage.current_sets`, `Storage.owner_vote`.
- Produces (HTTP):
  - `POST /generations` `{prompt, cardData, count: 3, commanderName, cmc}` → `{setId, cards}`. Missing or invalid `cmc` → 400.
  - `GET /me/sets/current` → `SetView[]`.
  - `SetView` adds `"cmc"` and `"rarity"`. Each `cards[]` item adds `"ownerVote": bool`.
  - `e2e_sets.build_set(spec, use_director, client, model)` keeps its signature. `SET_SPECS` entries gain `"cmc"` (Zur'ka 3, Mother Thornwild 4, Ixen 5, Bronze Warden 4) and `"rarity"` in `uncommon|rare|mythic`. All 3 versions use `commander_params(spec, spec["cmc"])`.

- [ ] **Step 1: Write failing tests** in `tests/test_api.py`. Add `"cmc", "rarity"` to `SET_VIEW_KEYS`. Commander requests send `"cmc": 4`. Rename the `CARD_DATA` fixture use for sets so its rarity is `rare`.
  - `test_commander_set_cards_share_params`: the 3 cards' `card` have the same `manaCost` (`{2}{B}{R}` for pips `{B}{R}` at 4), `cmc` 4, `rarity "rare"`, and power/toughness `"2"`/`"3"` (use a subtype with no lean, e.g. `"Human"`). Slots are 1, 2, 3.
  - `test_commander_set_requires_cmc`: missing `cmc`, `cmc: 6` and `cmc: "4"` → 400.
  - `test_invalid_commander_request_keeps_draft`: make a 4 CMC draft, then POST at 4 CMC with `power: "*"` → 400. `current_sets` still has the original draft as `draft`, and `count_pending` is unchanged.
  - `test_vehicle_commander`: `commanderKind: "vehicle"`, `power: "4"`, `toughness: "2"` at 3 → the cards have `type "Artifact"`, a subtype containing `Vehicle`, and P/T 4/2.
  - `test_rarity_conflict_is_409`: a 3/rare, then a 4/rare → 409 with the exact message.
  - `test_me_sets_current_is_a_list`: no sets → `[]`. After 3 and 5 → two views with `cmc` `[3, 5]`.
  - `test_owner_vote_flag`: the owner votes → that card has `ownerVote: true` and `votes: 2`, and the other cards have `false`.
  - The existing count-1 tests stay exactly as they are and must still pass without `cmc`.
  - Update the existing commander tests (the ones at lines ~72, 166–200, 306, 428–439, 491 and 581) to send `cmc` and read the list shape.

- [ ] **Step 2: Run** `python -m pytest tests/test_api.py -v`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement.** In `create_generations` (count 3):
  - Validate that `cmc` is an `int` in `COMMANDER_CMCS` (type check like `count`).
  - `params = commander_params({**card_data, "name": commander_name}, cmc)` **before** `create_lock`.
  - `storage.create_set(..., cmc=cmc, rarity=params["rarity"])`, then 3 × `create_card(..., params, set_id, slot)`.

  `my_current_set` returns `[set_view(s, user["id"]) for s in storage.current_sets(user["id"])]`. `set_view` adds `cmc` and `rarity`, and per card `"ownerVote": c["id"] == owner_vote_id`. Delete the old slot helpers and their test file, and update `e2e_sets.py` plus its docstring (the versions are now 3 takes at one CMC).

- [ ] **Step 4: Run** `python -m pytest tests -m "not slow" -v`. Expected: all PASS. Also run `python -c "import tools.e2e_sets"` from `proxy-server/` (it imports cleanly).

- [ ] **Step 5: Commit** with the message `API: commander sets take a CMC; me/sets/current lists one per CMC`.

---

### Task 4: Frontend models and the shared commander rules helper

**Files:**
- Modify: `src/app/models/api.model.ts`, `src/app/models/card.model.ts`
- Create: `src/app/services/commander-rules.ts`, `src/app/services/commander-rules.spec.ts`
- Modify: `src/app/services/event.service.ts` (+ spec), `src/app/services/generation.service.ts` (`cardParams`)
- Modify: `src/app/testing/fixtures.ts` (fixture `SetView`s gain `cmc: 4, rarity: 'rare'`; cards gain `ownerVote: false`)

**Interfaces:**
- Produces:
  - `SetView.cmc: number | null`, `SetView.rarity: string | null`
  - `SetCardView.ownerVote: boolean`
  - `GenerationRequest.cmc?: number`
  - `CardParams.commanderKind?: CommanderKind`; `Card.commanderKind?: CommanderKind`
  - `EventService.mySets(): Observable<SetView[]>` (replaces `mySet`)
  - From `commander-rules.ts`:
    - `type CommanderKind = 'creature' | 'vehicle'`
    - `COMMANDER_CMCS = [3, 4, 5]`
    - `COMMANDER_RARITIES: Rarity[] = [UNCOMMON, RARE, MYTHIC]`
    - `commanderPipValue(manaCost: string): number` (moved from the set builder page)
    - `statPoints(cmc: number, kind: CommanderKind): number`
    - `commanderStatsError(power: string, toughness: string, cmc: number, kind: CommanderKind): string | null` (same rules and the same message strings as the server)
    - `groupByCmc(sets: SetView[]): { label: string; cmc: number | null; sets: SetView[] }[]`: groups in order 3, 4, 5, then `cmc === null` labelled `'Earlier sets'`, with empty groups omitted and labels like `'3 CMC'`
- `cardParams` copies `commanderKind` when set.

- [ ] **Step 1: Write failing specs** in `commander-rules.spec.ts`:
  - `statPoints(3,'creature')===4` and `statPoints(3,'vehicle')===6`
  - `commanderStatsError('','',3,'creature')===null`
  - `('3','1',3,'creature')===null`
  - `('3','2',3,'creature')==='A 3-mana commander has 4 points; 3/2 uses 5'`
  - `('*','*',3,'creature')`, `('X','2',…)` and `('1.5','1',…)` return the whole-numbers message
  - `('2','0',…)` and `('','3',…)` are non-null
  - `(' 2 ','02',3,'creature')===null`
  - `groupByCmc` on sets with cmc `[5, null, 3, 3]` → labels `['3 CMC', '5 CMC', 'Earlier sets']` with counts `[2, 1, 1]`

  In `event.service.spec.ts`, `mySets` GETs `/me/sets/current` and returns the array.

- [ ] **Step 2: Run** `npx ng test --watch=false --include src/app/services/commander-rules.spec.ts`. Expected: FAIL (cannot find module).

- [ ] **Step 3: Implement** the helper and model changes. Compiler errors in the set builder page from removing `mySet` are expected and are fixed in Task 6. Meanwhile, keep a temporary `mySet()` alias so the build stays green, and delete it in Task 6.

- [ ] **Step 4: Run** the two specs. Expected: PASS. Run `npx ng build --configuration development`. Expected: success.

- [ ] **Step 5: Commit** with the message `Frontend: commander rules helper, SetView cmc/rarity/ownerVote`.

---

### Task 5: Card form commander mode: kind, rarity, P/T budget

**Files:**
- Modify: `src/app/components/card-form/card-form.component.ts`, `.html`, `.scss`, `.spec.ts`

**Interfaces:**
- Consumes: `CommanderKind`, `COMMANDER_RARITIES`, `statPoints`, `commanderStatsError` (Task 4).
- Produces:
  - `@Input() commanderCmc = 4`
  - `@Input() takenRarities: Partial<Record<Rarity, number>> = {}` (rarity → the CMC that uses it)
  - The emitted `Card` in commander mode carries `commanderKind`, `power` and `toughness` (raw strings, possibly empty), `type`/`supertype`/`subtype` per kind, and `rarity`.
  - `get statsError(): string | null` and `get statsHint(): string`
  - Exact hint copy: `"{points} points · e.g. 2/2, 3/1, 1/3 · leave blank for auto"`, or for a Vehicle `"{points} points (Vehicle +2) · leave blank for auto"`.

- [ ] **Step 1: Write failing specs** (component spec, commander mode):
  - `rarity options are Uncommon, Rare, Mythic only`
  - `a taken rarity is disabled with "used at 4 CMC"`: `takenRarities = {rare: 4}` → the Rare input is disabled and the label text contains `used at 4 CMC`
  - `vehicle toggle sets Artifact with Vehicle subtype`: the emitted card has `type 'Artifact'`, `supertype 'Legendary'`, a subtype containing `'Vehicle'` and `commanderKind 'vehicle'`
  - `creature toggle sets Legendary Creature`
  - `P/T field shows the budget hint for commanderCmc 3`: the text contains `4 points`; as a Vehicle, `6 points (Vehicle +2)`
  - `P/T 3/2 at 3 CMC shows the over-budget error`, and `statsError` equals the Task 4 message
  - `blank P/T emits empty power/toughness and no error`
  - `art prompt scale follows commanderCmc` (3 → `medium scale`, 5 → `large and imposing`)
  - `normal mode is unchanged`: the Common option is present and the P/T visibility rules are as before

- [ ] **Step 2: Run** `npx ng test --watch=false --include src/app/components/card-form/card-form.component.spec.ts`. Expected: the new specs FAIL.

- [ ] **Step 3: Implement.**
  - Add a `commanderKind` form control (default `'creature'`).
  - `applyCommanderDefaults` and a kind subscription set type/supertype, and append `Vehicle` to the subtype for a Vehicle (removing it again when switched back to creature).
  - In commander mode, show the rarity chips filtered to `COMMANDER_RARITIES`, plus the P/T input (the existing `powerToughness` control) with the hint/error under it.
  - In commander mode, `onFormValueChanges` keeps `power`/`toughness`.
  - Replace `COMMANDER_ART_CMC` with `commanderCmc`.
  - Update the `commanderMode` doc comment.
  - Follow the light, slim control style already used in the form.

- [ ] **Step 4: Run** the spec. Expected: PASS.

- [ ] **Step 5: Commit** with the message `Card form: commander kind, U/R/M rarity and point-buy P/T`.

---

### Task 6: Set builder: three commanders

**Files:**
- Create: `src/app/components/commander-panel/commander-panel.component.{ts,html,scss,spec.ts}`. This is today's per-set logic from `set-builder-page.component.ts`, scoped to one CMC.
- Modify: `src/app/pages/set-builder-page/set-builder-page.component.{ts,html,scss,spec.ts}`
- Modify: the module that declares components (`src/app/app.module.ts`)
- Modify: `src/app/services/event.service.ts` (remove the temporary `mySet` alias)

**Interfaces:**
- Consumes: `EventService.mySets()`, `GenerationService.submit/reroll/watch`, `commanderPipValue`, `commanderStatsError`, `COMMANDER_CMCS`, the card form inputs from Task 5.
- Produces:
  - `CommanderPanelComponent`:
    - `@Input() cmc: number`, `@Input() set: SetView | null`, `@Input() event: EventView | null`, `@Input() takenRarities`
    - `@Output() setChange = new EventEmitter<SetView | null>()`, emitted after generate/lock/unlock so the page can recompute taken rarities and the summary
    - Generate sends `{prompt, cardData: {...cardParams(card), name}, count: 3, commanderName, cmc}`
    - Slot labels are `Version 1..3`
  - `SetBuilderPageComponent`:
    - holds `sets: Record<number, SetView | null>`
    - `takenRaritiesFor(cmc): Partial<Record<Rarity, number>>`, built from the other CMCs' sets
    - `summary(cmc): string`, e.g. `'Uncommon · Locked'`, `'Rare · Draft'` or `'Not started'`
    - polls the event as today and passes it down

- [ ] **Step 1: Move and adapt the existing specs** from `set-builder-page.component.spec.ts` into `commander-panel.component.spec.ts` (generate, reroll, lock, unlock, rename, pending, event-closed cases), now driven through the panel's inputs. Add:
  - `generate sends cmc and count 3`
  - `generate is disabled with a hint when the rarity is taken`: `takenRarities = {rare: 3}`, form rarity rare → `canGenerate()` is false and the hint mentions `3 CMC`
  - `generate is disabled with the P/T error when stats are invalid`
  - `reroll asks the rule-6 question and does nothing on cancel`: spy `window.confirm` returning false → `generation.reroll` is not called, and the confirm text equals the exact copy
  - `reroll proceeds on OK`

  Page spec:
  - `loads mySets into panels by cmc`: `[{cmc:3}, {cmc:5}]` → panel 4 gets null
  - `taken rarities exclude the panel's own CMC`
  - `summary strip text`
  - `a panel's setChange updates the others' taken rarities`

- [ ] **Step 2: Run** `npx ng test --watch=false --include src/app/components/commander-panel/commander-panel.component.spec.ts` and the page spec. Expected: FAIL (component missing).

- [ ] **Step 3: Implement.**
  - The panel is today's page logic minus the event polling.
  - The page has the intro copy explaining the rules (three commanders, CMC+1 points, +2 for Vehicles, one each of Uncommon/Rare/Mythic, 3 versions each, lock each in), then the summary strip, then three panels.
  - On phones (< 720px) show one panel at a time behind a 3-way tab row, keeping the last tab in `localStorage` (try/catch). On wider screens the panels stack vertically with a heading per CMC.

- [ ] **Step 4: Run** both specs and `npx ng build --configuration development`. Expected: PASS and a successful build.

- [ ] **Step 5: Commit** with the message `Set builder: three commanders, one panel per CMC`.

---

### Task 7: Vote page, winners and history grouped by CMC; ×2 marker

**Files:**
- Modify: `src/app/pages/vote-page/vote-page.component.{ts,html,spec.ts}`
- Modify: `src/app/components/set-row/set-row.component.{html,scss}` (+ spec if present)
- Modify: `src/app/components/winners-banner/winners-banner.component.{ts,html}`
- Modify: `src/app/pages/event-history-page/event-history-page.component.{ts,html,spec.ts}`

**Interfaces:**
- Consumes: `groupByCmc`, `SetCardView.ownerVote`.
- Produces: no new public API. Each grouped page exposes `groups = groupByCmc(ev.sets)` (recomputed when the event refreshes).

- [ ] **Step 1: Write failing specs.**
  - Vote page:
    - `groups sets under 3/4/5 CMC headings`: headings appear in order and each set sits under its CMC
    - `legacy sets go under Earlier sets`
    - `header copy`: contains the exact vote header copy
  - Set row: `shows owner ×2 on the owner's pick`: a card with `ownerVote: true` renders the text `owner ×2`, and the others don't
  - Winners banner: `groups winners by CMC`
  - Event history: `groups a closed event's sets by CMC`

- [ ] **Step 2: Run** the four specs with `--include`. Expected: FAIL.

- [ ] **Step 3: Implement.**
  - The vote page and event history render `<h3 class="section-title">{{ g.label }}</h3>` followed by that group's `app-set-row`s.
  - The winners banner groups its `<li>`s under the same labels.
  - The set row adds `<span *ngIf="card.ownerVote" class="owner-x2 muted">owner ×2</span>` beside the count.
  - Update the empty-state copy from "set" to "commander" where it refers to the new flow.

- [ ] **Step 4: Run** the specs, then the full suite with `npm test -- --watch=false`. Expected: all PASS.

- [ ] **Step 5: Commit** with the message `Vote: commanders grouped by CMC, owner vote shown as x2`.

---

### Task 8: Docs and the generation check

**Files:**
- Modify: `CLAUDE.md` (Architecture: describe a set as one commander and the commander rules module; replace the 3/4/5-versions wording in the director and e2e `--sets` bullets)
- Modify: `docs/superpowers/specs/2026-09-28-ai-night-design.md` (a one-line note at the top: sets are now one commander each, see the 2026-10-03 spec)

- [ ] **Step 1: Update the docs** as listed, and keep the wording consistent with the spec.
- [ ] **Step 2: Ask the user before running the generation check.** It uses Ollama on the live machine. Once they agree, run `python tools/e2e_rules_text.py --label commander-rules --sets 4` from `proxy-server/`. Compare `over_budget_*` in `data/e2e/commander-rules/report.md` with the latest earlier `--sets` report and report both numbers to the user. If any card is more than `ERROR_OVER` over budget, report that instead of tuning.
- [ ] **Step 3: Run** `python -m pytest tests -m "not slow"` and `npm test -- --watch=false` one last time. Expected: all PASS.
- [ ] **Step 4: Commit** with the message `Docs: commander rules`.
