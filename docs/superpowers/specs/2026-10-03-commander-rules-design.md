# Commander rules: three commanders per player, three versions each

**Status:** Design approved 2026-10-03.

## 1. Purpose

AI Night now plays by a written rule sheet. The commander set flow has to enforce it. Today a "set" is one concept rendered at 3, 4 and 5 mana, and the vote picks one winner per player. The rules ask for something different:

1. Each player makes **three commanders**: one at exactly 3 CMC, one at 4 and one at 5. Multicolor is fine. Each is a Legendary **Creature or Vehicle**.
2. **P/T is a point buy.** A commander has CMC + 1 points (a Vehicle gets 2 more). Not every point has to be spent. A P/T of X or * is illegal.
3. **Vehicles:** the crew cost is left to the AI.
4. **Rarity:** across a player's three commanders, exactly one Uncommon, one Rare and one Mythic, assigned however the player likes.
5. **Generation:** each commander is generated **3 times** (3 versions, no tweaking).
6. **Rerolls** are only for a version that doesn't function under any circumstance.
7. **Voting:** each player votes for one version of each commander, with separate votes for 3, 4 and 5 CMC. A vote on your own commander counts as **two**. Owners should vote first.
8. **Winners:** the version with the most votes is legal. Each player ends with three legal commanders.

**Scope:** only the commander set flow (`/set`, `/vote`, event history). Normal create (`/create`, count 1) is unchanged.

Decisions from the design discussion:
- The three commanders are **independent designs**: each has its own name, prompt, pips, kind and rarity.
- **P/T:** the player may type one that fits the budget. If they leave it blank, the stat curve picks a split that spends every point. All three versions of a commander share the body.
- **Rerolls:** honor system. The Reroll button asks the rule-6 question first. There is no hard limit.
- **Locking:** each commander locks separately as soon as its 3 versions are done.
- **Model (approach A):** one `sets` row is **one commander** (3 versions plus a vote), and it gains a `cmc` column. The rules are checked across the player's sets. The alternative, a new `lineups` table, was rejected: it adds a lifecycle without enforcing anything more.

## 2. Rules module (`proxy-server/commander_rules.py`, new)

These are pure functions with no storage and no Flask. Errors are `StorageError(400, message)`, worded for players.

```python
COMMANDER_CMCS = (3, 4, 5)
COMMANDER_RARITIES = ("uncommon", "rare", "mythic")
VEHICLE_BONUS = 2

def stat_points(cmc: int, is_vehicle: bool) -> int        # cmc + 1 (+ 2 for a Vehicle)
def auto_stats(cmc: int, is_vehicle: bool, subtype: str) -> tuple[int, int]
def commander_params(card_data: dict, cmc: int) -> dict
```

`commander_params` builds the card params shared by all three versions:

- **CMC:** must be in `COMMANDER_CMCS`, else 400.
- **Mana cost:** keeps the requested colored, hybrid and Phyrexian pips and drops generic and X/Y/Z. The pips may be worth up to the commander's own CMC (`{2/W}` counts 2), else 400. (Was ≤ 3 for every commander, a leftover from one design printed at 3/4/5; changed 2026-10-03.) Generic is padded up to the CMC, for example `{W}{U}` at 4 becomes `{2}{W}{U}`. This is today's `commander_slot_params` logic, moved here.
- **Kind** (`card_data["commanderKind"]`, default `"creature"`):
  - `"creature"` → `supertype "Legendary"`, `type "Creature"`. The subtype is the player's.
  - `"vehicle"` → `supertype "Legendary"`, `type "Artifact"`, subtype exactly `Vehicle` (anything typed is ignored).
  - A creature's subtype may only hold creature types: other cards' subtypes and type words (Equipment, Aura, Vehicle, Instant, Legendary…) are a 400 ("Equipment isn't a creature type").
  - Anything else → 400.
- **Rarity:** lowercased, must be in `COMMANDER_RARITIES`, else 400 ("Commanders are Uncommon, Rare or Mythic").
- **P/T:**
  - If `power` and `toughness` are both blank or missing, use `auto_stats`.
  - If exactly one is given → 400.
  - Otherwise both must be whole numbers written as digits. `*`, `X`, `1+*`, decimals and negatives → 400 ("P/T must be whole numbers — X and * aren't allowed").
  - Power must be ≥ 0, toughness ≥ 1, and power + toughness ≤ `stat_points`, else 400 ("A 3-mana commander has 4 points; 3/2 uses 5").
  - Stored as strings, like the rest of the card params.
- **`auto_stats`:** spends every point. Power is `points // 2` and toughness gets the rest. It leans one point toward toughness or power using the same creature-type regexes as `power_level.creature_stats`. Those regexes are moved into a shared helper `power_level.lean(power, toughness, subtype)` so both callers use one copy. Examples: a 3 CMC creature is 2/2, a 4 CMC creature is 2/3, a 5 CMC Goblin is 4/2 and a 5 CMC Vehicle is 4/4.
- **Output:** a copy of the input with `manaCost`, `cmc`, `type`, `supertype`, `subtype`, `rarity`, `power` and `toughness` set, and `commanderKind` removed.

Because P/T is always set, `finalize_card` never makes up a body for a commander. `power_level.budget` already subtracts the printed body, so a smaller body leaves the abilities more room. That is deliberate and should be checked in the e2e run (§7). Vehicle crew is still added by `finalize_card` / `generate_vehicle_crew_cost` when the text lacks it.

`storage.commander_slot_params` and `COMMANDER_SLOT_CMC` are removed. The frontend's `COMMANDER_SLOT_CMC` becomes `COMMANDER_CMCS`.

## 3. Storage (`proxy-server/storage.py`)

**Migration** (`_migrate`): `ALTER TABLE sets ADD COLUMN cmc INTEGER` and `ADD COLUMN rarity TEXT`. Existing rows stay NULL. A set with `cmc IS NULL` is a **legacy set**: it keeps its stored params and is shown as it is today. Every rule below only looks at sets with a non-NULL `cmc`.

A user's **live** commanders are their `draft` sets plus their sets `locked` in the open event.

- **`create_set(user_id, commander_name, prompt, card_params, cmc, rarity)`**, all in one transaction:
  - 409 if the user has a set at this `cmc` locked in the open event: "Your 4 CMC commander is locked in — unlock it to start over".
  - 409 if a commander of the user at a **different** CMC is **locked in the open event** with this rarity: "You already have a Rare commander (4 CMC)". Drafts don't hold a rarity, so a player can reassign rarities while drafting; `lock_set` enforces one of each. (Changed after review, 2026-10-03: holding rarities on drafts stranded a player who had drafted all three.)
  - Abandon only the user's `draft` at the **same** `cmc`. Drafts at other CMCs are untouched, and legacy drafts are abandoned too.
  - Insert the set with `cmc` and `rarity`.
- **`current_sets(user_id) -> list[dict]`:** for each CMC in 3, 4, 5, the newest draft, or else the set locked in the open event. Returned in CMC order, with missing CMCs omitted. This replaces `current_set`.
- **`lock_set`:** unchanged checks (owner, draft, open event, all 3 current cards done, name). The check "one locked set per user per event" becomes:
  - 409 if the user already has a set at this `cmc` locked in the event.
  - 409 if a set of theirs locked in the event already has this rarity.
  - Legacy drafts can't be locked: 409 "This set was made under the old rules — start a new commander".
- **`unlock_set`:** still deletes the set's votes and returns it to draft. It now abandons only the user's other draft at the **same** `cmc`.
- **`reroll_card`:** unchanged. The new card copies the old one's params, so CMC, rarity, kind and P/T carry over.
- **`vote_tally(set_id)`:** a vote by the set's owner counts 2 (legacy sets keep one vote per voter, so past events' results don't change):
  `SUM(CASE WHEN v.voter_id = s.user_id THEN 2 ELSE 1 END)`, joined to `sets`. `leader_flags` is unchanged.
- **`cast_vote`:** unchanged (one vote per voter per set, and self-votes are allowed).
- `_SET_COLS` and the set decoder include `cmc` and `rarity`.

## 4. API (`proxy-server/api_routes.py`)

- **`POST /generations` with `count: 3`:**
  - Requires `cmc` in the body (3, 4 or 5) and `commanderName`.
  - Uses `cardData.rarity`, `cardData.commanderKind` and optional `cardData.power` / `cardData.toughness`.
  - Builds `params = commander_rules.commander_params({**cardData, name}, cmc)` before taking the lock, so a bad request fails without abandoning anything.
  - Calls `storage.create_set(..., cmc=cmc, rarity=params["rarity"])`, then creates 3 cards with identical `params`. Slots 1–3 are version numbers.
  - The response shape is unchanged. `count: 1` is unchanged.
- **`GET /me/sets/current`** now returns a **list** of `SetView` (0–3 items, in CMC order). The set builder is its only client.
- **`SetView`** gains `cmc: number | null` and `rarity: string | null`. `cards[].votes` is the weighted tally. Each card also gets `ownerVote: boolean`, true when the owner voted for it, so the UI can show "×2".
- **Event views** (`/events/current`, `/events/<id>`) list sets ordered by `cmc` (NULLs last), then by lock time.
- New routes are not needed. Existing ones keep `add_ngrok_headers` / `OPTIONS` handling.

**Director:** unchanged. A set still gets one call with 3 briefs that use different mechanics and one character, so the 3 versions are 3 takes on the same commander. A reroll avoids its siblings' mechanics.

## 5. Frontend

**Models** (`src/app/models/api.model.ts`): `SetView.cmc`, `SetView.rarity`, `CardView.ownerVote`. `GenerationRequest` gains `cmc`. `EventService.mySets()` returns `SetView[]`.

**Card form in commander mode** (`card-form.component`):
- Fields: colored pips (as now), a **Creature / Vehicle** toggle, subtype, and a **Rarity** chip row limited to Uncommon / Rare / Mythic.
- An optional **P/T** input with a budget hint: "4 points · e.g. 2/2, 3/1, 1/3 · leave blank for auto". For a Vehicle it reads "6 points (Vehicle +2)".
- Shown inline: X or *, non-numbers, a missing half, toughness 0 and over-budget. Generate stays disabled while any of these is shown.
- New inputs: `commanderCmc: number` (sets the budget and the art prompt's CMC, replacing `COMMANDER_ART_CMC`) and `takenRarities: {rarity: cmc}`. A taken rarity shows as disabled with "used at 4 CMC".
- `cardParams` sends `commanderKind`, `power` and `toughness`.

**Set builder** (`/set`, `set-builder-page.component`):
- **Summary strip:** "3 CMC · Uncommon · Locked ✓ | 4 CMC · Rare · Draft | 5 CMC · not started".
- **Three panels** (tabs on phones, one per CMC). Each has its own commander name, card form, 3 version slots labelled `Version 1..3`, and Generate / Lock in / Unlock / Change name. Today's single-set state becomes one state object per CMC.
- Loads from `mySets()`. `takenRarities` comes from the other CMCs' live sets.
- **Reroll** first asks `window.confirm("Rerolls are only for a version that doesn't function under any circumstance. Reroll it?")`.
- The intro copy explains the rules briefly: three commanders, CMC+1 points, one each of Uncommon / Rare / Mythic, 3 versions each, and lock each one in.
- Generate hints: too many pips, a taken rarity, and an invalid P/T. The server's 400/409 messages show in the existing error alert.

**Vote page** (`/vote`):
- Sets are grouped under **3 CMC**, **4 CMC** and **5 CMC** headings, then **Earlier sets** for legacy sets. Each commander uses the existing `app-set-row`.
- Header copy: "Vote for one version of each commander. A vote on your own commander counts as two — owners, vote first." Change it until the host closes the event.
- On a card with `ownerVote`, `app-set-row` shows a small "owner ×2" marker next to the count.

**Winners banner and event history:** winners are grouped by CMC using the same headings.

## 6. Error handling

- All rule checks run on the server. The UI checks the same rules for fast feedback but trusts the server's answer.
- Validation (`commander_params`) runs before `create_set`. So a bad P/T or rarity never abandons a draft and never queues cards.
- Rarity and CMC conflicts are checked inside the write transaction (`BEGIN IMMEDIATE`). Two tabs can't both create a Rare.
- The existing commander-name, pending-cap and event checks are unchanged.

## 7. Testing

- **`tests/test_commander_rules.py`** (new):
  - pips and padding at each CMC; pips worth more than the CMC rejected
  - both kinds; Vehicle subtype appended
  - each rarity accepted, common rejected
  - P/T accepted at and under budget; over budget, X, *, `1+*`, negative, toughness 0 and half-given rejected
  - `auto_stats` totals and leans
- **`tests/test_storage_sets.py` / `test_commander_slots.py`:**
  - updated for the per-CMC model: drafts at different CMCs coexist, and a same-CMC draft is replaced
  - rarity conflicts on create and lock
  - a locked CMC blocks a new draft at that CMC
  - unlock abandons only the same-CMC draft
  - reroll keeps the params
  - legacy sets are listed but can't be locked
- **`tests/test_storage_votes.py`:** a self-vote counts 2 and changes the leader; changing a vote moves the weight.
- **`tests/test_api.py`:** the `count: 3` contract (needs `cmc`; a bad P/T is a 400 and nothing is queued); `/me/sets/current` returns a list; `SetView` has `cmc`, `rarity` and `ownerVote`.
- **Frontend specs:**
  - set builder: per-CMC state, taken rarities, P/T validation, reroll confirm
  - card form: commander budget hint and validation
  - vote page: CMC grouping and the ×2 marker
- **Generation:** one `tools/e2e_rules_text.py` run with commander params at each CMC and kind, comparing `over_budget_*` with the last report. It needs Ollama, so it won't run while people are generating without asking first.
