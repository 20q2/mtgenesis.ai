# Knowledge Pool: shared custom cards for the real event

**Status:** Built 2026-09-29 with slots and bans. Reworked 2026-10-01 (this version) into one voted list.

## 1. Purpose

At the paper Magic event, every player may use a shared pool of custom cards (the "Knowledge Pool"). Last year each of 8 players simply put 2 cards in, and nothing checked the result: one player's *Pot of Green*, a 0-mana artifact that drew three cards, went into every deck.

This year players submit cards, everyone votes with medals, and the best-voted half make the pool. The first version split the pool into color/type slots with secret bans. It was never used at a real event, and it was more machinery than the table wanted, so this version replaces it with one flat, open vote.

## 2. Rules

| Topic | Rule |
|---|---|
| Lifecycle | The host opens a pool from `/admin` with a name and a per-player entry cap (1–10, default **3**). Only one pool can be open at a time. Closing freezes it for good, and past pools stay viewable. The pool is independent of commander-set events. |
| Eligibility | Only finished cards the player owns. **Only colorless or mono-colored cards**: a card's colors are its `colors` list, else the WUBRG symbols in its mana cost (`card_colors`); more than one color is rejected. |
| Entries | At most `maxEntriesPerUser` cards per player. A card can be entered once. Players submit from the create screen (the card they just made) or from a finished tile in the gallery's My cards. They can withdraw an entry from `/pool` while the pool is open, which also removes that card's medals. |
| Medals | Each player has **one gold (3 points), one silver (2) and one bronze (1) for the whole pool**. Giving a medal that is already on another card moves it. A player gives at most one medal per card, and **never to their own card**. |
| Cutoff | **N = ⌊submitters / 2⌋**, where submitters are players with at least one entry. It is computed live, so the line moves as players join. |
| Ranking | Entries rank by points, then golds, then silvers. The top N are **in**. Every card level with the Nth card on points, golds and silvers is also in (`tiedAtCutoff`), so the pool can be slightly bigger than N. A card with 0 points is never in. |
| Visibility | Everything is open while voting: who submitted each card, live medal counts and points. |
| Power check | Each entry shows `power_level.assess(rules text, card)` as *Fair*, *Pushed* (≥ 1 mana over its budget) or *Over the curve* (≥ 2 over). It is advisory, to make a Pot of Green obvious to voters. |

### Examples

- 8 submitters: N = 4. Points 9, 7, 6, 6, 6, 2. The three 6-point cards have 1, 0, 0 golds; the two 0-gold cards have 2 and 1 silvers. In: 9, 7, the 6 with a gold, and the 6 with 2 silvers.
- Same, but the last two 6-point cards both have 0 golds and 2 silvers: both are in (5 cards).
- 3 submitters: N = 1. 1 submitter: N = 0, nothing is in.

## 3. Storage (`storage.py`)

Tables:
- `pools(id, name, status open|closed, max_entries_per_user, created_at, closed_at)`, at most one open (unchanged)
- `pool_entries(id, pool_id, card_id, user_id, created_at)`, unique on `(pool_id, card_id)`
- `pool_medals(voter_id, pool_id, entry_id, medal gold|silver|bronze, created_at)`, unique on `(voter_id, pool_id, medal)` and `(voter_id, entry_id)`

`pool_slots` and `pool_bans` are removed, along with `card_fits_slot`, `slot_rule_text`, `clean_pool_slots`, `POOL_BANS_PER_PLAYER` and `POOL_BAN_THRESHOLD`.

**Migration.** On startup, if `pool_slots` exists (the old shape): when `pools`, `pool_slots`, `pool_entries`, `pool_medals` and `pool_bans` are all empty, drop them and create the new tables. If any of them holds rows, raise an error and refuse to start rather than lose data. (The live database had no pools when this was written.)

**Ranking** is a pure function, `pool_ranking(entries, medal_counts) -> {entry_id: {gold, silver, bronze, points, rank, in, tiedAtCutoff}}`, with the cutoff derived from the distinct `user_id`s in `entries`. `rank` is 1-based and equal keys share a rank.

Storage methods: `create_pool(name, max_entries_per_user)`, `close_pool`, `current_pool`, `get_pool`, `list_pools`, `pool_entries`, `pool_medal_counts`, `my_pool_medals`, `submit_pool_entry(user_id, card_id)`, `withdraw_pool_entry`, `award_pool_medal`, `clear_pool_medal`, and `open_pool_entry_ids() -> {card_id: entry_id}` for the open pool (one query per list request).

## 4. API (`api_routes.py`, under `/api/v1`)

| Method and path | Body | Returns |
|---|---|---|
| `GET /pools/current` | | `PoolView` or `null` (ETag) |
| `GET /pools` | | `PoolSummary[]`, newest first |
| `GET /pools/<id>` | | `PoolView` (ETag) |
| `POST /pools/entries` | `{cardId}` | `PoolView` |
| `POST /pools/entries/<id>/withdraw` | | `PoolView` |
| `POST /pools/medals` | `{entryId, medal}` | `PoolView` |
| `POST /pools/medals/clear` | `{entryId}` | `PoolView` |
| `POST /admin/pools` | `{name, maxEntriesPerUser}` | `PoolView` (admin PIN) |
| `POST /admin/pools/<id>/close` | | `PoolView` (admin PIN) |

The ban routes are removed.

Errors on `POST /pools/entries`: 404 unknown card or no open pool, 403 not your card, 400 unfinished card or more than one color ("Only colorless or mono-colored cards can enter the pool"), 409 already entered or cap reached.

Shapes (`src/app/models/api.model.ts`):

```ts
interface PoolView extends PoolSummary {
  submitters: number;        // players with at least one entry
  cutoff: number;            // floor(submitters / 2)
  myEntryCount: number;
  myMedals: Record<Medal, string | null>;   // entry id each of my medals is on
  entries: PoolEntryView[];  // sorted by rank, then oldest first
}

interface PoolEntryView {
  id: string; cardId: string; card: CardView;
  username: string; mine: boolean; power: PowerCheck | null;
  gold: number; silver: number; bronze: number; points: number;
  rank: number; in: boolean; tiedAtCutoff: boolean;
  myMedal: Medal | null; createdAt: string;
}
```

`PoolView` is personalized by `X-User-Id` (`mine`, `myMedal`, `myMedals`, `myEntryCount`).

`CardView` gains `poolEntryId: string | null`: the card's entry in the open pool, so the create screen and gallery can show "In the pool".

## 5. Frontend

- **`/pool` (the "Knowledge Pool" nav item):**
  - Header: pool name and status; a rules line ("Colorless or mono-colored cards only · up to 3 per player · gold 3 / silver 2 / bronze 1 · the top half make the pool"); "*S* players → top *N* make the pool"; "your entries *k* / cap"; three medal chips lit when given.
  - One grid of entries sorted by rank. Each tile: the card, maker, power check, points and medal counts, then Gold / Silver / Bronze buttons (pressed when it's the viewer's medal) or Withdraw on the viewer's own card.
  - A visible "pool line" divider after the last card that is in. Cards above it carry an "In" badge, and cards tied at the line "Tied · in".
  - Empty state: points to the create screen.
  - Closed pool: no buttons; a **"The pool"** section at the top shows the cards that are in.
- **`/pool/:id`:** a past pool, read-only, same layout.
- **Submit button** on the create screen (once the card is done) and on finished My cards tiles: "Submit to pool (k/cap)", then "In the pool ✓". Disabled with a visible reason: "Multicolor cards can't enter the pool", "You've used all *cap* entries", "No pool is open". `poolColorOk(card)` in `pool.service.ts` mirrors `card_colors`; the server has the final say. The open pool comes from `PoolService.current()` once per page load.
- **`/admin`:** the pool form is just name and cap (default 3); the slot editor is gone.

## 6. Testing

- `tests/test_pool.py`: color eligibility (mono, colorless, multicolor, hybrid), cap, duplicate card, own-card-only, medals unique per pool and moving, no self-medals, `pool_ranking` (cutoff, golds then silvers, tie at the line all in, 0 points never in, 0/1/3 submitters), withdraw drops medals, closed pool rejects changes, migration drops only empty old tables and refuses non-empty ones, `poolEntryId` on CardView.
- Frontend specs: pool page (sorting, pool line, badges, medal buttons, closed view), pool-admin form, `poolColorOk`, and the Submit buttons on the create page and gallery.
