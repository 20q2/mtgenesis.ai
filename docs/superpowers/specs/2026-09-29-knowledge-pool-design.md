# Knowledge Pool: shared custom cards for the real event

**Status:** Built 2026-09-29.

## 1. Purpose

At the paper Magic event, every player may use a shared pool of custom cards (the "Knowledge Pool"). Last year each of 8 players simply put 2 cards in (16 total). Nothing checked the result: one player's *Pot of Green*, a 0-mana artifact that drew three cards, went into every deck.

This year:
- the pool has **slots** so the cards cover a spread of colors and card types
- players may **submit several cards**
- a **group vote** decides which card fills each slot

## 2. Rules

| Topic | Rule |
|---|---|
| Lifecycle | The host opens a pool from `/admin` with a name, a per-player submission cap, and a slot list. Only one pool can be open at a time. Closing freezes it for good, and past pools stay viewable. The pool is independent of commander-set events. |
| Slots | Each slot has a label, a **color rule** (`W`, `U`, `B`, `R`, `G`, `multicolor`, `colorless`, `any`) and a **type rule** (`creature`, `noncreature`, `land`, `any`). The server rejects a card that doesn't fit. |
| Submissions | Only finished cards the player owns (from their gallery). At most one card per player per slot. At most `maxEntriesPerUser` cards per player in total. A card can be in one slot only. Players can withdraw while the pool is open, which removes that entry's medals and bans. |
| Medals | In each slot a player can give **gold (3 points), silver (2) and bronze (1)**, one medal per card. Only gold is expected; the others are optional. Giving a medal that is already on another card moves it. **Nobody can medal their own card.** Live medal counts are visible. |
| Bans | Each player has **2 bans** for the whole pool, and can lift a ban while the pool is open. Bans are secret until close: a player only sees their own. At close, a card with **3+ bans** is disqualified, whatever its medals. |
| Anonymity | Submitters' names and ban counts are hidden until the pool closes. Players only see a "Your card" flag on their own entries. |
| Winner | The card with the most points fills the slot, and more golds breaks a tie. Cards still level on both are all marked `tied`, and the host decides at the table. While the pool is open, "leader" ignores bans, so it doesn't leak them. Disqualified cards are listed as "Banned by the table" at close. A slot with no medals stays empty. |
| Power check | Each entry shows `power_level.assess(rules text, card)` as *Fair*, *Pushed* (≥ 1 mana over its budget) or *Over the curve* (≥ 2 over). It is advisory, to make a Pot of Green obvious to voters. |

### Default slot template (8 players, 16 slots)

- one creature slot and one noncreature slot for each of the five colors (10 slots)
- 2 multicolor slots
- 2 colorless slots
- 1 land slot
- 1 wild slot (any color, any type)

## 3. Storage (`storage.py`)

New tables:
- `pools(id, name, status open|closed, max_entries_per_user, created_at, closed_at)`, with at most one open
- `pool_slots(id, pool_id, position, label, color_rule, type_rule)`
- `pool_entries(id, pool_id, slot_id, card_id, user_id, created_at)`, unique on `(pool_id, card_id)` and `(slot_id, user_id)`
- `pool_medals(voter_id, pool_id, slot_id, entry_id, medal gold|silver|bronze, created_at)`, unique on `(voter_id, slot_id, medal)` and `(voter_id, entry_id)`
- `pool_bans(voter_id, pool_id, entry_id, created_at)`, unique on `(voter_id, entry_id)`

## 4. API (`api_routes.py`, under `/api/v1`)

| Method and path | Body | Returns |
|---|---|---|
| `GET /pools/current` | | `PoolView` or `null` (ETag) |
| `GET /pools` | | `PoolSummary[]`, newest first |
| `GET /pools/<id>` | | `PoolView` (ETag) |
| `POST /pools/entries` | `{slotId, cardId}` | `PoolView` |
| `POST /pools/entries/<id>/withdraw` | | `PoolView` |
| `POST /pools/medals` | `{entryId, medal}` | `PoolView` |
| `POST /pools/medals/clear` | `{entryId}` | `PoolView` |
| `POST /pools/bans` | `{entryId}` | `PoolView` |
| `POST /pools/bans/clear` | `{entryId}` | `PoolView` |
| `POST /admin/pools` | `{name, maxEntriesPerUser, slots: [{label, colorRule, typeRule}]}` | `PoolView` (admin PIN) |
| `POST /admin/pools/<id>/close` | | `PoolView` (admin PIN) |

The shapes are in `src/app/models/api.model.ts` (`PoolView`, `PoolSlotView`, `PoolEntryView`).

- A `PoolView` is personalized by `X-User-Id`, which sets `mine`, `myMedal`, `bannedByMe`, `myMedals`, `myEntryId`, `myEntryCount` and `myBansLeft`.
- `username` is `null` on other players' entries, and `bans` is `null` on every entry, until the pool closes.

## 5. Frontend

- **`/pool` (the "Knowledge Pool" nav item):**
  - a header with the rules and "your submissions N / cap"
  - a slot overview strip
  - one row per slot, showing its entries (card, power check, points, gold/silver/bronze and Ban buttons, or Withdraw on your own card) and a **Submit a card** tile
  - The tile opens a picker of my finished gallery cards that fit the slot. Cards already in the pool are left out.
  - After close, a **Legal cards** grid lists each slot's winner.
- **`/pool/:id`:** a read-only past pool.
- **`/admin`:** a Knowledge Pool section to open a pool (name, cap, editable slot list prefilled with the default template) and close it.
