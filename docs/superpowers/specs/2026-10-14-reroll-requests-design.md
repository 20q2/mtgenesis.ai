# Reroll requests on the vote page

Date: 2026-10-14

## Goal

During the commander vote, players sometimes find a version that is broken: text runs off the
card, the art is wrong, or the rules don't work. Today the owner's only fix is to unlock the
commander, which throws away every vote on all three versions. Players need a way to tell the
owner "this version is broken, reroll it". Once enough of them agree, the owner can reroll that
one version from the vote page, and the votes on the other two versions stay.

## Rules

- **Per version.** Each version of a locked commander has its own reroll request button on `/vote`.
- **Who can ask.** Any logged-in player except the owner. They can ask on a finished, current version
  of a commander locked in the open event, while voting is open. One request per player per version;
  pressing the button again takes the request back. No reason text.
- **Threshold.** 3 requests (`REROLL_REQUESTS_NEEDED` in `storage.py`) let the owner reroll that version.
  The owner sees the count before that ("2 of 3 asked for a reroll").
- **The reroll.** It uses the existing `POST /cards/<id>/reroll`. `reroll_card` now also accepts a version
  of a locked commander when the event is open and the version has 3+ requests or **failed**. A failed
  version (e.g. a reroll that failed, or "Server restarted") can always be rerolled again, so the owner is
  never stuck. Rerolling a locked commander's version:
  - deletes the votes on the old version (those voters see it as not voted again)
  - deletes the old version's requests
  - queues the new version with the same prompt and params (the director avoids its siblings' mechanics, as for a draft reroll)
  - keeps the commander locked, and keeps every vote on the other versions
- **Unfinished versions.** A version that is still generating (or failed) can't be voted on or asked
  about. `cast_vote` refuses it with 409 "That version isn't ready yet". Before this feature, a locked
  set never had an unfinished version.
- Counts are visible to everyone; who asked is not shown.

## Storage

New table:

```sql
CREATE TABLE IF NOT EXISTS reroll_requests (
    card_id     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (card_id, user_id)
);
```

- `set_reroll_request(card_id, user_id, requested)` toggles a request and returns the card's set id. Errors:
  - 404 card not found
  - 400 not a current commander version
  - 403 the owner's own card
  - 409 set not locked, voting closed, or version not finished
- `reroll_request_counts(set_id) -> {card_id: n}` and `my_reroll_requests(user_id, set_id) -> set[card_id]`.

## API

- `POST /cards/<id>/reroll-request` takes `{"requested": bool}` and returns the updated `SetView`.
- `SetView` gains `rerollRequestsNeeded`. Each card in it gains:
  - `rerollRequests` (count)
  - `myRerollRequest` (bool)
  - `canReroll` (true only for the owner, when a reroll would be accepted right now)

## Frontend

- **Player sheet, each version, while voting is open:**
  - Non-owners get a slim "Broken? Ask for a reroll" toggle (pressed: "You asked for a reroll") and the count when it's above 0.
  - The owner sees the count. When `canReroll` is true, a Reroll button appears; it confirms first ("votes on it are cleared").
  - An unfinished version shows "Rerolling version N…" (or "Version N failed") and no vote button.
- **Vote page header:** when one of my versions can be rerolled, a notice says so and opens my commanders at that commander.
- The page already polls every 10s, so the new version shows up when it's done.

## Tests

- `tests/test_reroll_requests.py`:
  - storage rules: toggle, owner refused, threshold, failed version, votes cleared, unfinished version refused
  - the API route and the new view fields
- Specs for the player sheet (buttons by role, unfinished version) and the vote page (request, reroll, owner notice).
