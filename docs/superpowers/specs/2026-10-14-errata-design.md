# Host errata

The host fixes a card's rules text or power/toughness from /admin, usually to balance a commander that won the vote before the event night. The card re-renders on its stored art for everyone, and players see an "Errata" tag on it.

## Which cards

- **Commander vote:** every current version of every set locked in the open event, so a card can be fixed while people are still voting.
- **Legal commanders:** in the most recently closed event, each commander's winning version, or every tied version when there's a tie.
- **Knowledge Pool:** every entry in the open pool, or, when no pool is open, the cards that made the most recent closed pool.

The server accepts an erratum for any finished card that is a current version of a set locked in an event, or an entry in any pool (`Storage.errata_eligible`). Anything else gets 409.

## What can change

- Rules text (at most 1000 characters). `~` and `CARDNAME` become the card's name, as in generation.
- Power and toughness, for creatures and Vehicles only: a number up to 99, `*`, `X`, `1+*` or `*+1`.
- Name, mana cost, type and rarity stay as printed, so an erratum can't break the commander rules (one per 3/4/5 CMC, each rarity once).
- Flavor text is not editable: the card renderer doesn't print flavor text, so an erratum to it would change nothing anyone sees. An existing value is kept as is.

## Storage and API

- `cards.original_card_json` keeps the card as printed from its first erratum (later errata don't overwrite it). `cards.errata_at` records the latest one. Both are added by `Storage._migrate`.
- `GET /admin/errata` (host PIN) returns `{groups: [{id: vote|legal|pool, title, items: [{card: CardView, original, label, username, detail}]}]}`.
- `POST /admin/cards/<id>/errata` (host PIN) takes `{description, power?, toughness?, preview?}`.
  - With `preview: true` it returns `{image: data URL, power: PowerCheck}` and saves nothing.
  - Otherwise it re-renders the card on its stored art, records the erratum, then swaps the new PNG in for the old one.
- `POST /admin/cards/<id>/errata/revert` re-renders the printed card and clears the erratum.
- Rendering goes through `app.render_card_image`, which `create_api_blueprint` receives as `render_card`. A lock serializes it with the generation queue's renders.
- Card views carry `errata: bool`. An erratum rewrites the same file, so an errata'd card's `cardImageUrl` gets `?v=<errata_at digits>`: a new URL for the browser cache and `MediaService`.

## Frontend

- `ErrataPanelComponent` on /admin:
  - It loads once the host PIN is entered and reloads after an event is closed.
  - Tabs for the sources (it opens on Legal commanders), a filter, and a grid of cards. Errata'd cards get a gold "Errata" tag.
- `ErrataEditorComponent`, a native modal `<dialog>`:
  - The card (as it is, or the preview) beside the form: rules text, P/T for creatures and Vehicles, and the printed text for reference.
  - Preview, which also shows the fair / pushed / over power check; Save errata; Revert to printed.
- Players see the `.errata-badge` on the vote sheet's version flags, the winners banner, "Your legal commanders" and the Knowledge Pool.

## Tests

- Backend `tests/test_errata.py`: storage (the original is kept once, revert restores it), PIN required, eligibility (vote, closed event, pool; not loose or draft cards), field validation, locked fields, preview doesn't save, the image is re-rendered from the art and its URL versioned, revert, the grouped listing, and the media route ignoring `?v=`.
- Frontend: the editor (P/T only for creatures and Vehicles, Save only after a change, preview, save, revert, no PIN) and the panel (waits for the PIN, the three sources, the filter, a save updating the card in every group).
