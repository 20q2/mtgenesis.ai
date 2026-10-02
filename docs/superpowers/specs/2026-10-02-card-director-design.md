# Card director: one brief that steers rules text and art

**Status:** Design approved 2026-10-02.

## 1. Purpose

Cards, and especially the three versions of a commander set, come out alike in both art and text. The cause is that almost nothing the player chooses reaches either model:

- **Art.** SDXL gets roughly `"<Name>, a legendary dragon, medium scale"` plus a fixed style and palette. A set sends one prompt for all three versions, so only the seed differs. CLIP mostly ignores invented names.
- **Rules text.** The LLM gets the card's facts, a "Concept" line that repeats the auto-built art string, and one random per-color design hook. That hook is the only thing separating a set's versions, and it ignores the name and subtype.

A **director** step reads the player's choices first and writes a short brief (who this card is, what it does mechanically, what the painting shows), which then steers both models. In a commander set, it writes three deliberately different briefs.

Decisions from the design discussion:
- **No new input field.** The director works only from what the player already picks: name, type line, colors, mana value, rarity.
- **Sets:** the three versions get **three different mechanical angles** on the same character, each with matching art.
- **Reroll:** a rerolled version gets a **new angle**, distinct from the other two current versions.
- **Hidden:** players never see the brief. It only steers the models.
- **Model:** the same model as the rules text (`qwen3:8b`), called separately. A second resident model would need VRAM that a 12 GB RTX 5070 shared by Ollama and SDXL doesn't have, so a small dedicated model is left as a later A/B (`DIRECTOR_MODEL`).

## 2. The brief (`proxy-server/director.py`)

```python
def write_briefs(card: dict, count: int, avoid: list[dict] | None = None) -> list[dict] | None
```

Each brief:

```json
{
  "identity": "who or what the card is, at most 20 words",
  "mechanic": "what its abilities revolve around, at most 15 words",
  "art": {"subject": "...", "action": "...", "setting": "...", "framing": "...", "light": "..."}
}
```

The art fields are at most 12 words each.

- **Input:**
  - the card's name, type line, colors, mana cost and mana value, rarity, and subtype
  - `power.describe_budget(card)`, so the mechanic fits the card's cost
  - what the card's colors are good at (`rules_text.COLOR_HOOKS`, or `COLORLESS_HOOKS`), which the mechanic must fit; the mechanic is asked for as a short theme (about 8 words), not rules text
  - for `count > 1`, an instruction that every brief uses a different mechanic and shows the same character (one appearance), varying only action, setting, framing and light
  - with `avoid`, the briefs to differ from
- **Call:** Ollama with a JSON `format` schema, `think=False`, a small `num_ctx` (as for rules text; see CLAUDE.md on keeping the context small), and model `DIRECTOR_MODEL`.
- **Checks:**
  - every field is present and is a string; fields are trimmed to their word limits
  - the `art.subject` names the card's subtype (when it has one)
  - the mechanics within a call, and against `avoid`, differ: content-word Jaccard overlap below 0.5 (`MECHANIC_MAX_OVERLAP`)
  - On failure, retry once. If it fails again, return `None`.
- **One character per set:** after validation, every brief of a set gets the first brief's `art.subject`; a reroll gets its siblings' `art.subject`. Added 2026-10-02 after the first measurement showed a commander changing appearance between versions.
- **Art word filter:** words such as nude, naked, topless, shirtless, bare-chested and cleavage are removed from every art field.
- **Never fatal:** any exception or timeout returns `None`. A card with no brief is generated exactly as it is today.

## 3. Where it runs (`generation_queue.py`)

A **brief stage** runs before text and art.

- `enqueue()` puts a card on a brief wait list first when `DIRECTOR_ENABLED`; otherwise it is enqueued as today.
- The text (Ollama) worker serves the brief wait list **before** its rules-text list.
- The image worker takes a card only once it has a brief, or the brief stage has finished without one. Rules text also waits for that.
- **Sets:** the first version of a set to reach the stage makes one `write_briefs(card, 3)` call and stores brief *i* on the set's slot *i* card. The other two versions find their brief already stored and skip the call. If the call returns `None`, all three versions go on without a brief.
- **Reroll:** the new card calls `write_briefs(card, 1, avoid=[briefs of the set's other two current cards])`.
- **Free play:** `write_briefs(card, 1)`.
- **Restart:** like any other unfinished card, a card still waiting for its brief is marked failed by `recover_on_startup` ("Server restarted - please reroll"). A queued card whose `brief_json` is already set skips the stage.
- The queue's position and ETA treat the brief stage as part of the wait before art.

## 4. Storage

- A new nullable `brief_json TEXT` column on `cards`, added on startup by `Storage._migrate` (the same way as `shared_at`).
- `brief` is decoded with the card row. A new `Storage.set_card_brief(card_id, brief)` writes it.
- It is **not** in `CardView`.

## 5. How the brief feeds the models

**Rules text** (`rules_text.build_messages`), when the card has a brief:
- The `Design hook to consider ...` line is replaced by: `Card idea: <identity>. Build the abilities around this mechanic, scaled to the power budget: <mechanic>.`
- The `Concept:` line is omitted.
- Everything else is unchanged: the system prompt, ability count, type-specific requirements, cleanup, lint, the power budget and its retries, and `FORBIDDEN` patterns.

**Art** (`image_generation.build_art_prompt`), when the card has a brief:
- The subject becomes `subject, action, setting, framing, light`, joined with commas. It replaces the request's prompt, and the card name is not included.
- If the subject lacks the card's subtype, it is prefixed with `a <subtype>`.
- When the 75-token budget is tight, the subject is trimmed from its end, so `light` goes first and then `setting`. This follows the existing rules (`SUBJECT_MIN_TOKENS`).
- Unchanged: `ART_STYLE`, the type contexts (including the clothed/armored contexts for people), the color mood and palette, and `NEGATIVE_PROMPT`.

**Without a brief**, both prompts are byte-for-byte what they are today.

## 6. Configuration (`config.py`)

- `DIRECTOR_ENABLED = os.environ.get("MTG_DIRECTOR", "1") != "0"`: switches the director off on the night if it misbehaves.
- `DIRECTOR_MODEL = os.environ.get("MTG_DIRECTOR_MODEL", TEXT_MODEL)`.

## 7. Measuring variety

**`tools/e2e_rules_text.py --sets N [--no-director]`** generates N fixed commander requests as sets of 3 and adds to `report.md`, for each set:
- the briefs
- the three versions' final text
- **ability overlap**: the mean pairwise Jaccard similarity of content words, keywords included, between versions
- lint findings and `over_budget` as today

**`tools/e2e_art.py --sets N [--no-director]`** renders the same sets with fixed seeds and reports:
- the mean pairwise **CLIP image-embedding cosine similarity** within each set
- today's checks for blown-out whites and contrast

It needs the GPU, so run it only when the site is not live.

**Success:**
- Mean within-set text overlap is at least 25% lower than the `--no-director` baseline.
- Mean within-set art distance (1 − CLIP similarity) is at least twice the baseline's. (Revised 2026-10-02: same-style paintings rarely fall far below 0.6 CLIP similarity, so a 25% cut in raw similarity was not a reachable bar.)
- Lint errors and over-budget cards are no worse than the baseline.
- Average card time rises by at most about 4 s.

## 8. Testing (pytest, no GPU, no live Ollama)

- **`tests/test_director.py`**, with a stubbed Ollama client:
  - parsing and trimming to the word limits
  - the subtype check
  - the different-mechanics check and its single retry
  - `avoid`
  - the art word filter
  - `None` on garbage, missing fields, exceptions or a timeout
- **Queue** (`tests/test_generation_queue.py`):
  - briefs are served before rules text
  - the image worker waits for the brief stage
  - a set makes exactly one director call and each slot gets its own brief
  - a reroll passes the other two current briefs as `avoid`
  - `DIRECTOR_ENABLED = False` skips the stage
  - a `None` brief lets the card continue
  - restart recovery
- **Prompts:** `build_messages` and `build_art_prompt` with a brief (the hook is replaced, the subject is built, the subtype is added, the trim order holds), and without one (output identical to today).
- **Storage:** the `brief_json` migration on an old database, and `set_card_brief`.
