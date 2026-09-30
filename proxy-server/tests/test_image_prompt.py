import itertools

from image_generation import build_art_prompt, estimate_tokens


def test_subject_first_and_style():
    positive, _ = build_art_prompt("a goblin shaman", {"colors": ["R"], "type": "Creature"})
    assert positive.startswith("a goblin shaman")
    assert "fiery" in positive
    assert "full figure" in positive
    assert "traditional oil painting on canvas, Magic: The Gathering card art, visible impasto brushstrokes" in positive
    assert "highly detailed" not in positive
    # Ordered: subject, art style, type context, color mood and palette.
    assert positive.index("impasto") < positive.index("full figure") < positive.index("fiery")


def test_negative_prompt():
    _, negative = build_art_prompt("a goblin shaman", {"colors": ["R"], "type": "Creature"})
    for word in ("text", "watermark", "border", "frame", "blurry", "nsfw", "nudity",
                 "photorealistic", "3d render", "digital painting", "airbrushed", "overexposed", "blown highlights",
                 "shirtless", "bare chest", "cleavage", "revealing clothing"):
        assert word in negative
    assert estimate_tokens(negative) <= 75


GLARE_WORDS = ("pure white", "pristine", "radiant", "dramatic lighting", "glowing",
               "electric", "prismatic", "highly detailed", "digital art")


def test_no_glare_words_for_any_colors_or_type():
    # These words produced blown-out whites and glossy HDR art instead of a painting.
    types = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Land", "Planeswalker", "Battle", ""]
    for n in range(6):
        for colors in itertools.combinations("WUBRG", n):
            for card_type in types:
                positive, _ = build_art_prompt("a knight", {"colors": list(colors), "type": card_type})
                for word in GLARE_WORDS:
                    assert word not in positive.lower(), (colors, card_type, word)


def test_uses_given_token_counter():
    # generate_art passes the real CLIP tokenizer, which counts more than the estimate.
    def double(text):
        return 2 * estimate_tokens(text)

    positive, _ = build_art_prompt(" ".join(["word"] * 100), {"colors": ["R"], "type": "Creature"}, double)
    assert double(positive) <= 75


def test_token_limit():
    words = [f"word{i}" for i in range(200)]
    subject = " ".join(words)
    positive, _ = build_art_prompt(subject, {"colors": ["W", "U", "B", "R", "G"], "type": "Legendary Creature"})
    assert estimate_tokens(positive) <= 75
    assert positive.startswith(" ".join(words[:5]))


def test_token_limit_subject_with_commas_and_style_words():
    # Subject parts that look like style words must not be dropped or reordered.
    subject = ", ".join(["a detailed map of the dragon lands"] * 30)
    positive, _ = build_art_prompt(subject, {"colors": ["G"], "type": "Land"})
    assert estimate_tokens(positive) <= 75
    assert positive.startswith("a detailed map of the dragon lands")


def test_no_card_data():
    positive, negative = build_art_prompt("x", None)
    assert positive.startswith("x")
    assert "fantasy scene" in positive
    assert "oil painting" in positive
    assert "watermark" in negative


def test_multicolor_and_type_variants():
    positive, _ = build_art_prompt("a relic", {"colors": [], "type": "Artifact"})
    assert "object focus" in positive
    positive, _ = build_art_prompt("a valley", {"colors": ["G"], "type": "Land"})
    assert "landscape" in positive and "verdant" in positive
    positive, _ = build_art_prompt("a pact", {"colors": ["B", "G"], "type": "Sorcery"})
    assert "ominous" in positive and "verdant" in positive


def test_human_creatures_get_a_human_context():
    # "creature portrait" drew Human cards as horned forest spirits.
    for subtype in ("Human", "Human Wizard", "Knight", "Rogue Cleric"):
        positive, _ = build_art_prompt("Ligma, a legendary human", {"colors": ["B", "G"], "type": "Legendary Creature", "subtype": subtype})
        assert "human character, fully clothed" in positive, subtype
        assert "creature portrait" not in positive, subtype
    for subtype in ("Dragon", "Beast", "Spirit", "", None):
        positive, _ = build_art_prompt("a beast", {"colors": ["G"], "type": "Creature", "subtype": subtype})
        assert "creature portrait" in positive, subtype


def test_humanoid_races_are_clothed_characters():
    # An elf drew as a horned monster and a vampire in a low-cut dress under "creature portrait".
    for subtype in ("Elf Druid", "Vampire Noble", "Dwarf Cleric", "Merfolk Wizard"):
        positive, _ = build_art_prompt("Countess Vael", {"colors": ["B"], "type": "Creature", "subtype": subtype})
        assert "fantasy character, fully clothed" in positive, subtype
        assert "creature portrait" not in positive, subtype


def test_fighters_are_armored():
    # Berserkers stayed bare-chested under "fully clothed"; naming the armor covered them.
    for subtype in ("Human Berserker", "Warrior", "Orc Barbarian", "Human Rogue Warrior"):
        positive, _ = build_art_prompt("Grom", {"colors": ["R"], "type": "Creature", "subtype": subtype})
        assert "chainmail shirt and breastplate" in positive, subtype
    for subtype in ("Human Wizard", "Human Monk", "Elf Druid", "Beast"):
        positive, _ = build_art_prompt("Grom", {"colors": ["R"], "type": "Creature", "subtype": subtype})
        assert "chainmail" not in positive, subtype
