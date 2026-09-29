import itertools

from image_generation import build_art_prompt, estimate_tokens


def test_subject_first_and_style():
    positive, _ = build_art_prompt("a goblin shaman", {"colors": ["R"], "type": "Creature"})
    assert positive.startswith("a goblin shaman")
    assert "fiery" in positive
    assert "full figure" in positive
    assert "oil painting, Magic: The Gathering card art, painterly brushstrokes" in positive
    assert "highly detailed" not in positive
    # Ordered: subject, art style, type context, color mood and palette.
    assert positive.index("painterly") < positive.index("full figure") < positive.index("fiery")


def test_negative_prompt():
    _, negative = build_art_prompt("a goblin shaman", {"colors": ["R"], "type": "Creature"})
    for word in ("text", "watermark", "border", "frame", "blurry", "nsfw", "nudity",
                 "photograph", "photorealistic", "3d render", "overexposed", "blown highlights"):
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
    assert "painterly" in positive
    assert "watermark" in negative


def test_multicolor_and_type_variants():
    positive, _ = build_art_prompt("a relic", {"colors": [], "type": "Artifact"})
    assert "object focus" in positive
    positive, _ = build_art_prompt("a valley", {"colors": ["G"], "type": "Land"})
    assert "landscape" in positive and "verdant" in positive
    positive, _ = build_art_prompt("a pact", {"colors": ["B", "G"], "type": "Sorcery"})
    assert "ominous" in positive and "verdant" in positive
