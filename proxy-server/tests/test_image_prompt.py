from image_generation import build_art_prompt, estimate_tokens


def test_subject_first_and_style():
    positive, _ = build_art_prompt("a goblin shaman", {"colors": ["R"], "type": "Creature"})
    assert positive.startswith("a goblin shaman")
    assert "fiery" in positive
    assert "character focus" in positive
    assert "painterly Magic: The Gathering fantasy illustration, dramatic lighting, highly detailed" in positive
    # Ordered: subject, type context, color mood and palette, style suffix.
    assert positive.index("character focus") < positive.index("fiery") < positive.index("painterly")


def test_negative_prompt():
    _, negative = build_art_prompt("a goblin shaman", {"colors": ["R"], "type": "Creature"})
    for word in ("text", "watermark", "border", "frame", "blurry", "nsfw", "nudity"):
        assert word in negative


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
    assert "magical fantasy scene" in positive
    assert "painterly" in positive
    assert "watermark" in negative


def test_multicolor_and_type_variants():
    positive, _ = build_art_prompt("a relic", {"colors": [], "type": "Artifact"})
    assert "object focus" in positive
    positive, _ = build_art_prompt("a valley", {"colors": ["G"], "type": "Land"})
    assert "landscape" in positive and "verdant" in positive
    positive, _ = build_art_prompt("a pact", {"colors": ["B", "G"], "type": "Sorcery"})
    assert "shadow" in positive and "verdant" in positive
