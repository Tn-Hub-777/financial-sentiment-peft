from src.parsing import parse_label

def test_parse_strict_answer():
    assert parse_label("Sentiment: positive") == "positive"

def test_parse_rejects_ambiguous_answer():
    assert parse_label("It could be positive or neutral") is None
