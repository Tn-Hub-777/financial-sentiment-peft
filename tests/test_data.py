import pytest
from src.data import map_label

@pytest.mark.parametrize("raw,expected", [("strongly positive", "positive"), ("mildly negative", "negative"), ("neutral", "neutral")])
def test_graded_labels_map_to_three_classes(raw, expected):
    assert map_label(raw) == expected

def test_unknown_label_rejected():
    with pytest.raises(ValueError): map_label("mixed")
