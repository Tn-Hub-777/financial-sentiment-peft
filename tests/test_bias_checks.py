from src.bias_checks import normalize, phrasebank_overlap

def test_normalization_makes_simple_variants_equal():
    assert normalize("Profit,  rose!") == normalize("profit rose")

def test_overlap_detects_exact_normalized_match():
    hits, report = phrasebank_overlap(["Acme profit rose."], ["ACME profit rose"], 100)
    assert hits == {0}
    assert report["overlap_removed"] == 1
