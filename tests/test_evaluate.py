from crispr_editsafe.evaluate import classification_metrics


def test_classification_metrics():
    m = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert m["accuracy"] == 1.0
    assert m["auprc"] is not None
    assert m["auroc"] is not None
