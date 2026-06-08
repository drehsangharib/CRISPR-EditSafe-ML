import pandas as pd
from crispr_editsafe.features import featurize_dataframe, feature_columns
from crispr_editsafe.models import fit_model, predict_proba


def test_fit_predict_small_model():
    df = pd.DataFrame({
        "sgRNA": ["ACGTACGTACGTACGTACGT"] * 6,
        "target": [
            "ACGTACGTACGTACGTACGT",
            "ACGTACGTACGTACGTACGA",
            "ACGTACGTACGTACGTACCC",
            "TTTTACGTACGTACGTACCC",
            "TTTTACGTACGTACGTAAAA",
            "TTTTTTTTACGTACGTAAAA",
        ],
        "label": [1, 1, 1, 0, 0, 0],
    })
    feat = featurize_dataframe(df)
    cols = feature_columns(feat)
    bundle = fit_model(feat[cols], feat["label"])
    probs = predict_proba(bundle, feat[cols])
    assert len(probs) == len(df)
    assert probs.min() >= 0
    assert probs.max() <= 1
