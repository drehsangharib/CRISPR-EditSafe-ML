import pytest
from crispr_editsafe.features import gc_content, mismatch_vector, featurize_pair, featurize_dataframe
import pandas as pd


def test_gc_content():
    assert gc_content("GGCC") == 1.0
    assert gc_content("AATT") == 0.0


def test_mismatch_vector():
    mm = mismatch_vector("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGA")
    assert sum(mm) == 1
    assert len(mm) == 20


def test_invalid_sequence():
    with pytest.raises(ValueError):
        featurize_pair("ACGTXYZACGTACGTACGT", "ACGTACGTACGTACGTACGT")


def test_featurize_dataframe():
    df = pd.DataFrame({"sgRNA":["ACGTACGTACGTACGTACGT"], "target":["ACGTACGTACGTACGTACGA"], "label":[1]})
    out = featurize_dataframe(df)
    assert "mismatch_count" in out.columns
    assert out.loc[0, "mismatch_count"] == 1
