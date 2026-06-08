import numpy as np
import pandas as pd
import pytest

from crispr_editsafe.deep_models import encode_sequence_pair, encode_dataframe_pairs


def test_encode_sequence_pair_shape_and_mismatch_channel():
    arr = encode_sequence_pair("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGA")
    assert arr.shape == (11, 20)
    assert np.isclose(arr[10].sum(), 1.0)


def test_encode_dataframe_pairs():
    df = pd.DataFrame({
        "sgRNA": ["ACGTACGTACGTACGTACGT", "AAAAAAAAAAAAAAAAAAAA"],
        "target": ["ACGTACGTACGTACGTACGA", "AAAAAAAAAAAAAAAAAAAA"],
    })
    X = encode_dataframe_pairs(df)
    assert X.shape == (2, 11, 20)


def test_build_cnn_forward_if_torch_available():
    torch = pytest.importorskip("torch")
    from crispr_editsafe.deep_models import build_cnn_model
    model = build_cnn_model()
    x = torch.zeros((2, 11, 20), dtype=torch.float32)
    y = model(x)
    assert tuple(y.shape) == (2,)
