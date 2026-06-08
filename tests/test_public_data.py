import pandas as pd
from crispr_editsafe.public_data import standardize_public_dataset, clean_sequence, infer_mapping


def test_clean_sequence_removes_pam_and_gaps():
    assert clean_sequence("GAGT_CCGAGCAGAAGAAGAATGG") == "GAGTCCGAGCAGAAGAAGAA"


def test_standardize_public_dataset_from_score():
    df = pd.DataFrame({
        "on_seq": ["GAGTCCGAGCAGAAGAAGAATGG", "GAGTCCGAGCAGAAGAAGAATGG"],
        "off_seq": ["GAGTTAGAGCAGAAGAAGAAAGG", "TTTTCCGAGCAGAAGAAGAAAGG"],
        "CRISPR_Net_score": [0.9, 0.0],
    })
    out = standardize_public_dataset(df, source_name="unit", score_col="CRISPR_Net_score", score_threshold=0.1)
    assert list(out.columns) == ["sample_id", "source", "sgRNA", "target", "label"]
    assert out["label"].tolist() == [1, 0]
    assert out["sgRNA"].str.len().eq(20).all()


def test_infer_mapping_with_common_names():
    df = pd.DataFrame({"guide_sequence": ["A" * 20], "off_target_sequence": ["C" * 20], "label": [1]})
    mapping = infer_mapping(df)
    assert mapping.guide_col == "guide_sequence"
    assert mapping.target_col == "off_target_sequence"
    assert mapping.label_col == "label"
