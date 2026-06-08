# PyTorch CNN Extension

Version 5 adds an optional PyTorch 1D CNN model for CRISPR guide/off-target sequence-pair learning.

## Install PyTorch

CPU-only installation is sufficient for this project:

```powershell
pip install torch
```

## Train CNN on the real Tsai GUIDE-seq-style benchmark

```powershell
python scripts/train_cnn_model.py --input data/processed/tsai_guideseq_2015_standardized.csv --output-dir results/tsai_guideseq_2015_cnn --epochs 20 --batch-size 64 --learning-rate 0.001 --patience 5
```

## Model input encoding

Each sgRNA/off-target pair is encoded as an `11 x 20` matrix:

- 5 channels for guide bases: A, C, G, T, N
- 5 channels for target bases: A, C, G, T, N
- 1 mismatch-indicator channel

## Notes

The CNN is intentionally lightweight and intended as a transparent baseline, not a state-of-the-art CRISPR off-target predictor. It complements the existing logistic regression and random forest baselines.
