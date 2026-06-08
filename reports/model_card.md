# Model Card: CRISPR-EditSafe-ML Baseline

## Intended use
Research and portfolio demonstration for CRISPR off-target risk prediction.

## Not intended use
Clinical, therapeutic, diagnostic, or regulatory decision-making.

## Model
Default: class-balanced logistic regression using engineered sequence features.

## Data
The initial repository includes synthetic toy examples only. Public benchmark datasets should be added before any scientific interpretation.

## Metrics
Recommended metrics: AUPRC, AUROC, F1, precision, recall, calibration, and sgRNA-disjoint validation.

## Limitations
Synthetic example data do not represent real genome-editing biology.
