# CRISPR-EditSafe-ML Benchmark Report

## Overview
This report summarizes baseline model performance for a standardized CRISPR off-target dataset.

## Benchmark summary

|   n |   positive_rate |   threshold |   accuracy |   precision |   recall |   f1 |   auprc |   auroc | confusion_matrix   | model_name          | split_strategy   |   n_features |   n_train |   n_test |
|----:|----------------:|------------:|-----------:|------------:|---------:|-----:|--------:|--------:|:-------------------|:--------------------|:-----------------|-------------:|----------:|---------:|
|   3 |        0.666667 |         0.5 |   0.666667 |    0.666667 |        1 |  0.8 |       1 |       1 | [[0, 1], [0, 2]]   | logistic_regression | row_random       |          188 |         7 |        3 |
|   3 |        0.666667 |         0.5 |   0.666667 |    0.666667 |        1 |  0.8 |       1 |       1 | [[0, 1], [0, 2]]   | random_forest       | row_random       |          188 |         7 |        3 |

## Best model by AUPRC

- Model: `logistic_regression`
- AUPRC: `1.0`
- AUROC: `1.0`
- F1: `0.8`

## Metric comparison

![Metric comparison](figures/benchmark_metrics.png)

## Risk-score distributions

![Risk distributions](figures/risk_score_distributions.png)

## Interpretation notes

- AUPRC is emphasized because CRISPR off-target data are often class-imbalanced.
- sgRNA-disjoint evaluation is preferred when enough unique guides are available.
- Small demo datasets can produce unstable or perfect-looking scores; use public benchmark datasets for meaningful interpretation.

## Reproducibility

Benchmark directory: `results\public_benchmark`
