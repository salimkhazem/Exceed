# EXCEED — false discovery control for anomaly localization from nominal images only

Official Implementation of the paper _False Discovery Control for Anomaly Localization from Nominal Images Only_.

A one-class anomaly detector scores every pixel. EXCEED turns exceedance counts of nominal calibration images into compound e-values,

    e(u) = (n+1) m g(S_0(u)) / (G_0 + G_1 + ... + G_n),   G_j = sum_u g(S_j(u)),

and applies e-BH. Under image exchangeability and null dominance, the FDR of flagged units
(pixels beyond an influence radius, or connected regions) is at most alpha. This holds in finite
samples and under arbitrary within-image dependence (Theorem 1).

## Quick start (one GPU, < 24 GB)

    make setup          # environment from requirements.lock
    make test           # unit, closed-form and Monte Carlo validity tests (< 1 min)
    export EXCEED_DATA=/path/to/datasets   # default: ./data
    make data scores    # public datasets and cached score maps
    make toy core       # synthetic experiments and the real-data evaluation
    make robustness     # post-registration experiments (calibration size, frontier, contamination, BTAD)

## Layout

    src/exceed/evalues.py    exceedance counts, EXCEED e-values, e-BH, BH/BY
    src/exceed/protocol.py   evaluation protocol; closed-form e-BH decisions (Prop. A.1)
    src/exceed/baselines.py  per-location / pooled conformal p-values, max-nominal, CRC, oracle
    src/exceed/detectors.py  PatchCore-, DINOv2- and PaDiM-style training-free scorers
    src/exceed/influence.py  influence-radius diagnostic on tuning images
    src/exceed/toy.py        synthetic fields and the intensity-extent law of Prop. 2
    src/exceed/robustness.py frontier and contamination experiments (reuses protocol.py)
    src/exceed/data_btad.py  BTAD records (kept separate from the main loaders)
    scripts/                 downloading, scoring, evaluation, synthetic experiments
    tools/                   aggregation, per-category views, figures
