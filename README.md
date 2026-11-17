# Shared recommender adaptation scopes

This repository contains the result records and study code for *Choosing Adaptation Scopes for Dormant-User Recovery in Shared Recommenders*.

## Quick check

Python 3.9 or later is sufficient for the result checker. No dataset download or accelerator is needed.

    python3 -B scripts/reproduce_results.py

The checker recomputes validation choices, target NDCG@10 differences, paired seed-level 95% t intervals, and non-target Replay/distillation contrasts. It covers seven backbone/target settings, five matched seeds, and 10-, 50-, and 100-epoch records. It also checks the separate L2 and precision diagnostics, packaged source hashes, and Python syntax. A successful run ends with PASS.

The main comparison uses 50 epochs; 100 epochs are a horizon check, and the 10-epoch records are a separate diagnostic. The statistics use paired seed-level differences. Intervals containing zero are not equivalence tests.

This is result-level reproduction. The quick check does not train models or rescore request-level predictions.

## Files

- records/ contains per-seed validation and test summaries and diagnostic results. Request-level prediction arrays are not included.
- configs/experiment.json records seeds, learning rates, horizons, cohorts, scopes, and screening thresholds.
- scripts/ contains the result checker and validation policy function.
- reference_source/ is an inspection snapshot of the study's model, training, scoring, data-construction, replay, and theory-check code. Relative directory layout is retained.
- source_manifest.json and file_hashes.json record source provenance and package integrity.
- DATA.md describes data sources and construction entry points.
- THIRD_PARTY.md records the external components that are not redistributed.

## Full training

Obtain Amazon Reviews 2018 and the Yelp Open Dataset from their providers and follow their access and reuse terms. The raw reviews, identifiers, trained weights, environment bindings, and request-level arrays are not included.

The SASRec and GRU runs used Python 3.9+ with PyTorch 2.7.1+cu118. SSD4Rec BiSSD used a separate Python 3.10.15 environment with PyTorch 2.1.1+cu118, mamba-ssm 2.2.2, causal-conv1d 1.4.0, RecBole 1.2.0, and Transformers 4.46.3. These are separate environments. NumPy and SciPy are needed for the optional mathematical checks. The SSD4Rec CPU fallback is a shape fixture, not the BiSSD encoder.

Training from this snapshot is not a one-command reproduction. It requires the original preprocessing outputs, base checkpoints or base training, environment bindings, and the separately obtained SSD4Rec dependencies. No raw data or checkpoints are redistributed.

## License

The MIT License in this repository covers the original study-specific code and documentation included here. Dataset terms and external component licenses remain separate.
