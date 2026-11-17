# Data construction and provenance

Amazon source: https://nijianmo.github.io/amazon/index.html
Yelp source: https://www.yelp.com/dataset

Do not redistribute the original reviews with this artifact. Follow the providers' current access and reuse terms. The model-facing arrays encode domain, history, target, timestamps, inactivity gaps and catalogue membership; aggregate records here contain neither review text nor raw user identifiers.

Amazon construction is implemented in `reference_source/paper-tois-revision/post-rejection/ipm_20261111/b0_audit/audit_amazon.py` and `reference_source/experiments/ipm_global_time_20261111/build_data.py`. Yelp's chronological construction is in `reference_source/paper-tois-revision/qa/20261112_ipm_yelp_replication/build_yelp.py`; it uses the category assignment helper from `reference_source/scripts/build_yelp_shared_backbone_split.py`. The latter file's legacy standalone main is not the paper's global chronological protocol.

Read the builders for cutoff, history, same-day-tie and coverage rules. Do not derive those choices from this artifact's test metrics. Development/test artifact hashes and covered-cohort counts are in `configs/experiment.json`. Rebuilding byte-identical arrays additionally requires the original dataset release, preprocessing environment and binding files, which are not shipped in this inspection package.
