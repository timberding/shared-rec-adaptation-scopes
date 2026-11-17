# Shared recommender adaptation scopes

Code and experiment records for *Choosing Adaptation Scopes for Dormant-User
Recovery in Shared Recommenders*. Updating a recommender for returning users
in one domain can change the lists it serves in others. We compare which
model components to update under constraints on ranking stability and update size.

## Quickstart

From the repository root, run the following with Python 3.9 or later. The
checker uses only the standard library and the included records.

```bash
python3 -B scripts/reproduce_results.py
```

The checker recomputes validation choices, paired test differences and their
95% intervals, including how replay and distillation affect non-target lists.
It covers seven encoder and target settings with five matched seeds. The
main comparison allows 50 adaptation epochs; the 10- and 100-epoch records
support checks at shorter and longer budgets.

Each output line gives the two choices under exact non-target preservation:
`exact` without a size limit and `1MiB` with the cap. For example,
`exact=S6, 1MiB=S4a` denotes Head + Time and Additive Time. `compact-exact`
is the capped choice's test NDCG@10 minus the uncapped choice's, in percentage
points. The check ends with `PASS`.

## Read the code and records

| To trace | Start here |
| --- | --- |
| Data sources, time splits, and cohort construction | [Data guide](DATA.md) and [experiment configuration](configs/experiment.json) |
| Validation screening and selection | [Scope policy](scripts/scope_policy.py) |
| Per-seed validation and test metrics | [Paired records](records/paired_records.json), indexed by setting, seed, epoch budget, and scope |
| Target adapters and shared updates | [Model definitions](reference_source/experiments/ipm_global_time_20261111/train.py) and [adaptation runner](reference_source/experiments/ipm_horizon_100_20261115/fit_adaptation.py) |
| Scoring and validation-to-test handoff | [Scoring script](reference_source/experiments/ipm_horizon_100_20261115/score.py) |

The core scope labels are `S0` (Base), `S4a` (Additive Time), `S5` (DeltaGate),
`S6` (Head + Time), `S7` (Regularized Shared), `S8` (Shared FT), and `Replay`.
The [records](records/) also include separate regularization and precision
diagnostics.

## Rerunning training

The quickstart starts from saved metrics. To rerun training, rebuild the
model inputs from Amazon Reviews 2018 and the Yelp Open Dataset, configure
the runtime, and obtain or train the Base checkpoints. Rescoring individual
requests also requires the corresponding model checkpoints. Raw reviews,
request-level predictions, and trained weights are not included.

SASRec and GRU used Python 3.9+ and PyTorch 2.7.1+cu118. SSD4Rec BiSSD used
Python 3.10.15, PyTorch 2.1.1+cu118, `mamba-ssm==2.2.2`,
`causal-conv1d==1.4.0`, `recbole==1.2.0`, and `transformers==4.46.3`.
See [third-party dependencies](THIRD_PARTY.md) for the upstream code.

The [MIT License](LICENSE) covers our study code and documentation. Datasets
and third-party components retain their respective terms.
