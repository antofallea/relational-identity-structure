# Benchmark correction and protocol

## Correction record

Before this revision, the repository and its PDF reported a synthetic customer
dataset, Cora, Amazon-Google comparisons, neural/rule-based baselines,
ablations, latency, memory, and large-scale HNSW results.  No experiment
directory, dataset copy, configuration, raw output, or runner supporting those
claims was present in the repository.  They are withdrawn rather than
retroactively reconstructed.

The results below are the first checked-in, executable evaluation for this
repository.  They are deliberately narrower than the withdrawn claims.

## Data

We use the fixed validation/test entity-matching tasks distributed by
[CompERBench](https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/),
which was created to make entity-matching tasks comparable and reproducible
(Primpeli and Bizer, CIKM 2020).  The runner downloads the official
`records.zip`, `gs_val.csv`, and `gs_test.csv` files at run time.

| Dataset | Domain | Records | Validation pairs / matches | Test pairs / matches |
|---|---|---:|---:|---:|
| Abt-Buy | products | 2,173 | 1,439 / 220 | 710 / 109 |
| Amazon-Google | products | 2,404 | 1,696 / 261 | 836 / 128 |
| DBLP-ACM | bibliographic records | 4,910 | 9,387 / 447 | 4,624 / 220 |

The expected SHA-256 values for the exact downloaded files are written to
[`results/benchmark_results.json`](results/benchmark_results.json).  This lets
later runs reject silently changed upstream files.

## Evaluated method

These public tasks supply labeled candidate pairs, rather than an online
stream of ground-truth merge events.  We therefore evaluate the **pre-merge
RIS signature** as a pairwise score, not the irreversible merge procedure.

For each source record, the adapter:

1. creates a `record` node;
2. applies Unicode NFKC normalization and case folding to every non-ID field;
3. extracts alphanumeric tokens with the language-neutral regular expression
   `\w+`;
4. creates a `token` node for each `(field, token)` pair; and
5. connects the record to that token using the relation label
   `has_token::<field>::<token>`.

The signature is the library's 64-dimensional deterministic hash signature.
Automatic merging is disabled so that a score for a candidate pair is not
affected by unrelated candidate ordering.  A pair is predicted positive when
its cosine score is strictly greater than `tau`.

The adapter sorts tokens before allocating graph nodes.  This matters because
the implementation hashes neighbor IDs: iterating an unordered Python `set`
would otherwise make the scores process-dependent.  A repeat run of
Amazon-Google after this correction produced identical metrics and input
checksums (wall-clock build time excluded).

## Selection procedure

For each method and dataset, `tau` is selected from 101 values from 0.00 to
1.00 inclusive using validation F1.  Exact ties prefer the higher threshold,
then higher precision.  The selected value is frozen before the test pairs are
scored.  Consequently, this is not a label-free evaluation: validation labels
are used for threshold calibration.

For a minimal, transparent comparator we also evaluate **Token Jaccard** on
the exact same field-prefixed tokens and with the same validation tuning.  It
is not presented as a state-of-the-art ER baseline.  Its purpose is to test
whether RIS's hash representation improves on the token-overlap information
provided to it.

## Fixed-test results

| Dataset | Method | tau from validation | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Abt-Buy | RIS signature | 0.28 | 0.353 | 0.550 | 0.430 | 60 | 110 | 49 |
| Abt-Buy | Token Jaccard | 0.14 | 0.312 | 0.450 | 0.368 | 49 | 108 | 60 |
| Amazon-Google | RIS signature | 0.33 | 0.575 | 0.508 | 0.539 | 65 | 48 | 63 |
| Amazon-Google | Token Jaccard | 0.10 | 0.471 | 0.688 | 0.559 | 88 | 99 | 40 |
| DBLP-ACM | RIS signature | 0.55 | 0.986 | 0.995 | 0.991 | 219 | 3 | 1 |
| DBLP-ACM | Token Jaccard | 0.41 | 1.000 | 0.986 | 0.993 | 217 | 0 | 3 |

The machine-readable source for this table is generated, not transcribed:
[`results/benchmark_results.json`](results/benchmark_results.json).  The
matching Markdown report is
[`results/benchmark_results.md`](results/benchmark_results.md).

## Interpretation

The results are heterogeneous:

- On DBLP-ACM, both methods are almost perfect.  RIS is **not** superior to
  simple field-token overlap (0.991 vs 0.993).
- On Amazon-Google, RIS is lower than the baseline by 0.020 F1.
- On Abt-Buy, RIS is higher by 0.062 F1, but the absolute F1 remains 0.430.

This small evaluation does not establish that structural signatures solve ER
without rules.  With this graph adapter, RIS mainly approximates token overlap
through a lossy hash projection.  The test has no evidence for the previous
claims about performance relative to Dedupe, DeepMatcher, Ditto, learned GNNs,
or custom rules: none of those systems was run in this repository.

## End-to-end streaming merges

The [Leipzig Affiliations benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)
is a single-source clustering task with 2,260 affiliation strings, 330 gold
clusters, and a complete perfect mapping.  We take connected components of the
mapping as the ground-truth entities.  A hash of every whole cluster assigns it
to calibration (70 clusters / 497 records) or test (260 clusters / 1,763
records), preventing an entity from leaking across the split.

The online policy is explicit: after its field-token relations are loaded, an
arriving record can merge only into its highest-scoring active `record` node
when the RIS score is above `tau`.  `tau` is selected on the lexical-order
calibration stream from 0.05 to 0.95 in 0.05 steps, then frozen.  We evaluate
one lexical order and four shuffled test orders.  The output metric is pairwise
cluster precision, recall and F1, calculated from the final aliases.

| Method | tau | Lexical F1 | Random F1 mean +/- sd | Random range | Random P | Random R |
|---|---:|---:|---:|---:|---:|---:|
| RIS online merge | 0.45 | 0.242 | 0.235 +/- 0.011 | 0.225–0.254 | 0.266 | 0.211 |
| Greedy Jaccard profile | 0.15 | 0.263 | 0.250 +/- 0.008 | 0.238–0.260 | 0.277 | 0.227 |
| Jaccard connected components | 0.55 | 0.068 | 0.068 +/- 0.000 | 0.068–0.068 | 0.036 | 0.615 |

RIS is lower than the greedy Jaccard baseline on this task.  Its 0.029 F1
range across four fixed shuffled orders is evidence of arrival-order
sensitivity under the specified policy.  Connected components are invariant to
arrival order but collapse many distinct records: the method has 0.615 recall
and only 0.036 precision.  These data do not establish a dynamic benefit for
RIS; they provide a reproducible counterweight to that hypothesis.

The runner and raw results are
[`experiments/run_streaming_benchmark.py`](experiments/run_streaming_benchmark.py),
[`results/streaming/streaming_results.json`](results/streaming/streaming_results.json),
and [`results/streaming/streaming_results.md`](results/streaming/streaming_results.md).

## Improved relational merge policy

The initial streaming test exposes two implementation choices worth testing:
unweighted common tokens can dominate the signature, and a merge retained only
the maximum weight when two members had the same relation.  We change both
within RIS:

1. Each `record -- token` edge receives an unsupervised IDF weight
   `log((N + 1) / (df(token) + 1)) + 1`, estimated once from all 2,260 raw,
   unlabeled records.
2. When two RIS records merge and both have the same relation to a token, their
   weights are **summed**. This preserves member multiplicity in the merged
   relational signature rather than dropping all but one contribution.

The follow-up selects three RIS hyperparameters only on the same calibration
clusters: signature dimension in {128, 256, 512, 1024}, relation aggregation
in {max, sum}, and tau from 0.30 to 0.70 in 0.05 steps.  It selects 256
dimensions, `sum`, and tau=0.35 (calibration F1=0.794), then freezes them for
the identical test records and four shuffled orders.  We add a stricter greedy
baseline: profile assignment with exact IDF-weighted cosine, using those same
unlabeled IDF values.

| Method | Lexical F1 | Random F1 mean +/- sd | Random range | Random P | Random R |
|---|---:|---:|---:|---:|---:|
| RIS weighted signature, summed relations | **0.567** | **0.576 +/- 0.014** | 0.552–0.589 | 0.567 | 0.587 |
| Greedy weighted-cosine profile | 0.468 | 0.443 +/- 0.011 | 0.429–0.458 | 0.484 | 0.408 |
| Greedy Jaccard profile | 0.263 | 0.250 +/- 0.008 | 0.238–0.260 | 0.277 | 0.227 |
| Jaccard connected components | 0.068 | 0.068 +/- 0.000 | 0.068–0.068 | 0.036 | 0.615 |

On this fixed benchmark, the revised RIS policy exceeds the exact
IDF-weighted greedy baseline by 0.133 mean F1.  That supports the narrower
claim that preserving accumulated relation evidence can improve this online
policy beyond weighted lexical profile similarity.  The randomized interval
shows the policy remains order-sensitive; it is not a claim of confluence.

The IDF calculation sees the entire **unlabeled** raw batch before the stream
starts. This is reasonable for a batch-to-stream ingestion setup, but not for a
strict future-blind online setting; the experiment must not be cited as proof
of the latter.  It is also one dataset, so no cross-domain generalization claim
is justified.

The runner and raw results are
[`experiments/run_weighted_streaming_benchmark.py`](experiments/run_weighted_streaming_benchmark.py),
[`results/streaming_weighted/weighted_streaming_results.json`](results/streaming_weighted/weighted_streaming_results.json),
and [`results/streaming_weighted/weighted_streaming_results.md`](results/streaming_weighted/weighted_streaming_results.md).

## What remains unmeasured

- Behavior on more than one streaming dataset and a strictly future-blind IDF
  update policy.
- Robustness to temporal drift and adversarial graph changes.
- Scalability, HNSW recall/latency, and memory use.
- Comparisons with trained or classical ER systems under a shared protocol.

These are concrete next experiments, not current findings.  Any future paper
claim about them should ship the runner, configuration, raw output, versioned
data checksums, and a fixed evaluation split.

## Reproduction

```bash
python experiments/run_benchmarks.py --download --output-dir results
python experiments/run_streaming_benchmark.py --download --output-dir results/streaming
python experiments/run_weighted_streaming_benchmark.py --download --output-dir results/streaming_weighted
```

The current run used Python 3.12.13 and NumPy 2.3.5.  `hnswlib` was not
installed and is not involved in either evaluation.  Re-running with the same
downloaded files and code should reproduce scores exactly; only elapsed wall
time is expected to vary.
