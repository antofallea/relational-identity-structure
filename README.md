# Relational Identity Structure (RIS)

## Reproducibility status

This repository implements a prototype for representing entity identity through
relational signatures.  On 13 August 2026, the experimental claims previously
published in this repository and in `RIS.pdf` were audited.  The repository did
not contain the stated experiment scripts, datasets, configurations, or raw
results.  Those numbers are therefore withdrawn.

The current repository contains a runnable benchmark, its raw output, and a
revised paper that reports only results produced from the checked-in code.

| Fixed test set | RIS signature F1 | Generic token-Jaccard F1 |
|---|---:|---:|
| Abt-Buy | 0.430 | 0.368 |
| Amazon-Google | 0.539 | 0.559 |
| DBLP-ACM | 0.991 | 0.993 |

These measurements are a **pairwise pre-merge evaluation**, not a validation
of the online, irreversible merge dynamics.  They do not support a claim that
RIS is state of the art, better than standard entity-matching tools, or ready
for production.  Read [BENCHMARKS.md](BENCHMARKS.md) before interpreting the
table.

## Streaming merge result

The next experiment evaluates actual, irreversible merges on the public
[Leipzig Affiliations](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)
single-source clustering task (2,260 records, 330 gold clusters). Whole gold
clusters are held out for calibration or test; the test split has 1,763 records
in 260 clusters. Each incoming record is compared only with active records,
then either becomes a new cluster or merges into its best compatible candidate.

| Method | Test F1, random-order mean +/- sd | Range across 4 orders |
|---|---:|---:|
| RIS online merge | 0.235 +/- 0.011 | 0.225–0.254 |
| Greedy Jaccard profile | 0.250 +/- 0.008 | 0.238–0.260 |
| Jaccard connected components | 0.068 +/- 0.000 | 0.068–0.068 |

That initial unweighted configuration was below the greedy baseline and
arrival-order sensitive.  It remains in
[`results/streaming/`](results/streaming/) as a baseline, not a discarded
result.

## Improved relational stream policy

The current RIS policy addresses two concrete weaknesses: rare tokens receive
larger, unsupervised IDF relation weights; and when records merge, the weight of
a shared relation is **summed**, preserving evidence from every member rather
than retaining only the maximum.  Dimension, aggregation strategy, and `tau`
were chosen solely on calibration clusters; the fixed test split was not used
for selection.

| Method | Test F1, random-order mean +/- sd | Range across 4 orders |
|---|---:|---:|
| RIS: weighted signature + summed relations | **0,576 ± 0,014** | 0,552–0,589 |
| Greedy weighted-cosine profile | 0,443 ± 0,011 | 0,429–0,458 |
| Greedy Jaccard profile | 0,250 ± 0,008 | 0,238–0,260 |

This is a real gain over both reported greedy baselines for this benchmark,
under the stated policy.  It is not proof of general superiority: it comes from
one dataset and computes IDF once from the complete **unlabeled** corpus before
the arrival stream begins.  The reproducible runner and raw results are in
[`experiments/run_weighted_streaming_benchmark.py`](experiments/run_weighted_streaming_benchmark.py)
and [`results/streaming_weighted/`](results/streaming_weighted/).

## What RIS implements

`RelationalIdentityStructure` stores a typed, undirected graph.  A node
signature is the normalized sum of deterministic hash encodings of its
relations.  The engine can retrieve nearest signatures and, in its default
mode, automatically merge sufficiently similar nodes of the same type that
each have at least two relations.

```python
from src.relational_identity import RelationalIdentityStructure

ris = RelationalIdentityStructure(merge_threshold=0.99)
email = ris.insert({"type": "email", "value": "shared@example.com"})
phone = ris.insert({"type": "phone", "value": "+39 333 1234567"})
alice = ris.insert({"type": "person", "name": "Alice Rossi"})
charlie = ris.insert({"type": "person", "name": "Charlie Brown"})

ris.connect(alice, email, "has_email")
ris.connect(alice, phone, "has_phone")
ris.connect(charlie, phone, "has_phone")
ris.connect(charlie, email, "has_email")
```

Run the example with a Python environment that has NumPy installed:

```bash
python examples/alice_charlie.py
```

`hnswlib` is optional.  When it is unavailable the implementation uses an
exact linear scan for `who_am_i`; no performance claim for HNSW is made by the
current benchmark.

## Reproducing the benchmark

The experiment uses the public fixed-split tasks published by
[CompERBench](https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/):
Abt-Buy, Amazon-Google, and DBLP-ACM.  It downloads the record descriptions,
validation split, and test split directly from that distribution.  Dataset
files are deliberately not versioned in this repository.

```bash
python experiments/run_benchmarks.py --download --output-dir results
```

The script writes:

- `results/benchmark_results.json` — machine-readable metrics, thresholds and
  SHA-256 checksums of every downloaded input;
- `results/benchmark_results.md` — generated human-readable report.

The checked-in `results/` files were created by that command using a 64
dimensional hash signature.  Thresholds are selected separately for each
method and dataset on the official validation pairs, then evaluated once on
the official test pairs.  No test labels are used to select thresholds.

The graph construction is intentionally simple and fully specified:

1. Create one `record` node for every source record.
2. Normalize each non-ID value with Unicode NFKC and case folding.
3. Add one relation to a `token` node for every field-prefixed alphanumeric
   token.
4. Compute deterministic 64-dimensional RIS signatures with automatic
   merging disabled.
5. Classify each supplied candidate pair by cosine score above the
   validation-selected threshold.

The comparison baseline is exact-token Jaccard over the same field-prefixed
tokens, with the same validation procedure.  It is not an external
state-of-the-art baseline; it is included to establish whether the RIS hash
readout adds value beyond the representation it receives.

## Reproducing online merges

```bash
python experiments/run_streaming_benchmark.py --download --output-dir results/streaming
```

This runner uses the complete perfect mapping of the Leipzig Affiliations
clustering benchmark to derive gold clusters.  It assigns entire clusters to a
deterministic calibration/test split, selects `tau` from 0.05–0.95 on the
calibration stream, then evaluates one lexical and four shuffled arrival
orders.  It reports pairwise cluster precision, recall, F1, merge count, and
the variation across orders.  The two baselines are a greedy token-profile
assignment and thresholded token-Jaccard connected components.

To reproduce the improved IDF-weighted policy and its stronger controls:

```bash
python experiments/run_weighted_streaming_benchmark.py --download --output-dir results/streaming_weighted
```

## Results and interpretation

RIS is near-perfect on DBLP-ACM, but the simple Jaccard baseline is marginally
better there (0.993 versus 0.991).  On Amazon-Google, RIS is lower than that
baseline (0.539 versus 0.559).  On Abt-Buy it is higher, but both methods are
low (0.430 versus 0.368).  The available evidence thus indicates that the
current hash signature behaves as a lossy random projection of token overlap,
not as a demonstrated improvement in entity resolution.

The pairwise public benchmarks cannot establish claims about dynamic merging.
The Affiliations experiment supplies the first such measurement in this
repository, but it is only one tokenized single-source dataset and uses a
specified greedy policy.  The improved policy additionally assumes a complete
unlabeled corpus to calculate IDF before arrivals.  Neither result establishes
temporal-drift behavior, adversarial robustness, HNSW scalability, or
performance relative to trained ER systems.

## Limitations

- A merge is irreversible; the implementation has no split or rollback
  operation.
- Results depend on a labeled validation set to choose a threshold.
- The signature encoder includes neighbor IDs and has no learned semantic
  representation; fuzzy lexical matches can be lost in the hash projection.
- The end-to-end benchmark measures pairwise cluster quality and order
  sensitivity on one dataset only; it does not measure memory use, HNSW query
  latency, temporal drift, or adversarial behavior.
- The graph loader in `experiments/run_benchmarks.py` is a generic benchmark
  adapter, not a production ingest pipeline.

## Repository layout

- `src/relational_identity.py` — RIS prototype.
- `examples/alice_charlie.py` — minimal merge demonstration.
- `experiments/run_benchmarks.py` — download and benchmark runner.
- `experiments/run_streaming_benchmark.py` — end-to-end online merge runner.
- `experiments/run_weighted_streaming_benchmark.py` — calibrated
  IDF-weighted streaming evaluation with a TF-IDF control baseline.
- `results/benchmark_results.{json,md}` — generated results from the audited
  run.
- `results/streaming/` — generated online-clustering results.
- `results/streaming_weighted/` — generated results for the improved policy.
- `BENCHMARKS.md` — detailed protocol and correction record.
- `RIS.pdf` — revised paper; its editable source is
  `paper/RIS_revised_paper.md`.

## References

1. A. Primpeli and C. Bizer. *Profiling Entity Matching Benchmark Tasks.*
   CIKM, 2020. DOI: 10.1145/3340531.3412781.
2. I. P. Fellegi and A. B. Sunter. *A Theory for Record Linkage.* JASA,
   1969.
3. A. Grover and J. Leskovec. *node2vec: Scalable Feature Learning for
   Networks.* KDD, 2016.
4. F. Lorrain and H. C. White. *Structural Equivalence of Individuals in
   Social Networks.* Journal of Mathematical Sociology, 1971.
