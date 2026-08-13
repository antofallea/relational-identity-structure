# Relational Identity Structure (RIS): reproducibility correction and preliminary benchmark

Antonio Fallea  
13 August 2026

## Abstract

Relational Identity Structure (RIS) is a prototype which derives node signatures
from typed graph relations and can merge highly similar nodes. Earlier versions
of this project reported benchmark, ablation, and scalability results without
the datasets, scripts, configurations, or raw outputs needed to verify them.
This paper withdraws those claims and replaces them with an executable,
narrower evaluation. On fixed CompERBench entity-matching test sets, a
64-dimensional pre-merge RIS signature obtains F1 0.430 (Abt-Buy), 0.539
(Amazon-Google), and 0.991 (DBLP-ACM); a generic token-Jaccard baseline obtains
0.368, 0.559, and 0.993. We also evaluate actual online merges on the Leipzig
Affiliations clustering benchmark: RIS reaches random-order F1 0.235 +/- 0.011,
below a greedy Jaccard profile baseline at 0.250 +/- 0.008, and varies from
0.225 to 0.254 across four arrival orders. We then revise the policy with
unsupervised IDF relation weights and summed evidence on merge. Chosen solely
on calibration clusters, this version reaches F1 0.576 +/- 0.014, exceeding
both greedy Jaccard (0.250 +/- 0.008) and exact weighted-cosine profile
assignment (0.443 +/- 0.011) on the fixed test stream. This is a one-dataset,
offline-vocabulary result, not a general performance claim. The contribution is
a falsifiable baseline: runnable code, raw metrics, checksums, and explicit
limitations.

Keywords: entity resolution, record linkage, graph representation,
reproducibility, structural equivalence

## 1. Correction and scope

RIS represents entities as graph nodes, forms a signature from neighboring
relations, and offers a thresholded, irreversible merge. The premise is related
to structural equivalence in network analysis [Lorrain and White, 1971], but it
is a design hypothesis rather than a correctness guarantee.

The prior repository documentation and PDF reported results on synthetic
customers, Cora and Amazon-Google; comparisons with rule-based, Dedupe,
DeepMatcher, Ditto and GNN systems; ablations; HNSW latency; and memory or
large-scale results. At audit time the repository had only the prototype and an
example: no experiment directory, data, configuration, raw result, or runner
supported those statements. They are withdrawn, not reconstructed.

This revision evaluates both a pairwise signature readout on labeled candidate
pairs and one specified online merge policy. It does not establish general
correctness, order-stability, utility, or scalability of automatic merging.

## 2. Prototype

For a node `v` with neighboring relations `(u, lambda, w)`, the code uses
`phi(v) = normalize(sum w * encode(u, lambda))`. `encode` is a deterministic
pseudorandom vector keyed by the relation label and neighboring node ID. Pair
score is cosine(`phi(a)`, `phi(b)`). In normal library operation, similarly
typed nodes with at least two relations can merge when their score exceeds a
threshold. Merges are irreversible and change the graph.

Automatic merging is disabled for the pairwise study. Candidate-pair test sets
do not specify a ground-truth temporal merge stream or a final gold partition,
so allowing unrelated online merges would make that evaluation ill-defined.
Section 5 separately evaluates a fully specified online policy on data with
cluster ground truth.

## 3. Protocol

### 3.1 Data

We use the public fixed validation/test entity-matching tasks distributed by
CompERBench [Primpeli and Bizer, 2020].

| Task | Domain | Records | Validation pairs / positives | Test pairs / positives |
|---|---|---:|---:|---:|
| Abt-Buy | products | 2,173 | 1,439 / 220 | 710 / 109 |
| Amazon-Google | products | 2,404 | 1,696 / 261 | 836 / 128 |
| DBLP-ACM | bibliography | 4,910 | 9,387 / 447 | 4,624 / 220 |

The runner downloads the official `records.zip`, `gs_val.csv`, and
`gs_test.csv`; SHA-256 values for the actual inputs are in
`results/benchmark_results.json`.

### 3.2 Graph and selection

The adapter creates a record node per source record. It applies Unicode NFKC
and case folding to every non-ID value, extracts generic `\w+` tokens, creates
one field-token node per distinct `(field, token)`, and links record to token
with `has_token::<field>::<token>`. There are no domain lexicons, fuzzy-string
rules, external knowledge, or labels in graph construction.

We compute 64-dimensional signatures and predict a candidate match only when
cosine score is strictly greater than `tau`. Token allocation is sorted before
IDs are assigned: because the encoder uses neighboring IDs, unordered set
iteration would otherwise make the output process-dependent.

For each method and dataset, `tau` is selected from 0.00, 0.01, ..., 1.00 by
validation F1. Ties prefer higher precision, then the higher threshold. The
value is frozen before scoring test pairs. Thus the procedure uses validation
labels for calibration; it is not a label-free thresholding claim.

### 3.3 Baseline and metrics

The only rerun comparator is exact-token Jaccard over the same field-prefixed
tokens and with the same validation procedure. It is a sanity baseline, not a
state-of-the-art claim. We report fixed-test pairwise precision, recall and F1;
we do not report systems that were not executed in this repository.

## 4. Results

| Dataset | Method | tau | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Abt-Buy | RIS signature | 0.28 | 0.353 | 0.550 | 0.430 | 60 | 110 | 49 |
| Abt-Buy | Token Jaccard | 0.14 | 0.312 | 0.450 | 0.368 | 49 | 108 | 60 |
| Amazon-Google | RIS signature | 0.33 | 0.575 | 0.508 | 0.539 | 65 | 48 | 63 |
| Amazon-Google | Token Jaccard | 0.10 | 0.471 | 0.688 | 0.559 | 88 | 99 | 40 |
| DBLP-ACM | RIS signature | 0.55 | 0.986 | 0.995 | 0.991 | 219 | 3 | 1 |
| DBLP-ACM | Token Jaccard | 0.41 | 1.000 | 0.986 | 0.993 | 217 | 0 | 3 |

RIS is marginally below Jaccard on DBLP-ACM and Amazon-Google. It is higher on
Abt-Buy, though its absolute F1 is only 0.430. The results are heterogeneous
and do not establish an entity-matching advantage. With this adapter the RIS
signature is more plausibly a lossy random projection of token overlap.

## 5. Streaming merge evaluation

The public Leipzig Affiliations benchmark contains 2,260 affiliation strings,
330 gold clusters, and a complete perfect mapping. Connected components of the
mapping define the reference clusters. A deterministic hash assigns whole gold
clusters to calibration (70 clusters; 497 records) or test (260 clusters; 1,763
records), preventing records of the same entity from crossing the split.

For each arriving record, RIS loads its affiliation-token relations, finds the
highest-scoring active record node, and merges only that candidate when its
score exceeds tau. The calibration stream selects tau from 0.05 to 0.95 in
0.05 increments; the test uses that fixed threshold. We compare the policy to
a greedy Jaccard token-profile merge and a thresholded Jaccard connected-
components algorithm. All use the same NFKC, case-folded generic tokens.

| Method | tau | Lexical F1 | Random F1 mean +/- sd | Range | Random P | Random R |
|---|---:|---:|---:|---:|---:|---:|
| RIS online merge | 0.45 | 0.242 | 0.235 +/- 0.011 | 0.225–0.254 | 0.266 | 0.211 |
| Greedy Jaccard profile | 0.15 | 0.263 | 0.250 +/- 0.008 | 0.238–0.260 | 0.277 | 0.227 |
| Jaccard connected components | 0.55 | 0.068 | 0.068 +/- 0.000 | 0.068–0.068 | 0.036 | 0.615 |

RIS is below the greedy baseline and varies by 0.029 F1 across shuffled arrival
orders. That variation is a policy property, not sampling noise: the records
and test partition do not change. The connected-components result makes a
separate point: transitive closure is order-invariant but creates many false
links. This single evaluation does not show a dynamic advantage for RIS.

### 5.1 Weighted evidence revision

The first streaming evaluation identifies two limitations of the prototype:
common relation tokens have the same influence as discriminative tokens, and a
merge retains only one edge when multiple members share a relation. We introduce
two changes that preserve the relational model. First, record-token edge weight
is IDF, `log((N+1)/(df+1))+1`, computed once from the complete unlabeled raw
corpus. Second, a merge **sums** weights of identical relations, so a cluster
signature preserves how many of its members support a relation.

Using only the calibration split, we select dimension from {128, 256, 512,
1024}, aggregation from {maximum, sum}, and tau from 0.30–0.70 in 0.05 steps.
The selected configuration is 256 dimensions, sum aggregation, and tau=0.35
(calibration F1=0.794). It is frozen before the test stream. We introduce a
stronger control: the same greedy profile policy scored by exact IDF-weighted
cosine, using the identical unlabeled IDF values.

| Method | Lexical F1 | Random F1 mean +/- sd | Range | Random P | Random R |
|---|---:|---:|---:|---:|---:|
| RIS weighted signature, summed relations | **0.567** | **0.576 +/- 0.014** | 0.552–0.589 | 0.567 | 0.587 |
| Greedy weighted-cosine profile | 0.468 | 0.443 +/- 0.011 | 0.429–0.458 | 0.484 | 0.408 |
| Greedy Jaccard profile | 0.263 | 0.250 +/- 0.008 | 0.238–0.260 | 0.277 | 0.227 |
| Jaccard connected components | 0.068 | 0.068 +/- 0.000 | 0.068–0.068 | 0.036 | 0.615 |

On this fixed test, summed relational evidence improves mean F1 by 0.133 over
the exact weighted-cosine greedy control. The gain is consistent with the
mechanism: after a correct merge, support for characteristic relations grows
and subsequent candidates are scored against that accumulated context. The
0.037 F1 random-order range shows this does not solve path dependence.

This is not a strict future-blind streaming experiment: IDF is estimated from
the entire unlabeled corpus before arrivals begin. Nor is it evidence of
cross-domain generalization. It is a reproducible, bounded result for the
specified Affiliations policy only.

## 6. Limitations

The streaming studies measure one single-source dataset and use an offline
unlabeled IDF vocabulary; they do not generalize to other domains, strict
future-blind streams, temporal drift, adversarial changes, HNSW recall or
latency, memory consumption, scalability, learned signatures, or external ER
systems. The prototype cannot undo a false merge. It should be treated as a
research artifact, not used for unattended production deduplication.

Future end-to-end work needs a specified record-arrival stream, update and
merge policy, alias policy, and merge-level or partition-level ground truth.
Claims about any such work should ship the runner, configuration, raw outputs,
fixed splits, and input checksums.

## 7. Reproducibility and conclusion

From the repository root run:

    python experiments/run_benchmarks.py --download --output-dir results
    python experiments/run_streaming_benchmark.py --download --output-dir results/streaming
    python experiments/run_weighted_streaming_benchmark.py --download --output-dir results/streaming_weighted

The runner produces raw JSON metrics, a Markdown report, selected thresholds,
input checksums, and graph sizes. It requires Python and NumPy; optional
`hnswlib` is not used in this pairwise evaluation. With the same code and data,
scores reproduce exactly; wall-clock graph-build time may vary.

RIS remains a research prototype, but the current evidence now includes a
bounded dynamic merge advantage: on one fixed, offline-vocabulary clustering
stream, weighted relation signatures with evidence-preserving merges exceed
two greedy lexical controls. It still performs inconsistently in the earlier
pairwise tasks, and the streaming policy remains order-sensitive. The proper
claim is therefore specific and testable, not universal: accumulated,
rarity-weighted relational context can improve this merge policy on the
Affiliations benchmark. Cross-domain and genuinely online validation remain
necessary.

## References

- I. P. Fellegi and A. B. Sunter. A theory for record linkage. *JASA*, 1969.
- A. Grover and J. Leskovec. node2vec. *KDD*, 2016.
- F. Lorrain and H. C. White. Structural equivalence. *J. Mathematical
  Sociology*, 1971.
- A. Primpeli and C. Bizer. Profiling entity matching benchmark tasks. *CIKM*,
  2020. DOI: 10.1145/3340531.3412781.
- Database Group Leipzig. Benchmark datasets for entity resolution: Affiliations
  clustering task. https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution
  (accessed 2026-08-13).
