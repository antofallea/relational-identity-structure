"""Evaluate online RIS merging against cluster ground truth.

The public Leipzig Affiliations benchmark supplies a single source of noisy
affiliation strings and a complete perfect mapping.  The mapping is converted
to gold clusters.  A deterministic cluster-level holdout chooses thresholds;
the remaining clusters are evaluated under one lexical and four shuffled record
arrival orders.

The script compares three transparent methods:
  * RIS online merge: build record--token relations, merge the best compatible
    active RIS record above tau;
  * greedy Jaccard profile: attach each record to the best token-union profile
    above tau;
  * Jaccard connected components: union every earlier raw record above tau.

All methods use the same generic NFKC/case-fold/token representation.  The
evaluation reports pairwise cluster precision, recall, and F1.  It is an
end-to-end merge benchmark, but it is still not a comparison to trained ER
systems or a production workload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import sys
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.relational_identity import RelationalIdentityStructure


DATA_URL = (
    "https://git.informatik.uni-leipzig.de/api/v4/projects/dbs%2FFAMER/"
    "repository/files/benchmarkData%2Faffiliations.zip/raw?ref=master"
)
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
THRESHOLDS = tuple(round(value / 100, 2) for value in range(5, 100, 5))


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in self.parent}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def tokens(value: str) -> Set[str]:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return set(TOKEN_RE.findall(normalized))


def jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_dir(data_dir: Path) -> Path:
    """Locate the CSV directory in either archive layout used by FAMER."""
    candidates = (data_dir / "affiliations", data_dir / "affiliations" / "affiliations")
    for candidate in candidates:
        if (candidate / "Affiliations.csv").exists():
            return candidate
    return candidates[0]


def download(data_dir: Path) -> None:
    archive = data_dir / "affiliations.zip"
    if not archive.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL}")
        with urllib.request.urlopen(DATA_URL) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    if not (source_dir(data_dir) / "Affiliations.csv").exists():
        with zipfile.ZipFile(archive) as package:
            package.extractall(data_dir)


def load_data(data_dir: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, List[str]]]:
    dataset_dir = source_dir(data_dir)
    records: Dict[str, Set[str]] = {}
    with (dataset_dir / "Affiliations.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = row["id"].strip()
            if record_id in records:
                raise ValueError(f"duplicate record ID: {record_id}")
            records[record_id] = tokens(row["affiliation"])

    clusters = UnionFind(records)
    with (dataset_dir / "PerfectMapping.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["id1"] not in records or row["id2"] not in records:
                raise ValueError("perfect mapping references a missing record")
            clusters.union(row["id1"], row["id2"])
    labels = {record_id: clusters.find(record_id) for record_id in records}
    members: Dict[str, List[str]] = defaultdict(list)
    for record_id, label in labels.items():
        members[label].append(record_id)
    return records, labels, dict(members)


def split_clusters(members: Dict[str, List[str]]) -> Tuple[Set[str], Set[str]]:
    """Assign whole gold clusters to calibration or test without leakage."""
    calibration, test = set(), set()
    for label, record_ids in members.items():
        key = "|".join(sorted(record_ids)).encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 5
        (calibration if bucket == 0 else test).add(label)
    if not calibration or not test:
        raise RuntimeError("cluster split unexpectedly produced an empty partition")
    return calibration, test


def pairwise_metrics(predicted: Dict[str, str | int], gold: Dict[str, str]) -> Dict[str, float | int]:
    if set(predicted) != set(gold):
        raise ValueError("predicted and gold record sets differ")
    gold_counts = Counter(gold.values())
    gold_pairs = sum(count * (count - 1) // 2 for count in gold_counts.values())

    predicted_groups: Dict[str | int, Counter] = defaultdict(Counter)
    for record_id, predicted_label in predicted.items():
        predicted_groups[predicted_label][gold[record_id]] += 1
    predicted_pairs = 0
    true_positive = 0
    for counts in predicted_groups.values():
        group_size = sum(counts.values())
        predicted_pairs += group_size * (group_size - 1) // 2
        true_positive += sum(count * (count - 1) // 2 for count in counts.values())
    false_positive = predicted_pairs - true_positive
    false_negative = gold_pairs - true_positive
    precision = true_positive / predicted_pairs if predicted_pairs else 0.0
    recall = true_positive / gold_pairs if gold_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "predicted_pairs": predicted_pairs,
        "predicted_clusters": len(predicted_groups),
    }


def run_ris(
    records: Dict[str, Set[str]],
    gold: Dict[str, str],
    order: Sequence[str],
    threshold: float,
    embedding_dim: int = 64,
    token_weights: Dict[str, float] | None = None,
    merge_weight_strategy: str = "max",
) -> Dict[str, float | int]:
    ris = RelationalIdentityStructure(
        embedding_dim=embedding_dim,
        merge_threshold=threshold,
        verbose=False,
        auto_merge=False,
        merge_weight_strategy=merge_weight_strategy,
    )
    record_nodes: Dict[str, int] = {}
    token_nodes: Dict[str, int] = {}
    active_record_nodes: Set[int] = set()
    merge_attempts = 0
    merges = 0
    for record_id in order:
        node_id = ris.insert({"type": "record", "record_id": record_id})
        record_nodes[record_id] = node_id
        for token in sorted(records[record_id]):
            token_node = token_nodes.get(token)
            if token_node is None:
                token_node = ris.insert({"type": "token", "token": token})
                token_nodes[token] = token_node
            relation = {
                "type": f"has_token::affiliation::{token}",
                "weight": token_weights[token] if token_weights is not None else 1.0,
            }
            ris.nodes[node_id]["relations"][token_node] = relation
            ris.nodes[token_node]["relations"][node_id] = relation
        # Token nodes are never candidates in this evaluation. Computing the
        # arriving record once after its complete neighborhood is loaded is
        # equivalent to eager per-edge refresh for this score, but avoids
        # repeatedly recalculating irrelevant token-node signatures.
        ris._update_signature(node_id)
        candidate = ris.best_match(node_id, same_type=True, candidate_ids=active_record_nodes)
        if candidate is not None:
            merge_attempts += 1
            if candidate[1] > threshold and ris.merge_if_similar(
                candidate[0], node_id, update_neighbor_signatures=False
            ):
                merges += 1
                continue
        active_record_nodes.add(node_id)
    predicted = {record_id: ris._resolve_alias(node_id) for record_id, node_id in record_nodes.items()}
    result = pairwise_metrics(predicted, gold)
    result.update({"merges": merges, "merge_attempts": merge_attempts})
    return result


def run_greedy_jaccard(records: Dict[str, Set[str]], gold: Dict[str, str], order: Sequence[str], threshold: float) -> Dict[str, float | int]:
    profiles: Dict[str, Set[str]] = {}
    predicted: Dict[str, str] = {}
    merges = 0
    for record_id in order:
        best_cluster = None
        best_score = -1.0
        for cluster_id, profile in profiles.items():
            score = jaccard(records[record_id], profile)
            if score > best_score:
                best_cluster, best_score = cluster_id, score
        if best_cluster is not None and best_score > threshold:
            profiles[best_cluster].update(records[record_id])
            predicted[record_id] = best_cluster
            merges += 1
        else:
            profiles[record_id] = set(records[record_id])
            predicted[record_id] = record_id
    result = pairwise_metrics(predicted, gold)
    result.update({"merges": merges, "merge_attempts": max(0, len(order) - 1)})
    return result


def run_jaccard_components(records: Dict[str, Set[str]], gold: Dict[str, str], order: Sequence[str], threshold: float) -> Dict[str, float | int]:
    clusters = UnionFind(order)
    seen: List[str] = []
    edges = 0
    for record_id in order:
        for previous_id in seen:
            if jaccard(records[record_id], records[previous_id]) > threshold:
                clusters.union(record_id, previous_id)
                edges += 1
        seen.append(record_id)
    predicted = {record_id: clusters.find(record_id) for record_id in order}
    result = pairwise_metrics(predicted, gold)
    result.update({"merges": edges, "merge_attempts": max(0, len(order) - 1)})
    return result


METHODS: Dict[str, Callable[[Dict[str, Set[str]], Dict[str, str], Sequence[str], float], Dict[str, float | int]]] = {
    "RIS online merge": run_ris,
    "Greedy Jaccard profile": run_greedy_jaccard,
    "Jaccard connected components": run_jaccard_components,
}


def rounded(metrics: Dict[str, float | int]) -> Dict[str, float | int]:
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()}


def select_threshold(method, records, gold, order) -> Dict[str, float | int]:
    candidates = []
    for threshold in THRESHOLDS:
        candidate = method(records, gold, order, threshold)
        candidate["threshold"] = threshold
        candidates.append(candidate)
    return max(candidates, key=lambda result: (result["f1"], result["precision"], result["threshold"]))


def summary(metrics: Sequence[Dict[str, float | int]]) -> Dict[str, float]:
    return {
        key: round(statistics.mean(float(result[key]) for result in metrics), 6)
        for key in ("precision", "recall", "f1")
    } | {
        "f1_std": round(statistics.pstdev(float(result["f1"]) for result in metrics), 6),
        "f1_min": round(min(float(result["f1"]) for result in metrics), 6),
        "f1_max": round(max(float(result["f1"]) for result in metrics), 6),
    }


def markdown_report(result: Dict) -> str:
    lines = [
        "# Streaming merge benchmark: Affiliations",
        "",
        "Generated by `experiments/run_streaming_benchmark.py`; numerical values are not edited by hand.",
        "",
        "## Protocol",
        "",
        "- Dataset: Leipzig Affiliations single-source entity-clustering benchmark; gold clusters are the connected components of its perfect mapping.",
        "- Split: whole gold clusters are assigned deterministically to calibration (20% hash bucket) or test, so no entity appears in both partitions.",
        "- Tuning: tau is selected on lexical-order calibration records from 0.05–0.95 in 0.05 increments, then frozen for test.",
        "- Policies: lexical ID order plus four random arrival orders (seeds 0–3). Metrics are pairwise cluster precision, recall and F1.",
        "- RIS: each arriving record is linked to generic affiliation-token nodes; it merges only the highest-scoring active record when score > tau.",
        "",
        "## Test results",
        "",
        "| Method | tau | Lexical F1 | Random mean F1 +/- sd | Random range | Random P | Random R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in result["methods"].items():
        lexical = method["lexical"]
        random_summary = method["random_summary"]
        lines.append(
            f"| {name} | {method['validation']['threshold']:.2f} | {lexical['f1']:.3f} | "
            f"{random_summary['f1']:.3f} +/- {random_summary['f1_std']:.3f} | "
            f"{random_summary['f1_min']:.3f}–{random_summary['f1_max']:.3f} | "
            f"{random_summary['precision']:.3f} | {random_summary['recall']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The benchmark measures a specified online policy, not an optimal RIS partition. The token graph and Jaccard baselines share the same generic tokenization. A difference between lexical and shuffled runs is evidence of arrival-order sensitivity, not random experimental noise.",
        "",
        f"Input SHA-256: `{result['input_sha256']}`. Calibration: {result['calibration_records']} records / {result['calibration_clusters']} clusters; test: {result['test_records']} records / {result['test_clusters']} clusters.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "stream")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "streaming")
    parser.add_argument("--download", action="store_true", help="download the public Affiliations archive when missing")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3], help="random stream-order seeds")
    args = parser.parse_args()

    required = source_dir(args.data_dir) / "Affiliations.csv"
    if args.download:
        download(args.data_dir)
    if not required.exists():
        parser.error(f"missing {required}; re-run with --download")

    records, labels, members = load_data(args.data_dir)
    calibration_clusters, test_clusters = split_clusters(members)
    calibration_ids = sorted(record_id for record_id, label in labels.items() if label in calibration_clusters)
    test_ids = sorted(record_id for record_id, label in labels.items() if label in test_clusters)
    calibration_records = {record_id: records[record_id] for record_id in calibration_ids}
    calibration_labels = {record_id: labels[record_id] for record_id in calibration_ids}
    test_records = {record_id: records[record_id] for record_id in test_ids}
    test_labels = {record_id: labels[record_id] for record_id in test_ids}

    start = perf_counter()
    methods = {}
    for name, method in METHODS.items():
        validation = select_threshold(method, calibration_records, calibration_labels, calibration_ids)
        threshold = float(validation["threshold"])
        lexical = method(test_records, test_labels, test_ids, threshold)
        shuffled = []
        for seed in args.seeds:
            order = list(test_ids)
            random.Random(seed).shuffle(order)
            run = method(test_records, test_labels, order, threshold)
            run["seed"] = seed
            shuffled.append(run)
        methods[name] = {
            "validation": rounded(validation),
            "lexical": rounded(lexical),
            "random_orders": [rounded(run) for run in shuffled],
            "random_summary": summary(shuffled),
        }

    archive = args.data_dir / "affiliations.zip"
    result = {
        "protocol_version": 1,
        "dataset": "Leipzig Affiliations",
        "input_sha256": sha256(archive),
        "records": len(records),
        "gold_clusters": len(members),
        "calibration_records": len(calibration_ids),
        "calibration_clusters": len(calibration_clusters),
        "test_records": len(test_ids),
        "test_clusters": len(test_clusters),
        "threshold_grid": list(THRESHOLDS),
        "random_seeds": args.seeds,
        "elapsed_seconds": round(perf_counter() - start, 6),
        "methods": methods,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "streaming_results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "streaming_results.md").write_text(markdown_report(result), encoding="utf-8")
    print(markdown_report(result))
    print(f"Wrote {args.output_dir / 'streaming_results.json'}")
    print(f"Wrote {args.output_dir / 'streaming_results.md'}")


if __name__ == "__main__":
    main()
