"""Run an honest, reproducible evaluation of the RIS signature readout.

The datasets are public fixed-split entity-matching tasks from CompERBench
(Primpeli & Bizer, 2020).  A record is represented as a ``record`` node and
each field-prefixed, normalized token as a ``token`` node.  An edge therefore
means “this record has this token in this field”.  No gold labels are used to
build the graph.  Gold labels from the validation split select one global
threshold per method and dataset; the test split is touched only once.

This evaluates *pairwise, pre-merge* RIS signatures.  It deliberately does
not claim to evaluate schedule-dependent online merging: the supplied public
benchmarks contain labeled candidate pairs, rather than an event stream with a
ground truth for irreversible graph merges.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.relational_identity import RelationalIdentityStructure


DATA_BASE_URL = "https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data"
DATASETS = ("abt-buy", "amazon-google", "dblp-acm")
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
THRESHOLDS = tuple(round(value / 100, 2) for value in range(0, 101))


def normalized_tokens(value: str) -> Set[str]:
    """Return generic, schema-agnostic tokens for a field value.

    NFKC normalization and case folding only make the representation stable
    across Unicode/case variants.  There are no stop-word lists, dictionaries,
    field-specific rules, external knowledge bases, or labels in this step.
    """
    text = unicodedata.normalize("NFKC", value or "").casefold()
    return set(TOKEN_RE.findall(text))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_dataset(data_dir: Path, dataset: str) -> None:
    """Download the official public files needed for one fixed-split task."""
    dataset_dir = data_dir / dataset
    records_dir = dataset_dir / "record_descriptions"
    files = {
        data_dir / f"{dataset}-records.zip": f"{DATA_BASE_URL}/{dataset}/records.zip",
        data_dir / f"{dataset}-val.csv": f"{DATA_BASE_URL}/{dataset}/gs_val.csv",
        data_dir / f"{dataset}-test.csv": f"{DATA_BASE_URL}/{dataset}/gs_test.csv",
    }
    for destination, url in files.items():
        if not destination.exists():
            download_file(url, destination)
    if not records_dir.exists():
        with zipfile.ZipFile(data_dir / f"{dataset}-records.zip") as archive:
            archive.extractall(dataset_dir)


def read_records(dataset_dir: Path) -> Tuple[Dict[str, Dict[str, Set[str]]], List[Tuple[str, str, str]]]:
    """Read source records and materialize generic field-token relations."""
    records: Dict[str, Dict[str, Set[str]]] = {}
    relations: List[Tuple[str, str, str]] = []
    for csv_path in sorted((dataset_dir / "record_descriptions").glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_record_id = (row.get("subject_id") or "").strip()
                if not raw_record_id:
                    continue
                # CompERBench's DBLP-ACM candidate files lowercase some DBLP
                # identifiers while the record description preserves case.
                # Canonicalizing keys reconciles that transport inconsistency;
                # it is not applied to the attributes used for matching.
                record_id = raw_record_id.casefold()
                field_tokens: Dict[str, Set[str]] = {}
                for field, value in row.items():
                    if field == "subject_id":
                        continue
                    tokens = normalized_tokens(value or "")
                    if tokens:
                        field_tokens[field] = tokens
                        # Sorting fixes node allocation order across Python
                        # processes.  RIS encodes neighbour IDs, so iterating
                        # a hash-randomized set here would otherwise change
                        # the measured signatures from run to run.
                        for token in sorted(tokens):
                            relations.append((record_id, field, token))
                if record_id in records:
                    raise ValueError(f"Duplicate record identifier: {record_id}")
                records[record_id] = field_tokens
    return records, relations


def read_pairs(path: Path) -> List[Tuple[str, str, bool]]:
    pairs: List[Tuple[str, str, bool]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            label = str(row["matching"]).strip().casefold() == "true"
            pairs.append((row["source_id"].strip().casefold(), row["target_id"].strip().casefold(), label))
    return pairs


def build_ris(records: Dict[str, Dict[str, Set[str]]], relations: Iterable[Tuple[str, str, str]], dimension: int) -> Tuple[RelationalIdentityStructure, Dict[str, int]]:
    """Bulk-load a bipartite record--field-token graph and compute signatures."""
    ris = RelationalIdentityStructure(
        embedding_dim=dimension,
        verbose=False,
        auto_merge=False,
    )
    record_nodes = {
        record_id: ris.insert({"type": "record", "record_id": record_id})
        for record_id in records
    }
    token_nodes: Dict[Tuple[str, str], int] = {}

    # Loading in bulk avoids evaluating transient, order-dependent merges.  It
    # writes the same undirected relations that ``connect`` would create.
    for record_id, field, token in relations:
        token_key = (field, token)
        token_node = token_nodes.get(token_key)
        if token_node is None:
            token_node = ris.insert({"type": "token", "field": field, "token": token})
            token_nodes[token_key] = token_node
        record_node = record_nodes[record_id]
        relation_type = f"has_token::{field}::{token}"
        relation = {"type": relation_type, "weight": 1.0}
        ris.nodes[record_node]["relations"][token_node] = relation
        ris.nodes[token_node]["relations"][record_node] = relation

    # Candidate scores use record signatures only; token-node signatures are
    # not needed for this pairwise readout.
    for node_id in record_nodes.values():
        ris._update_signature(node_id)
    return ris, record_nodes


def jaccard(left: Dict[str, Set[str]], right: Dict[str, Set[str]]) -> float:
    left_values = {f"{field}\u241f{token}" for field, tokens in left.items() for token in tokens}
    right_values = {f"{field}\u241f{token}" for field, tokens in right.items() for token in tokens}
    union = left_values | right_values
    return len(left_values & right_values) / len(union) if union else 0.0


def scores_for_pairs(
    pairs: Sequence[Tuple[str, str, bool]],
    records: Dict[str, Dict[str, Set[str]]],
    ris: RelationalIdentityStructure,
    record_nodes: Dict[str, int],
) -> Tuple[List[float], List[float], List[bool]]:
    ris_scores: List[float] = []
    jaccard_scores: List[float] = []
    labels: List[bool] = []
    for source_id, target_id, label in pairs:
        if source_id not in record_nodes or target_id not in record_nodes:
            raise KeyError(f"Candidate pair references a missing record: {source_id}, {target_id}")
        ris_scores.append(ris.similarity(record_nodes[source_id], record_nodes[target_id]))
        jaccard_scores.append(jaccard(records[source_id], records[target_id]))
        labels.append(label)
    return ris_scores, jaccard_scores, labels


def metrics(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> Dict[str, float | int]:
    predicted = [score > threshold for score in scores]
    tp = sum(pred and label for pred, label in zip(predicted, labels))
    fp = sum(pred and not label for pred, label in zip(predicted, labels))
    fn = sum(not pred and label for pred, label in zip(predicted, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_positive": tp + fp,
    }


def select_threshold(scores: Sequence[float], labels: Sequence[bool]) -> Dict[str, float | int]:
    # The higher threshold wins an exact F1 tie, avoiding a gratuitous false
    # positive when validation evidence is otherwise identical.
    candidates = [metrics(scores, labels, threshold) for threshold in THRESHOLDS]
    return max(candidates, key=lambda result: (result["f1"], result["precision"], result["threshold"]))


def rounded(result: Dict[str, float | int]) -> Dict[str, float | int]:
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in result.items()
    }


def markdown_report(result: Dict) -> str:
    lines = [
        "# RIS benchmark results",
        "",
        "This file is generated by `experiments/run_benchmarks.py`; do not edit its numbers by hand.",
        "",
        "## Protocol",
        "",
        "- Datasets: CompERBench fixed validation/test splits for Abt-Buy, Amazon-Google and DBLP-ACM.",
        "- Graph: a record node is linked to every field-prefixed alphanumeric token in that record.",
        "- RIS readout: 64-dimensional deterministic hash signatures, cosine score, no automatic merge.",
        "- Tuning: one threshold per method and dataset, selected by validation F1 from 101 values (0.00–1.00); test labels are not used for selection.",
        "- Baseline: generic exact-token Jaccard over the same field-prefixed tokens, also validation-tuned. It is included to distinguish the RIS hash readout from plain token overlap.",
        "- Scope: candidate-pair classification only; this does not benchmark online, irreversible merge schedules.",
        "",
        "## Test results",
        "",
        "| Dataset | Method | Validation threshold | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in result["datasets"]:
        for method_name, method in dataset["methods"].items():
            test = method["test"]
            lines.append(
                f"| {dataset['name']} | {method_name} | {method['validation']['threshold']:.2f} | "
                f"{test['precision']:.3f} | {test['recall']:.3f} | {test['f1']:.3f} | "
                f"{test['tp']} | {test['fp']} | {test['fn']} |"
            )
    lines += ["", "## Data provenance", ""]
    for dataset in result["datasets"]:
        lines.append(
            f"- **{dataset['name']}**: {dataset['records']} records, {dataset['relations']} graph edges, "
            f"{dataset['validation_pairs']} validation and {dataset['test_pairs']} test pairs. "
            f"Input SHA-256: `{dataset['input_sha256']}`."
        )
    lines += [
        "",
        "The data are downloaded at run time from the [CompERBench distribution](https://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/). "
        "See `BENCHMARKS.md` for interpretation and limitations.",
        "",
    ]
    return "\n".join(lines)


def run_dataset(data_dir: Path, dataset: str, dimension: int) -> Dict:
    dataset_dir = data_dir / dataset
    records, relations = read_records(dataset_dir)
    validation_pairs = read_pairs(data_dir / f"{dataset}-val.csv")
    test_pairs = read_pairs(data_dir / f"{dataset}-test.csv")

    start = perf_counter()
    ris, record_nodes = build_ris(records, relations, dimension)
    build_seconds = perf_counter() - start
    val_ris, val_jaccard, val_labels = scores_for_pairs(validation_pairs, records, ris, record_nodes)
    test_ris, test_jaccard, test_labels = scores_for_pairs(test_pairs, records, ris, record_nodes)

    methods = {}
    for name, validation_scores, test_scores in (
        ("RIS signature", val_ris, test_ris),
        ("Token Jaccard baseline", val_jaccard, test_jaccard),
    ):
        validation = select_threshold(validation_scores, val_labels)
        methods[name] = {
            "validation": rounded(validation),
            "test": rounded(metrics(test_scores, test_labels, float(validation["threshold"]))),
        }

    input_files = [
        data_dir / f"{dataset}-records.zip",
        data_dir / f"{dataset}-val.csv",
        data_dir / f"{dataset}-test.csv",
    ]
    return {
        "name": dataset,
        "records": len(records),
        "relations": len(relations),
        "validation_pairs": len(validation_pairs),
        "validation_positive": sum(label for _, _, label in validation_pairs),
        "test_pairs": len(test_pairs),
        "test_positive": sum(label for _, _, label in test_pairs),
        "graph_build_seconds": round(build_seconds, 6),
        "input_sha256": {path.name: sha256(path) for path in input_files},
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "comperbench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--download", action="store_true", help="download missing official dataset files")
    args = parser.parse_args()

    if args.dimension <= 0:
        parser.error("--dimension must be positive")
    for dataset in args.datasets:
        required = [
            args.data_dir / dataset / "record_descriptions",
            args.data_dir / f"{dataset}-records.zip",
            args.data_dir / f"{dataset}-val.csv",
            args.data_dir / f"{dataset}-test.csv",
        ]
        if args.download:
            download_dataset(args.data_dir, dataset)
        if not all(path.exists() for path in required):
            missing = ", ".join(str(path) for path in required if not path.exists())
            parser.error(f"missing data for {dataset}: {missing}. Re-run with --download.")

    result = {
        "protocol_version": 1,
        "signature_dimension": args.dimension,
        "thresholds": list(THRESHOLDS),
        "datasets": [run_dataset(args.data_dir, dataset, args.dimension) for dataset in args.datasets],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_results.json"
    markdown_path = args.output_dir / "benchmark_results.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    print(markdown_report(result))
    print(f"\nWrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
