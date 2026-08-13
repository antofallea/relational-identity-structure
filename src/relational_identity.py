import numpy as np
import json
import hashlib
from typing import Dict, Iterable, List, Tuple, Optional
import os

try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    print("[WARNING] hnswlib not found. Semantic search will use O(n) fallback.")
    print("          Install with: pip install hnswlib")


class RelationalIdentityStructure:
    """
    Relational Identity Structure (RIS) Engine v3.0
    - Merges ONLY nodes with the SAME type (mandatory)
    - Threshold 0.99 (near‑perfect identity only)
    - Strict checks: both nodes must have at least 2 relations
    """

    def __init__(self, embedding_dim: int = 64, merge_threshold: float = 0.99,
                 max_elements: int = 100000, verbose: bool = True,
                 auto_merge: bool = True, merge_weight_strategy: str = "max"):
        """Create an RIS graph.

        Set ``auto_merge=False`` when the graph is used only to compute
        relational signatures (for example, to evaluate a labeled set of
        candidate record pairs).  The default preserves the original online
        merge behaviour.
        """
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.verbose = verbose
        self.auto_merge = auto_merge
        if merge_weight_strategy not in {"max", "sum"}:
            raise ValueError("merge_weight_strategy must be 'max' or 'sum'")
        self.merge_weight_strategy = merge_weight_strategy

        self.nodes: Dict[int, dict] = {}
        self.aliases: Dict[int, int] = {}
        self._next_id = 0

        self.use_hnsw = HNSW_AVAILABLE
        if self.use_hnsw:
            self.index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
            self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
            self.index.set_ef(50)
            self._active_in_index = set()

    def _resolve_alias(self, node_id: int) -> int:
        original = node_id
        visited = set()
        while node_id in self.aliases:
            if node_id in visited:
                break
            visited.add(node_id)
            node_id = self.aliases[node_id]
            if node_id == original:
                break
        return node_id

    def _hash_relation(self, neighbor_id: int, rel_type: str, weight: float) -> np.ndarray:
        # Base vector from relation type
        hash_bytes = hashlib.sha256(rel_type.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')
        rng = np.random.RandomState(seed)
        base_vector = rng.randn(self.embedding_dim)

        # Noise to differentiate same relation to different neighbours
        noise_seed = int.from_bytes(
            hashlib.sha256(f"edge_{neighbor_id}_{rel_type}".encode()).digest()[:4], 'big'
        )
        noise_rng = np.random.RandomState(noise_seed)
        noise_vector = noise_rng.randn(self.embedding_dim)

        # An edge weight must scale the complete relation encoding.  Scaling
        # only the relation-type component would leave neighbour-specific
        # evidence unweighted and make weighted signatures internally
        # inconsistent.
        return weight * (base_vector + (noise_vector * 0.7))

    def _compute_signature(self, node_id: int) -> np.ndarray:
        if node_id not in self.nodes:
            return np.zeros(self.embedding_dim)

        signature = np.zeros(self.embedding_dim)
        relations = self.nodes[node_id]['relations']

        if not relations:
            return signature

        for neighbor_id, rel_info in relations.items():
            rel_vector = self._hash_relation(neighbor_id, rel_info['type'], rel_info['weight'])
            signature += rel_vector

        signature /= np.sqrt(len(relations)) if relations else 1.0
        norm = np.linalg.norm(signature)
        return signature / norm if norm > 0 else signature

    def _update_index(self, node_id: int, signature: np.ndarray):
        if not self.use_hnsw:
            return

        if node_id in self._active_in_index:
            try:
                self.index.mark_deleted(node_id)
            except RuntimeError:
                pass

        self.index.add_items(signature.reshape(1, -1), np.array([node_id]))
        self._active_in_index.add(node_id)

    def insert(self, data: dict = None) -> int:
        node_id = self._next_id
        self._next_id += 1

        self.nodes[node_id] = {
            'data': data or {},
            'relations': {},
            'signature': np.zeros(self.embedding_dim)
        }

        zero_vec = np.zeros(self.embedding_dim)
        self._update_index(node_id, zero_vec)
        return node_id

    def connect(self, source: int, target: int, rel_type: str, weight: float = 1.0):
        source = self._resolve_alias(source)
        target = self._resolve_alias(target)

        if source not in self.nodes or target not in self.nodes or source == target:
            return

        self.nodes[source]['relations'][target] = {'type': rel_type, 'weight': weight}
        self.nodes[target]['relations'][source] = {'type': rel_type, 'weight': weight}

        self._update_signature(source)
        self._update_signature(target)

        if self.auto_merge:
            self._check_merge(source, target)
            self._check_merge_for_node(source)
            self._check_merge_for_node(target)

    def _update_signature(self, node_id: int):
        if node_id in self.nodes:
            new_sig = self._compute_signature(node_id)
            self.nodes[node_id]['signature'] = new_sig
            self._update_index(node_id, new_sig)

    def _can_merge(self, node_a: int, node_b: int) -> bool:
        """Check if two nodes can be merged (same type and both have ≥2 relations)."""
        if node_a not in self.nodes or node_b not in self.nodes:
            return False

        # MANDATORY TYPE CHECK
        type_a = self.nodes[node_a]['data'].get('type')
        type_b = self.nodes[node_b]['data'].get('type')

        # If either lacks a type, do NOT merge (safety)
        if type_a is None or type_b is None:
            if self.verbose:
                print(f"  [DEBUG] Merge blocked: missing type (A:{type_a}, B:{type_b})")
            return False

        # If types differ, do NOT merge
        if type_a != type_b:
            if self.verbose:
                print(f"  [DEBUG] Merge blocked: different types ({type_a} vs {type_b})")
            return False

        # Both must have at least 2 relations
        if len(self.nodes[node_a]['relations']) < 2:
            if self.verbose:
                print(f"  [DEBUG] {node_a} has only {len(self.nodes[node_a]['relations'])} relations (<2)")
            return False
        if len(self.nodes[node_b]['relations']) < 2:
            if self.verbose:
                print(f"  [DEBUG] {node_b} has only {len(self.nodes[node_b]['relations'])} relations (<2)")
            return False

        return True

    def _check_merge(self, node_a: int, node_b: int):
        if not self._can_merge(node_a, node_b):
            return

        sig_a = self.nodes[node_a]['signature']
        sig_b = self.nodes[node_b]['signature']
        norm_a = np.linalg.norm(sig_a)
        norm_b = np.linalg.norm(sig_b)
        if norm_a == 0 or norm_b == 0:
            return

        similarity = np.dot(sig_a, sig_b) / (norm_a * norm_b)
        if self.verbose:
            print(f"  [DEBUG] Similarity between {node_a} and {node_b}: {similarity:.4f} (threshold {self.merge_threshold})")

        if similarity > self.merge_threshold:
            self._merge_nodes(node_a, node_b)

    def _check_merge_for_node(self, node_id: int, top_k: int = 3):
        if node_id not in self.nodes:
            return

        similar = self.who_am_i(node_id, top_k=top_k)
        for other_id, sim in similar:
            if other_id == node_id:
                continue
            if other_id not in self.nodes:
                continue

            if not self._can_merge(node_id, other_id):
                continue

            if sim > self.merge_threshold:
                if self.verbose:
                    print(f"  [DEBUG] Found candidate {other_id} with sim {sim:.4f}")

                # Choose the node with more relations as the survivor
                if len(self.nodes[node_id]['relations']) >= len(self.nodes[other_id]['relations']):
                    self._merge_nodes(node_id, other_id)
                else:
                    self._merge_nodes(other_id, node_id)

                self._check_merge_for_node(node_id)
                break

    def _merge_nodes(self, node_a: int, node_b: int, update_neighbor_signatures: bool = True):
        if self.verbose:
            print(f"  [⚡ MERGE] Nodes {node_a} and {node_b} merged (same type: {self.nodes[node_a]['data'].get('type')})")

        relations_to_transfer = [
            (nbr, info) for nbr, info in self.nodes[node_b]['relations'].items() if nbr != node_a
        ]
        updated_neighbors = set()

        for neighbor_id, rel_info in relations_to_transfer:
            existing = self.nodes[node_a]['relations'].get(neighbor_id)
            if existing is not None:
                if (self.merge_weight_strategy == "sum" and
                        existing['type'] == rel_info['type']):
                    merged_relation = {
                        'type': existing['type'],
                        'weight': existing['weight'] + rel_info['weight']
                    }
                elif rel_info['weight'] > existing['weight']:
                    merged_relation = rel_info
                else:
                    merged_relation = existing
            else:
                merged_relation = rel_info

            self.nodes[node_a]['relations'][neighbor_id] = merged_relation

            if node_b in self.nodes[neighbor_id]['relations']:
                del self.nodes[neighbor_id]['relations'][node_b]
            self.nodes[neighbor_id]['relations'][node_a] = merged_relation
            updated_neighbors.add(neighbor_id)

        self.nodes[node_a]['relations'].pop(node_b, None)

        del self.nodes[node_b]
        self.aliases[node_b] = node_a

        self._update_signature(node_a)
        if update_neighbor_signatures:
            for nbr in updated_neighbors:
                self._update_signature(nbr)

        if self.use_hnsw and node_b in self._active_in_index:
            try:
                self.index.mark_deleted(node_b)
                self._active_in_index.remove(node_b)
            except RuntimeError:
                pass

    def disconnect(self, source: int, target: int):
        source = self._resolve_alias(source)
        target = self._resolve_alias(target)

        if source not in self.nodes or target not in self.nodes:
            return

        self.nodes[source]['relations'].pop(target, None)
        self.nodes[target]['relations'].pop(source, None)

        self._update_signature(source)
        self._update_signature(target)

    def who_am_i(self, node_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        node_id = self._resolve_alias(node_id)
        if node_id not in self.nodes:
            return []

        target_sig = self.nodes[node_id]['signature']
        if np.linalg.norm(target_sig) == 0:
            return []

        if self.use_hnsw and node_id in self._active_in_index:
            labels, distances = self.index.knn_query(target_sig.reshape(1, -1), k=top_k + 1)
            results = []
            for label, dist in zip(labels[0], distances[0]):
                label = int(label)
                if label == node_id or label not in self.nodes:
                    continue
                similarity = 1.0 - dist
                results.append((label, similarity))
            return results[:top_k]
        else:
            similarities = []
            for other_id, other_node in self.nodes.items():
                if other_id == node_id:
                    continue
                other_sig = other_node['signature']
                other_norm = np.linalg.norm(other_sig)
                if other_norm == 0:
                    continue
                sim = np.dot(target_sig, other_sig) / (np.linalg.norm(target_sig) * other_norm)
                similarities.append((other_id, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

    def similarity(self, node_a: int, node_b: int) -> float:
        """Return the cosine similarity between two active RIS nodes.

        This read-only helper is useful when a caller owns the decision to
        merge, or when a fixed set of candidate pairs must be evaluated.
        Empty signatures have no relational evidence and therefore score 0.
        """
        node_a = self._resolve_alias(node_a)
        node_b = self._resolve_alias(node_b)
        if node_a not in self.nodes or node_b not in self.nodes:
            return 0.0

        sig_a = self.nodes[node_a]['signature']
        sig_b = self.nodes[node_b]['signature']
        norm_a = np.linalg.norm(sig_a)
        norm_b = np.linalg.norm(sig_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(sig_a, sig_b) / (norm_a * norm_b))

    def best_match(
        self,
        node_id: int,
        same_type: bool = True,
        candidate_ids: Optional[Iterable[int]] = None,
    ) -> Optional[Tuple[int, float]]:
        """Return the most similar active node, optionally restricted by type.

        The lookup is an exact scan.  It intentionally does not use the HNSW
        index because filtering a short approximate-neighbour list can omit the
        best compatible node.  This is useful for callers that make an
        explicit, auditable online merge decision.
        """
        node_id = self._resolve_alias(node_id)
        if node_id not in self.nodes:
            return None

        target_signature = self.nodes[node_id]['signature']
        if np.linalg.norm(target_signature) == 0:
            return None
        target_type = self.nodes[node_id]['data'].get('type')

        compatible_ids = []
        candidate_vectors = []
        candidate_iterable = self.nodes.keys() if candidate_ids is None else candidate_ids
        for other_id in candidate_iterable:
            other_node = self.nodes.get(other_id)
            if other_node is None:
                continue
            if other_id == node_id:
                continue
            if same_type and other_node['data'].get('type') != target_type:
                continue
            other_signature = other_node['signature']
            if not np.any(other_signature):
                continue
            compatible_ids.append(other_id)
            candidate_vectors.append(other_signature)

        if not compatible_ids:
            return None
        scores = np.vstack(candidate_vectors).dot(target_signature)
        best_index = int(np.argmax(scores))
        return compatible_ids[best_index], float(scores[best_index])

    def merge_if_similar(
        self,
        primary: int,
        secondary: int,
        update_neighbor_signatures: bool = True,
    ) -> bool:
        """Merge two compatible nodes when their score exceeds the threshold.

        Unlike the automatic path in :meth:`connect`, the caller controls when
        candidates are examined and which node survives.  The same type,
        relation-count, threshold, and alias safeguards remain in force. Set
        ``update_neighbor_signatures=False`` only when those signatures will
        not be queried before a later refresh.
        """
        primary = self._resolve_alias(primary)
        secondary = self._resolve_alias(secondary)
        if primary == secondary or not self._can_merge(primary, secondary):
            return False
        if self.similarity(primary, secondary) > self.merge_threshold:
            self._merge_nodes(primary, secondary, update_neighbor_signatures)
            return True
        return False

    def get_node_data(self, node_id: int) -> Optional[dict]:
        node_id = self._resolve_alias(node_id)
        if node_id in self.nodes:
            return self.nodes[node_id]['data']
        return None

    def get_active_nodes(self) -> List[int]:
        return list(self.nodes.keys())

    def save(self, filepath: str):
        serializable_nodes = {}
        for k, v in self.nodes.items():
            serializable_nodes[str(k)] = {
                "data": v["data"],
                "relations": {str(nk): ni for nk, ni in v["relations"].items()},
                "signature": v["signature"].tolist()
            }

        data = {
            "embedding_dim": self.embedding_dim,
            "merge_threshold": self.merge_threshold,
            "auto_merge": self.auto_merge,
            "merge_weight_strategy": self.merge_weight_strategy,
            "next_id": self._next_id,
            "aliases": {str(k): v for k, v in self.aliases.items()},
            "nodes": serializable_nodes
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"  [💾 SAVE] Structure saved to: {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.embedding_dim = data["embedding_dim"]
        self.merge_threshold = data["merge_threshold"]
        self.auto_merge = data.get("auto_merge", self.auto_merge)
        self.merge_weight_strategy = data.get("merge_weight_strategy", self.merge_weight_strategy)
        self._next_id = data["next_id"]
        self.aliases = {int(k): v for k, v in data["aliases"].items()}

        self.nodes = {}
        if self.use_hnsw:
            self.index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
            self.index.init_index(max_elements=100000, ef_construction=200, M=16)
            self.index.set_ef(50)
            self._active_in_index = set()

        for k_str, v in data["nodes"].items():
            k = int(k_str)
            relations = {int(nk): ni for nk, ni in v["relations"].items()}
            signature = np.array(v["signature"])

            self.nodes[k] = {
                "data": v["data"],
                "relations": relations,
                "signature": signature
            }
            if self.use_hnsw:
                self.index.add_items(signature.reshape(1, -1), np.array([k]))
                self._active_in_index.add(k)

        if self.verbose:
            print(f"  [📂 LOAD] Structure loaded from: {filepath}")
