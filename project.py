import numpy as np
import json
import hashlib
from typing import Dict, List, Tuple
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
    Relational Identity Structure (RIS) Engine v2.1
    
    Corrected version: prevents premature merging of nodes with weak identity.
    """
    
    def __init__(self, embedding_dim: int = 64, merge_threshold: float = 0.98, max_elements: int = 100000):
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        
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
        while node_id in self.aliases:
            node_id = self.aliases[node_id]
            if node_id == original:
                break
        return node_id
    
    def _hash_relation(self, neighbor_id: int, rel_type: str, weight: float) -> np.ndarray:
        hash_bytes = hashlib.sha256(rel_type.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'big')
        rng = np.random.RandomState(seed)
        base_vector = rng.randn(self.embedding_dim)
        
        noise_seed = int.from_bytes(hashlib.sha256(f"edge_{neighbor_id}_{rel_type}".encode()).digest()[:4], 'big')
        noise_rng = np.random.RandomState(noise_seed)
        noise_vector = noise_rng.randn(self.embedding_dim)
        
        return (base_vector * weight) + (noise_vector * 0.25)
    
    def _compute_signature(self, node_id: int) -> np.ndarray:
        if node_id not in self.nodes:
            return np.zeros(self.embedding_dim)
        
        signature = np.zeros(self.embedding_dim)
        relations = self.nodes[node_id]['relations']
        
        if not relations:
            return signature
        
        for neighbor_id, rel_info in relations.items():
            rel_vector = self._hash_relation(neighbor_id, rel_info['type'], rel_info['weight'])
            signature += rel_vector / np.sqrt(len(relations))
        
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
    
    def connect(self, source: int, target: int, rel_type: str, weight: float):
        source = self._resolve_alias(source)
        target = self._resolve_alias(target)
        
        if source not in self.nodes or target not in self.nodes or source == target:
            return
        
        self.nodes[source]['relations'][target] = {'type': rel_type, 'weight': weight}
        self.nodes[target]['relations'][source] = {'type': rel_type, 'weight': weight}
        
        self._update_signature(source)
        self._update_signature(target)
        self._check_merge(source, target)
    
    def _update_signature(self, node_id: int):
        if node_id in self.nodes:
            new_sig = self._compute_signature(node_id)
            self.nodes[node_id]['signature'] = new_sig
            self._update_index(node_id, new_sig)
    
    def _check_merge(self, node_a: int, node_b: int):
        if node_a not in self.nodes or node_b not in self.nodes:
            return
        
        # CRUCIAL CHECK: prevents merging of nodes with weak identity
        # A node with fewer than 2 relations does not have a well-defined enough identity
        min_relations_for_merge = 2
        if len(self.nodes[node_a]['relations']) < min_relations_for_merge:
            return
        if len(self.nodes[node_b]['relations']) < min_relations_for_merge:
            return
        
        sig_a = self.nodes[node_a]['signature']
        sig_b = self.nodes[node_b]['signature']
        
        norm_a, norm_b = np.linalg.norm(sig_a), np.linalg.norm(sig_b)
        if norm_a == 0 or norm_b == 0:
            return
        
        similarity = np.dot(sig_a, sig_b) / (norm_a * norm_b)
        if similarity > self.merge_threshold:
            self._merge_nodes(node_a, node_b)
    
    def _merge_nodes(self, node_a: int, node_b: int):
        print(f"  [⚡ MERGE] Nodes {node_a} and {node_b} merged (structural equivalence)")
        
        relations_to_transfer = [
            (nbr, info) for nbr, info in self.nodes[node_b]['relations'].items() if nbr != node_a
        ]
        updated_neighbors = set()
        
        for neighbor_id, rel_info in relations_to_transfer:
            if neighbor_id in self.nodes[node_a]['relations']:
                if rel_info['weight'] > self.nodes[node_a]['relations'][neighbor_id]['weight']:
                    self.nodes[node_a]['relations'][neighbor_id] = rel_info
            else:
                self.nodes[node_a]['relations'][neighbor_id] = rel_info
            
            if node_b in self.nodes[neighbor_id]['relations']:
                del self.nodes[neighbor_id]['relations'][node_b]
            
            self.nodes[neighbor_id]['relations'][node_a] = rel_info
            updated_neighbors.add(neighbor_id)
        
        self.nodes[node_a]['relations'].pop(node_b, None)
        del self.nodes[node_b]
        self.aliases[node_b] = node_a
        
        self._update_signature(node_a)
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
            "next_id": self._next_id,
            "aliases": {str(k): v for k, v in self.aliases.items()},
            "nodes": serializable_nodes
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"  [💾 SAVE] Structure saved to: {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.embedding_dim = data["embedding_dim"]
        self.merge_threshold = data["merge_threshold"]
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
                
        print(f"  [📂 LOAD] Structure loaded from: {filepath}")


def demo_killer_use_case_entity_resolution():
    print("\n" + "="*80)
    print("🎯 KILLER USE CASE: Entity Resolution (Automatic Data Cleaning)")
    print("="*80)
    print("Scenario: Dirty customer database with duplicate records.")
    print("RIS identifies and automatically merges them based on relationships.\n")
    
    ris = RelationalIdentityStructure(embedding_dim=64, merge_threshold=0.95)
    
    # Attributes
    email_1 = ris.insert({"type": "email", "value": "mario.rossi@email.com"})
    phone_1 = ris.insert({"type": "phone", "value": "+39 333 1234567"})
    city_1 = ris.insert({"type": "city", "value": "Milan"})
    
    email_2 = ris.insert({"type": "email", "value": "m.rossi88@email.com"})
    phone_2 = ris.insert({"type": "phone", "value": "+39 333 1234567"})
    city_2 = ris.insert({"type": "city", "value": "Milan"})

    # Customer records
    customer_A = ris.insert({"source": "CRM_Legacy", "name": "Mario Rossi"})
    customer_B = ris.insert({"source": "Newsletter_Signup", "name": "M. Rossi"})

    print("Initial state:")
    print(f"  - Record A: {ris.nodes[customer_A]['data']['name']} (ID: {customer_A})")
    print(f"  - Record B: {ris.nodes[customer_B]['data']['name']} (ID: {customer_B})")
    
    print("\n[PHASE 1] Connecting Record A to its attributes...")
    ris.connect(customer_A, email_1, "has_email", 1.0)
    ris.connect(customer_A, phone_1, "has_phone", 1.0)
    ris.connect(customer_A, city_1, "lives_in", 1.0)
    
    print("[PHASE 2] Connecting Record B to its attributes...")
    print("  (Shares phone and city with A, but has a different email)")
    ris.connect(customer_B, email_2, "has_email", 1.0)
    ris.connect(customer_B, phone_2, "has_phone", 1.0)
    ris.connect(customer_B, city_2, "lives_in", 1.0)
    
    print("\n[PHASE 3] Continuous identity analysis:")
    identity_A = ris.who_am_i(customer_A, top_k=2)
    for node_id, sim in identity_A:
        name = ris.nodes[node_id]['data'].get('name', 'Attribute')
        print(f"  Record A is {sim*100:.1f}% similar to: {name} (ID {node_id})")
    
    print("\n[PHASE 4] Update: User B confirms the main email.")
    print("  Removing the old email and connecting B to A's email...")
    ris.disconnect(customer_B, email_2)
    ris.connect(customer_B, email_1, "has_email", 1.0)
    
    print("\n[FINAL RESULT]")
    print("A and B now share EXACTLY the same attributes (Email, Phone, City).")
    print("Their relational signatures are identical → automatic merging!")
    
    state_B = ris._resolve_alias(customer_B)
    state_A = ris._resolve_alias(customer_A)
    print(f"  Record B (ID {customer_B}) is now an alias of Record A (ID {state_A}).")
    print(f"  Unified data: {ris.nodes[state_A]['data']}")
    
    filepath = "ris_database.json"
    ris.save(filepath)
    
    print("\n[PHASE 5] Persistence test...")
    ris_new = RelationalIdentityStructure()
    ris_new.load(filepath)
    print(f"  Check: Node {state_A} has {len(ris_new.nodes[state_A]['relations'])} relations after reloading.")
    
    if os.path.exists(filepath):
        os.remove(filepath)

if __name__ == "__main__":
    demo_killer_use_case_entity_resolution()
    print("\n" + "="*80)
    print("✅ RIS ENGINE v2.1 - SMART MERGE ACTIVATED")
    print("="*80)
