import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.relational_identity import RelationalIdentityStructure


class TestRIS:

    def test_insert_and_resolve(self):
        ris = RelationalIdentityStructure(verbose=False)
        n1 = ris.insert(data={"name": "A"})
        n2 = ris.insert(data={"name": "B"})
        
        assert n1 in ris.nodes
        assert n2 in ris.nodes
        assert ris._resolve_alias(n1) == n1

    def test_connection_and_signature(self):
        ris = RelationalIdentityStructure(verbose=False)
        a = ris.insert()
        b = ris.insert()
        ris.connect(a, b, "knows")
        
        assert len(ris.nodes[a]['relations']) == 1
        assert len(ris.nodes[b]['relations']) == 1
        # La firma non deve essere zero vettore
        assert np.linalg.norm(ris.nodes[a]['signature']) > 0

    def test_no_merge_with_weak_identity(self):
        """Nodi con meno di 2 relazioni NON devono fondersi."""
        ris = RelationalIdentityStructure(merge_threshold=0.5, verbose=False)  # soglia bassa per forzare
        a = ris.insert()
        b = ris.insert()
        c = ris.insert()  # nodo condiviso
        
        # a e b hanno UNA sola relazione (entrambi con c)
        ris.connect(a, c, "friend")
        ris.connect(b, c, "friend")
        
        # Nonostante la soglia bassa, non si fondono perché hanno < 2 relazioni
        assert a in ris.nodes
        assert b in ris.nodes
        # Verifichiamo che non ci siano alias
        assert a not in ris.aliases.values()
        assert b not in ris.aliases.values()

    def test_merge_with_strong_identity(self):
        """Nodi con >2 relazioni identiche devono fondersi."""
        ris = RelationalIdentityStructure(merge_threshold=0.95, verbose=False)
        
        # Creiamo 3 entità foglia (A, B, C)
        leaf1 = ris.insert()
        leaf2 = ris.insert()
        leaf3 = ris.insert()
        
        # Nodo X
        x = ris.insert()
        ris.connect(x, leaf1, "rel1")
        ris.connect(x, leaf2, "rel2")
        ris.connect(x, leaf3, "rel3")
        
        # Nodo Y (inizialmente con una sola relazione)
        y = ris.insert()
        ris.connect(y, leaf1, "rel1")
        ris.connect(y, leaf2, "rel2")
        ris.connect(y, leaf3, "rel3")
        
        # Ora X e Y hanno le stesse 3 relazioni -> devono fondersi
        # Eseguiamo un ultimo connect per innescare il controllo
        ris.connect(y, leaf3, "rel3")  # ridondante ma innesca il check
        
        # Verifichiamo che Y sia diventato alias di X
        assert y in ris.aliases
        assert ris._resolve_alias(y) == x
        assert x in ris.nodes
        assert y not in ris.nodes
        
        
        

    def test_save_and_load(self):
        ris = RelationalIdentityStructure(verbose=False)
        n1 = ris.insert(data={"test": "value"})
        n2 = ris.insert()
        ris.connect(n1, n2, "knows")
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            ris.save(tmp_path)
            
            ris2 = RelationalIdentityStructure(verbose=False)
            ris2.load(tmp_path)
            
            assert len(ris2.nodes) == 2
            assert ris2._next_id == ris._next_id
            assert ris2.get_node_data(n1) == {"test": "value"}
            
        finally:
            os.unlink(tmp_path)

    def test_who_am_i(self):
        ris = RelationalIdentityStructure(verbose=False)
        # Creiamo un cluster di 3 nodi identici
        leaves = [ris.insert() for _ in range(3)]
        center = ris.insert()
        for l in leaves:
            ris.connect(center, l, "link")
        
        # Nodo esterno diverso
        outsider = ris.insert()
        ris.connect(outsider, ris.insert(), "different")
        
        sim = ris.who_am_i(center, top_k=5)
        # Deve trovare gli altri nodi del cluster (che sono stati fusi? 
        # In realtà se hanno meno di 2 relazioni non si fondono)
        # Quindi la similarità sarà alta.
        assert len(sim) > 0
        # Il primo risultato deve essere un nodo del cluster (sim > 0.9)
        assert sim[0][1] > 0.9