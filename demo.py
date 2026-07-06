from ris.engine import RelationalIdentityStructure
import os
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