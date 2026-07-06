import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from src.relational_identity import RelationalIdentityStructure

def main():
    # Use a high threshold to avoid false positives (only near‑perfect matches)
    ris = RelationalIdentityStructure(embedding_dim=64, merge_threshold=0.99, verbose=True)

    # 1. Leaf nodes with clear types
    email = ris.insert(data={"type": "email", "value": "shared@example.com"})
    phone = ris.insert(data={"type": "phone", "value": "+39 333 1234567"})

    # 2. Two person nodes
    alice = ris.insert(data={"name": "Alice Rossi", "type": "person"})
    charlie = ris.insert(data={"name": "Charlie Brown", "type": "person"})

    print(f"Alice ID: {alice}, Charlie ID: {charlie}")
    print(f"Email ID: {email}, Phone ID: {phone}")

    # 3. Alice connects to both email and phone (2 relations)
    ris.connect(alice, email, "has_email")
    ris.connect(alice, phone, "has_phone")

    # 4. Charlie connects only to the phone (1 relation) → weak identity → no merge
    ris.connect(charlie, phone, "has_phone")

    # 5. Charlie also connects to the SAME email → now he also has 2 identical relations → MERGE!
    ris.connect(charlie, email, "has_email")

    print("\n--- Final result ---")
    active = ris.get_active_nodes()
    print(f"Active nodes: {active}")

    # Resolve aliases: if they were merged, both IDs will point to the same survivor
    final_alice = ris._resolve_alias(alice)
    final_charlie = ris._resolve_alias(charlie)

    print(f"Alice resolved to: {final_alice}")
    print(f"Charlie resolved to: {final_charlie}")

    if final_alice == final_charlie:
        print(f"✅ Alice and Charlie have been merged into node {final_alice}")
        data = ris.get_node_data(final_alice)
        print(f"Survivor data: {data}")
    else:
        print("❌ They were not merged (unexpected)")

    ris.save("alice_charlie_clean.json")

if __name__ == "__main__":
    main()