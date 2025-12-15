import sys
from pathlib import Path

# Asigurăm că rădăcina proiectului (un nivel mai sus față de tools/) e în sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pickle
import csv

P = Path("data/agent2_qtables.pkl")
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

print(f"[inspect_q] ROOT = {ROOT}")
print(f"[inspect_q] sys.path[0] = {sys.path[0]}")
print(f"[inspect_q] looking for pickle at: {P.resolve()}")

if not P.exists():
    print("File not found:", P)
    raise SystemExit(1)

try:
    with P.open("rb") as f:
        data = pickle.load(f)
except Exception as e:
    print("Failed to load pickle:")
    import traceback
    traceback.print_exc()
    raise

Q = data.get("Q", {})
for comp, tab in Q.items():
    out = OUT_DIR / f"{comp}_q.csv"
    with out.open("w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["state_x", "state_y", "action", "q_value"])
        for (state, action), q in tab.items():
            try:
                x, y = state
            except Exception:
                x = state
                y = ""
            action_name = getattr(action, "name", str(action))
            writer.writerow([x, y, action_name, q])
    print("Wrote", out)
