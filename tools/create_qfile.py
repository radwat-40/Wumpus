# tools/create_qfile.py
import sys
from pathlib import Path
import traceback

# Asigurăm că rădăcina proiectului (un nivel mai sus față de tools/) e în sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # acum importurile locale vor funcționa
    from agents.agent2 import YahiaAgent
    p = Path("data")
    p.mkdir(parents=True, exist_ok=True)
    fq = p / "agent2_qtables.pkl"

    a = YahiaAgent(0, 0, "A2")
    a.save_q_tables(str(fq))

    print("Saved Q-file:", fq.resolve())
    print("Exists:", fq.exists(), "Size:", fq.stat().st_size if fq.exists() else "N/A")
except Exception:
    print("Exception while creating Q-file:")
    traceback.print_exc()