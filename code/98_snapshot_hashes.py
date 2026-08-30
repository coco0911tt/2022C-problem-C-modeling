from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def semantic_csv_hash(path: Path) -> str:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.reindex(sorted(frame.columns), axis=1)
    if len(frame):
        frame = frame.sort_values(list(frame.columns), na_position="last").reset_index(drop=True)
    payload = frame.to_csv(index=False, float_format="%.15g", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    hashes = {}
    for path in sorted((ROOT / "results").rglob("*.csv")):
        if "99_summary" in path.parts:
            continue
        hashes[path.relative_to(ROOT).as_posix()] = semantic_csv_hash(path)
    output = ROOT / "reports" / "pre_rerun_core_hashes.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SNAPSHOT_CSV_COUNT={len(hashes)}")


if __name__ == "__main__":
    main()
