from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "00_ingest_preprocess.py", "test_common.py", "10_q1_1_association.py",
    "11_q1_2_weather_effect.py", "12_q1_3_counterfactual.py",
    "20_q2_1_classification.py", "21_q2_2_subclasses.py",
    "30_q3_unknown.py", "40_q4_network.py", "99_finalize.py",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    started = time.time()
    log_path = ROOT / "logs" / "run_all.log"
    with log_path.open("w", encoding="utf-8") as log:
        for name in SCRIPTS:
            stamp = f"RUN {name}\n"
            print(stamp, end="", flush=True)
            log.write(stamp)
            log.flush()
            completed = subprocess.run(
                [sys.executable, str(ROOT / "code" / name)], cwd=ROOT,
                text=True, encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            log.write(completed.stdout)
            log.write(f"EXIT {name}={completed.returncode}\n")
            log.flush()
            if completed.returncode:
                print(completed.stdout)
                raise SystemExit(completed.returncode)
            print(f"OK  {name}", flush=True)
        elapsed = time.time() - started
        log.write(f"TOTAL_SECONDS={elapsed:.3f}\n")
        print(f"ALL_DONE_SECONDS={elapsed:.3f}")


if __name__ == "__main__":
    main()
