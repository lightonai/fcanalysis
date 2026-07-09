"""Validate the semantic correction invariant.

Invariant: a turn's final label is anti  <=>  a usable correction lives at
verification.correction. The only allowed exceptions are legacy old-stage-2
failures (anti with no correction). A justified turn must NEVER carry a
correction. Prints PASS/FAIL; exit code 1 on violation so a chain can gate on it.

By default this checks the local generated mixture output. Override with
``FINAL_DIR=/path/to/results``.
"""

import glob
import json
import os
import sys
from pathlib import Path

FINAL = Path(
    os.environ.get(
        "FINAL_DIR",
        "/mnt/nfs/ytahtah/fcanalysis-clean/semantic_results_mixture",
    )
)


def is_anti(category: str) -> bool:
    return bool(category) and (
        category.startswith("ANTI_") or category == "OTHER_UNJUSTIFIED"
    )


anti = anti_corr = anti_nocorr = just = just_corr = contested = 0
viol = []
for f in glob.glob(str(FINAL / "*.jsonl")):
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "error" in d:
            continue
        for c in d.get("classifications", []):
            ver = c.get("verification") or {}
            has_corr = isinstance(ver, dict) and bool(ver.get("correction"))
            if c.get("contested"):
                contested += 1
            if is_anti(c.get("category", "")):
                anti += 1
                if has_corr:
                    anti_corr += 1
                else:
                    anti_nocorr += 1
            else:
                just += 1
                if has_corr:
                    just_corr += 1
                    if len(viol) < 5:
                        viol.append(
                            (
                                d["sample_id"],
                                c["turn_index"],
                                c.get("category"),
                                "JUSTIFIED w/ corr",
                            )
                        )

print(
    f"anti={anti}  anti_with_corr={anti_corr}  anti_without_corr={anti_nocorr} (legacy)"
)
print(f"justified={just}  justified_with_corr={just_corr} (MUST be 0)")
print(f"contested={contested}")
ok = just_corr == 0
print("INVARIANT:", "PASS" if ok else "FAIL")
for v in viol:
    print("  violation:", v)
sys.exit(0 if ok else 1)
