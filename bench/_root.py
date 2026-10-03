#!/usr/bin/env python3
"""Where everything lives, resolved instead of hardcoded.

Every script in bench/ used to open with ROOT = r"C:\\Users\\idont\\Downloads\\4", which
is fine on the machine it was written on and useless anywhere else. Resolution order:

    1. $MOE_ROOT, if set
    2. the parent of the directory this file is in  (bench/.. == the project root)

Override the model or the binary individually with $MOE_MODEL / $MOE_SERVER if your
layout differs.
"""
import os

ROOT = os.environ.get("MOE_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENCH = os.path.join(ROOT, "bench")
LOGD  = os.path.join(BENCH, "logs")

EXE = os.environ.get("MOE_SERVER") or os.path.join(
        ROOT, "src", "lcpp", "build-ms", "bin", "llama-server.exe")

MODEL = os.environ.get("MOE_MODEL") or os.path.join(
        ROOT, "models", "flashnext", "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf")

HOT  = os.path.join(ROOT, "models", "flashnext", "experts.hot")
HEAT_TRAIN = os.path.join(BENCH, "expert_train.heat")


def check(what=("exe", "model")):
    """Fail early and legibly instead of 600 s into a sweep."""
    missing = []
    if "exe" in what and not os.path.exists(EXE):
        missing.append("server binary: %s  (build it, or set $MOE_SERVER)" % EXE)
    if "model" in what and not os.path.exists(MODEL):
        missing.append("model: %s  (download it, or set $MOE_MODEL)" % MODEL)
    if missing:
        raise SystemExit("cannot start:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    for k in ("ROOT", "BENCH", "EXE", "MODEL", "HOT", "HEAT_TRAIN"):
        v = globals()[k]
        print("%-10s %s%s" % (k, v, "" if os.path.exists(v) else "   [missing]"))
