"""Phase 3 for a model that has already been trained: test it on fresh parameters
from its region and shrink the region until enough of them pass.

train.py runs this step itself after training the final model. This script does
the same for the models in checkpoints/, which were trained before Phase 3 was
part of train.py.

    python scripts/verify_region.py vdp
    python scripts/verify_region.py slow_manifold --threshold 15 --steps 12

Writes <out>/<system>.pt (a copy of the model), <out>/<system>_region.npy and
<out>/<system>_phase3.json, so that the other scripts can use the folder with
--checkpoints.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pkoopman.model import load_model
from pkoopman.region import verify_region
from pkoopman.systems import experiment_system
from train import CONFIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system")
    parser.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--acceptance", type=float, default=0.95)
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--out", default=os.path.join(ROOT, "runs", "verified"))
    args = parser.parse_args()

    name = args.system
    system = experiment_system(name)
    model = load_model(os.path.join(args.checkpoints, f"{name}.pt"))
    region = np.load(os.path.join(args.checkpoints, f"{name}_region.npy"))
    check = verify_region(model, system, region, np.random.default_rng(args.seed),
                          n_samples=args.n_samples, steps=args.steps, threshold=args.threshold,
                          acceptance=args.acceptance, require_valid=CONFIGS[name].get("require_valid", False))

    os.makedirs(args.out, exist_ok=True)
    shutil.copy(os.path.join(args.checkpoints, f"{name}.pt"), os.path.join(args.out, f"{name}.pt"))
    np.save(os.path.join(args.out, f"{name}_region.npy"), check["final"] if check["accepted"] else region)
    with open(os.path.join(args.out, f"{name}_phase3.json"), "w") as f:
        json.dump(dict(accepted=check["accepted"], threshold=args.threshold, steps=args.steps,
                       acceptance=args.acceptance, seed=args.seed, history=check["history"]), f, indent=1)
    print(f"[{name}] phase 3 {'accepted' if check['accepted'] else 'FAILED'}, written to {args.out}")


if __name__ == "__main__":
    main()
