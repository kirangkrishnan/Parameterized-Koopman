"""Summarize everything under runs/ into one Markdown file (runs/summary.md).

Reads whatever is there, so it can be run while cluster jobs are still going:
  runs/region_check/<system>/chunk*.npz           (region_check.py)
  runs/prediction_benchmark/<system>[_mode].json  (prediction_benchmark.py)
  runs/control_benchmark/<system>/<tag>/*.jsonl   (control_benchmark.py)
  runs/ablation_retraining/seed_*.json            (ablation_retraining.py)
  runs/ablation_dictionary/**/results.json        (ablation_dictionary.py)

    python scripts/collect_results.py
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import beta

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOLERANCES = (10.0, 15.0, 20.0)


def clopper_pearson(k, n, level=0.95):
    """Two-sided interval for a binomial proportion k/n."""
    a = 1.0 - level
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{digits}f}"


def region_section(runs):
    lines = ["## Region check (Phase 3 on fresh parameters)", "",
             "Survival = fraction of fresh parameters from the hull whose worst-case error over "
             "the verification initial conditions is below the tolerance. The interval is the "
             "95% Clopper-Pearson interval for the violation probability under uniform sampling "
             "of the hull. 'valid' also requires the true closed loop to pass the system check "
             "used when the region was built.", ""]
    systems = sorted(d for d in glob.glob(os.path.join(runs, "region_check", "*")) if os.path.isdir(d))
    if not systems:
        return []
    lines += ["| system | n | hull / box volume | horizon | tolerance | survival | violation 95% CI | survival (valid) |",
              "|---|---|---|---|---|---|---|---|"]
    for d in systems:
        parts = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "chunk*.npz")))]
        if not parts:
            continue
        errors = np.concatenate([p["errors"] for p in parts])
        valid = np.concatenate([p["valid"] for p in parts])
        horizons = parts[0]["horizons"]
        vol = next((float(p["volume_fraction"]) for p in parts if "volume_fraction" in p.files), None)
        n = len(errors)
        for h_i, h in enumerate(horizons):
            for tol in TOLERANCES:
                ok = errors[:, h_i] < tol
                lo, hi = clopper_pearson(int(n - ok.sum()), n)
                vol_txt = "-" if vol is None else f"{vol:.2e}"
                lines.append(f"| {os.path.basename(d)} | {n} | {vol_txt} | {h} | {tol:g}% | "
                             f"{ok.mean():.3f} | [{lo:.3f}, {hi:.3f}] | {(ok & valid).mean():.3f} |")
    return lines + [""]


def prediction_section(runs):
    files = sorted(glob.glob(os.path.join(runs, "prediction_benchmark", "*.json")))
    if not files:
        return []
    lines = ["## Prediction", "",
             "Worst-case error over 12 steps (%), mean +- std over scored cases. "
             "'<15%' is the fraction of cases below the threshold; 'not scored' counts cases "
             "whose true trajectory left the bounds.", "",
             "| system | test parameters | method | mean +- std | median | <15% | scored | not scored |",
             "|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.load(open(f))
        mode = r.get("thetas", "region")
        h = str(r["train_horizon"])
        for m, v in r["methods"].items():
            s = v["worst_case"][h]
            lines.append(f"| {r['system']} | {mode} | {m} | {fmt(s['mean'])} +- {fmt(s['std'])} | "
                         f"{fmt(s['median'])} | {fmt(s['below_threshold'])} | {v['n_cases']} | "
                         f"{v.get('n_true_failed', '-')} |")
    return lines + [""]


def control_section(runs):
    root = os.path.join(runs, "control_benchmark")
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(root, "*", "*", "*.jsonl")):
        system, tag = f.split(os.sep)[-3:-1]
        groups[(system, tag)] += [json.loads(line) for line in open(f) if line.strip()]
    if not groups:
        return []
    lines = ["## Control from many initial states", "",
             "Success = regulated to ||x_T|| below the tolerance without early termination. "
             "Cost = integral of ||x||^2 + 0.1 ||u||^2 over the episode (successful episodes). "
             "t_online includes the selection time of the one-shot methods.", "",
             "| system | method | episodes | success (95% CI) | median settling (s) | mean cost | "
             "mean effort | t_select (s) | t_online (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (system, tag) in sorted(groups):
        recs = groups[(system, tag)]
        n = len(recs)
        k = sum(r["success"] for r in recs)
        lo, hi = clopper_pearson(k, n)
        succ = [r for r in recs if r["success"]]
        settle = np.median([r["settling_time"] for r in succ]) if succ else None
        cost = np.mean([r["cost"] for r in succ]) if succ else None
        effort = np.mean([r["effort"] for r in succ]) if succ else None
        t_sel = np.mean([r["t_select"] for r in recs])
        t_on = [r["t_online"] for r in recs]
        lines.append(f"| {system} | {tag} | {n} | {k / n:.2f} [{lo:.2f}, {hi:.2f}] | {fmt(settle)} | "
                     f"{fmt(cost, 3)} | {fmt(effort, 3)} | {fmt(t_sel, 3)} | "
                     f"{np.mean(t_on):.3f} +- {np.std(t_on):.3f} |")

    # paired comparison with the simulator reference: cost of the surrogate's choice
    # relative to the choice made on the true simulator, same initial state
    pairs = []
    for (system, tag) in sorted(groups):
        if not tag.startswith("sim_opt"):
            continue
        ref = {r["index"]: r for r in groups[(system, tag)] if r["success"]}
        suffix = tag[len("sim_opt"):]
        for other in ("ours_x", "ours_z", "centroid"):
            recs = groups.get((system, other + suffix), [])
            ratios = [r["cost"] / ref[r["index"]]["cost"] for r in recs
                      if r["success"] and r["index"] in ref and ref[r["index"]]["cost"] > 0]
            if ratios:
                pairs.append(f"| {system} | {other}{suffix} | {len(ratios)} | {np.median(ratios):.3f} | "
                             f"{np.percentile(ratios, 90):.3f} |")
    if pairs:
        lines += ["", "Closed-loop cost relative to the simulator reference (sim_opt), "
                  "paired by initial state, both successful:", "",
                  "| system | method | pairs | median ratio | 90th percentile |", "|---|---|---|---|---|"] + pairs
    return lines + [""]


def ablation_section(runs):
    lines = []
    files = sorted(glob.glob(os.path.join(runs, "ablation_retraining", "seed_*.json")))
    if files:
        rs = [json.load(open(f)) for f in files]
        cond_train = [r["conditioned_train_time"] for r in rs]
        cond_data = [r["conditioned_data_time"] for r in rs]
        unc = [u for r in rs for u in r["unconditioned"]]
        lines += ["## Ablation A (one conditioned model vs a model per parameter)", "",
                  f"Seeds: {len(rs)}. Conditioned model: data {np.mean(cond_data):.0f} s, "
                  f"training {np.mean(cond_train):.0f} +- {np.std(cond_train):.0f} s "
                  f"(trained once for all parameters).",
                  f"Per-parameter model: data {np.mean([u['data_time'] for u in unc]):.0f} s, "
                  f"training {np.mean([u['train_time'] for u in unc]):.0f} +- "
                  f"{np.std([u['train_time'] for u in unc]):.0f} s (each parameter).", "",
                  "| parameter | conditioned error (%) | per-parameter error (%) |", "|---|---|---|"]
        for i in range(len(rs[0]["alphas"])):
            c = [x["mean_err"] for r in rs for x in r["conditioned"] if x["alpha_idx"] == i]
            u = [x["mean_err"] for r in rs for x in r["unconditioned"] if x["alpha_idx"] == i]
            lines.append(f"| {i} | {np.mean(c):.3f} +- {np.std(c):.3f} | "
                         f"{fmt(np.mean(u) if u else None, 3)} +- {fmt(np.std(u) if u else None, 3)} |")
        lines.append("")
    files = sorted(glob.glob(os.path.join(runs, "ablation_dictionary", "**", "results.json"), recursive=True))
    if files:
        recs = {"conditioned": [], "fixed_dictionary": []}
        for f in files:
            r = json.load(open(f))
            for v in recs:
                recs[v] += r[v]
        n_alpha = 1 + max(x["alpha_idx"] for x in recs["conditioned"])
        per = {v: np.array([np.mean([x["mean_err"] for x in recs[v] if x["alpha_idx"] == i])
                            for i in range(n_alpha)]) for v in recs}
        wins = int(np.sum(per["conditioned"] < per["fixed_dictionary"]))
        lines += ["## Ablation B (conditioned vs fixed dictionary), all test parameters", "",
                  f"Mean over {n_alpha} parameters: conditioned {per['conditioned'].mean():.3f}%, "
                  f"fixed dictionary {per['fixed_dictionary'].mean():.3f}%. "
                  f"Conditioned is lower at {wins} of {n_alpha}.", ""]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    args = parser.parse_args()
    lines = ["# Experiment summary", ""]
    for section in (region_section, prediction_section, control_section, ablation_section):
        lines += section(args.runs)
    path = os.path.join(args.runs, "summary.md")
    os.makedirs(args.runs, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
