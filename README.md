# Parameterized Koopman

A single model is learned for a whole family of closed-loop systems
x_{k+1} = F_theta(x_k), where theta is the parameter of a feedback law. The encoder,
the decoder and the Koopman matrix all depend on theta:

```
x_hat_k = psi(K(theta)^k phi(x_0, theta), theta)
```

The model is trained in two stages. A scout model is used to find a region C of
feedback parameters where the predictions pass a finite-horizon test, and the final
model is trained on parameters from C. The final model is then tested on fresh
parameters from C, and C is shrunk and tested again until at least 95% of them pass
(15% worst-case error over 12 steps). The same model can then be used to pick a
feedback parameter once, at the initial state, by minimizing a cost of the predicted
rollout inside C.

## Preprint
https://github.com/kirangkrishnan/Parameterized-Koopman/blob/main/Krishnan_Deka_2026_preprint.pdf

The code of the first preprint is kept under the tag `arxiv-v1`. The main branch
matches the revised version, in which every region passes the final check with the
final model, the prediction benchmark uses fresh parameters on all six systems, and
ablation B tests 30 parameters drawn from the region.

## Setup

```
conda create -n pkoopman python=3.11
conda activate pkoopman
pip install -r requirements.txt
```

Only the two MuJoCo benchmarks need `gymnasium[mujoco]`. CasADi is needed for the
MPC baselines.

## Layout

```
pkoopman/
  model.py          PolicyKoopman: encoder, decoder and operator network
  training.py       multi-step loss (forward and backward rollouts) and training loop
  region.py         construction of the verified region C
  selection.py      one-shot selection of theta (SLSQP inside the hull of C)
  metrics.py        worst-case normalized prediction error
  systems/          Van der Pol, slow manifold, two-link arm, cartpole, MuJoCo pendulums
baselines/          PKNN, EDMD, kernel EDMD, deep Koopman, NMPC, Koopman MPC
scripts/            training, experiments and plots
cluster/            SLURM job scripts for the experiments
checkpoints/        trained models and their verified regions
results/            the numbers behind the tables and figures of the paper
```

`checkpoints/<system>.pt` is the model used in the paper and
`checkpoints/<system>_region.npy` holds samples of its verified region, the one that
passed the final check. `<system>_phase3.json` records that check, and
`<system>_region_pre.npy` is the region before it, which ablation B samples from.
Baseline models are in `checkpoints/baselines/` and `checkpoints/control/`.

```python
import torch
from pkoopman.model import load_model

model = load_model("checkpoints/vdp.pt")
x0 = torch.tensor([[0.5, -0.2]])
theta = torch.tensor([[2.0, 3.0, 0.0, 0.0]])
x_hat = model.predict(x0, theta, steps=12)    # shape (1, 12, 2)
```

## Systems

| name | system | state dim | theta dim | dt (s) |
|---|---|---|---|---|
| `vdp` | Van der Pol | 2 | 4 | 0.05 |
| `slow_manifold` | polynomial slow manifold | 2 | 4 | 0.05 |
| `manipulator` | two-link arm with PD feedback | 4 | 4 | 0.05 |
| `cartpole` | cart-pole | 4 | 4 | 0.05 |
| `mujoco_ip` | Gymnasium InvertedPendulum-v5 | 4 | 4 | 0.04 |
| `mujoco_idp` | Gymnasium InvertedDoublePendulum-v5 | 6 | 6 | 0.05 |

For the cartpole and the MuJoCo systems the model sees a normalized parameter in
[-1, 1]^p, which is mapped linearly to the physical gains (`to_gains`). The
manipulator is the one system where no region passes at 15%; its model is trained
on a 1.2 times wider box of initial states and its region is built and checked at
20% (settings in `scripts/train.py`).

## Training

```
python scripts/train.py vdp
python scripts/plot_region.py runs/train/vdp_region_history.npz
```

This trains the scout model, builds the region, trains the final model, runs the
final check and writes everything to `runs/train/`. Settings for each system are
at the top of `scripts/train.py`. With 5000 epochs per stage a run takes several
hours on a GPU. `scripts/verify_region.py <system>` runs only the final check, for
a model that is already trained.

## Reproducing the paper

All commands use the models in `checkpoints/` and are run from the repository root.

**Table II and Fig. 7 (prediction).**

```
python scripts/prediction_benchmark.py vdp
python scripts/prediction_benchmark.py slow_manifold
python scripts/prediction_benchmark.py manipulator
python scripts/prediction_benchmark.py cartpole
python scripts/prediction_benchmark.py mujoco_ip
python scripts/prediction_benchmark.py mujoco_idp
python scripts/plot_prediction.py
```

Each system is tested on 30 new parameters drawn uniformly from its verified region,
with a seed not used in training, and 20 initial conditions each. The script reports
the worst-case error over the first 12 steps and the error at every step up to 20.
PKNN and deep Koopman are loaded from `checkpoints/baselines/`; add `--retrain` to
train them again (set `OMP_NUM_THREADS=2`, otherwise training is slow on many-core
machines). EDMD and kernel EDMD are fitted at the centroid of the region. Retrained
PKNN models can differ noticeably from the shipped ones, as its training is
sensitive to floating point rounding. `scripts/region_check.py <system>` estimates
the pass rate of the region on an independent draw of 1500 parameters.

**Figs. 4-6 (ablations).**

```
python scripts/ablation_retraining.py --seeds 8 17 42 0 99
python scripts/ablation_dictionary.py --seeds 0 8 17 42 99
python scripts/ablation_dictionary.py --system slow_manifold --seeds 0 8 17 42 99
python scripts/plot_ablations.py
```

`ablation_retraining.py` (ablation A) compares one conditioned model with a separate
Koopman model retrained at each of five cartpole parameters and records the
training cost. `ablation_dictionary.py` (ablation B) compares full conditioning
with a fixed dictionary that uses the same conditioned operator K(theta). Both
variants are trained on the same random-parameter data and tested at 30 parameters
drawn from the region, and the script also computes the latent diagnostic. The
paper reports the cartpole and, in the text, `vdp`, `slow_manifold` and
`manipulator`. Both scripts take hours per seed; `--seeds` splits the work.

**Table III and Fig. 8 (one-shot selection).**

```
python scripts/control_comparison.py --system vdp --method ours_z
python scripts/control_comparison.py --system vdp --method ours_x
python scripts/control_comparison.py --system vdp --method nmpc
python scripts/control_comparison.py --system vdp --method kmpc_edmd
python scripts/control_comparison.py --system vdp --method kmpc_deep
python scripts/plot_control.py
```

`ours_z` and `ours_x` select theta once inside the verified region, with the latent
or the decoded objective, and then run the fixed feedback law; their `t_tot`
includes the selection. The per-system weights and horizons are in
`pkoopman/selection.py`. The baselines are nonlinear MPC on the true model and
receding-horizon Koopman MPC with an EDMD or a deep Koopman model. The runs are
deterministic, so the five seeds of the paper (8, 17, 42, 0, 99) only change the
measured wall-clock time. The NMPC script for the double pendulum was not kept.

`results/` holds the numbers of the paper as produced by these scripts.

**On a SLURM cluster** (written for Aalto Triton):

```
bash cluster/setup_env.sh        # once, creates the conda environment
bash cluster/submit_all.sh       # region check, prediction and both ablations
python scripts/collect_results.py   # afterwards, on the login node
```

`cluster/train.sh <system> <out>` runs the full training pipeline for one system.
Logs go to `logs/`, results to `runs/`.

## Citation

```
@article{krishnan2026policy,
  title  = {Policy-Conditioned Koopman Representations for Parameterized Closed-Loop Systems},
  author = {Krishnan, Kiran G. and Deka, Shankar A.},
  year   = {2026}
}
```
