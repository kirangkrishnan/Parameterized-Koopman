# Parameterized Koopman

A single model is learned for a whole family of closed-loop systems
x_{k+1} = F_theta(x_k), where theta is the parameter of a feedback law. The encoder,
the decoder and the Koopman matrix all depend on theta:

```
x_hat_k = psi(K(theta)^k phi(x_0, theta), theta)
```

The model is trained in two stages. A scout model is used to find a region C of
feedback parameters where the predictions pass a finite-horizon test, and the final
model is trained on parameters from C. The same model can then be used to pick a
feedback parameter once, at the initial state, by minimizing a cost of the predicted
rollout inside C.

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
```

`checkpoints/<system>.pt` is the model used in the paper and
`checkpoints/<system>_region.npy` holds the samples of its verified region.
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

| name | state | theta | dt |
|---|---|---|---|
| `vdp` | Van der Pol, 2 | 4 | 0.05 |
| `slow_manifold` | polynomial slow manifold, input in the first equation, 2 | 4 | 0.05 |
| `slow_manifold_u2` | same system, input in the second equation, 2 | 4 | 0.05 |
| `manipulator` | two-link arm with PD feedback, 4 | 4 | 0.05 |
| `cartpole` | cart-pole, 4 | 4 | 0.05 |
| `mujoco_ip` | Gymnasium InvertedPendulum-v5, 4 | 4 | 0.05 |
| `mujoco_idp` | Gymnasium InvertedDoublePendulum-v5, 6 | 6 | 0.05 |

For the cartpole and the MuJoCo systems the model sees a normalized parameter in
[-1, 1]^p, which is mapped linearly to the physical gains (`to_gains`).

## Training

```
python scripts/train.py vdp
python scripts/plot_region.py runs/train/vdp_region_history.npz
```

This trains the scout model, builds the region, trains the final model and writes
everything to `runs/train/`. Settings for each system are at the top of
`scripts/train.py`. With 5000 epochs per stage a run takes a few hours on a GPU.
For a quick check use `--scout-epochs 100 --final-epochs 100`.

## Experiments

**Prediction (Table II, Fig. 7).**

```
python scripts/prediction_benchmark.py vdp
python scripts/prediction_benchmark.py slow_manifold_u2
python scripts/prediction_benchmark.py manipulator
python scripts/prediction_benchmark.py cartpole
python scripts/plot_prediction.py
```

Each system is evaluated on 30 parameters from its region with 20 initial conditions
each. The script reports the worst-case error over the first 12 steps and the error
per step up to 20. PKNN and deep Koopman are loaded from `checkpoints/baselines/`;
add `--retrain` to train them again (set `OMP_NUM_THREADS=2`, otherwise training is
slow on many-core machines). EDMD and kernel EDMD are fitted at the centroid of the
region. Retrained PKNN models can differ noticeably from the shipped ones, its
training is sensitive to floating point rounding.

**Ablations on the cartpole (Figs. 4-6).**

```
python scripts/ablation_retraining.py --seeds 8 17 42 0 99
python scripts/ablation_dictionary.py --seeds 0 8 17 42 99
python scripts/plot_ablations.py
```

`ablation_retraining.py` compares one conditioned model with a separate Koopman model
retrained at each of five test parameters, and records the training cost.
`ablation_dictionary.py` compares full conditioning with a fixed dictionary that
uses the same conditioned operator K(theta), at 30 test parameters, and computes the
latent diagnostic. Both are expensive (hours per seed); `--seeds` lets you split the
work across jobs.

**One-shot selection and control (Table III, Fig. 8).**

```
python scripts/control_comparison.py --system vdp --method ours_z
python scripts/control_comparison.py --system vdp --method ours_x
python scripts/control_comparison.py --system vdp --method nmpc
python scripts/control_comparison.py --system vdp --method kmpc_edmd
python scripts/control_comparison.py --system vdp --method kmpc_deep
python scripts/plot_control.py
```

`ours_z` and `ours_x` select theta once with the latent or the state objective and
then run the fixed feedback law. The per-system weights and horizons are in
`pkoopman/selection.py`. The baselines are nonlinear MPC on the true model and
receding-horizon Koopman MPC with an EDMD or a deep Koopman model. The runs are
deterministic, so different seeds only change the measured wall-clock time. NMPC for
the double pendulum is not included.

## Further experiments

These go beyond the tables of the paper and are meant to be run on a cluster.
`python scripts/collect_results.py` gathers whatever has finished into
`runs/summary.md`.

**Region check.** `scripts/region_check.py <system>` draws 1500 new parameters
from the hull of the verified region (with a seed not used in training), checks
the model on each one and reports the survival rate at 10, 15 and 20% with
confidence intervals, together with the fraction of the box Theta the hull covers.
`scripts/train.py --threshold` rebuilds the region with another tolerance.

**Prediction on held-out parameters.** `scripts/prediction_benchmark.py <system>
--thetas fresh` uses new parameters from the region instead of the region samples
the final model was trained on, and `--thetas box` uses parameters from the whole
box, so every method is also tested outside the region chosen for our model. The
two MuJoCo systems are supported too (add `--retrain` to include PKNN and deep
Koopman). For the inverted pendulum these scripts map theta onto the gain box the
model was trained with (`experiment_system` in `pkoopman/systems`).

**Control from many initial states.** `scripts/control_benchmark.py` runs a method
from 50 initial states and records success, settling time, closed-loop cost,
control effort and online time (selection included). Besides the methods of
Table III it has `centroid` (no selection, the mean of the region is deployed) and
`sim_opt` (the same selection problem solved on the true simulator, the reference
for what the surrogate gives up). `--T` and `--kick` give longer episodes with a
mid-episode perturbation.

**On a SLURM cluster** (written for Aalto Triton):

```
bash cluster/setup_env.sh        # once, creates the conda environment
bash cluster/submit_all.sh       # every experiment above, plus the two ablations
```

The job scripts in `cluster/` can also be submitted one by one; each has its usage
at the top. Logs go to `logs/`, results to `runs/`.

## Citation

```
@article{krishnan2026policy,
  title  = {Policy-Conditioned Koopman Representations for Parameterized Closed-Loop Systems},
  author = {Krishnan, Kiran G. and Deka, Shankar A.},
  year   = {2026}
}
```
