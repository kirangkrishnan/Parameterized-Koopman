import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# default loss weights (rec, pred, lin, cons, reg); systems can override them
DEFAULT_WEIGHTS = dict(rec=1.0, pred=2.0, lin=1.0, cons=0.1, reg=1e-4)


def rollout_losses(model, X, theta, K):
    """Prediction and linearity losses along one direction.

    X has shape (batch, H+1, n) and is ordered in the direction of the rollout,
    so the same function is used forward (K_fwd, X) and backward (K_bwd, X flipped).
    """
    H = X.shape[1] - 1
    z = model.encode(X[:, 0], theta)
    pred, lin = 0.0, 0.0
    for k in range(1, H + 1):
        z = torch.bmm(K, z.unsqueeze(-1)).squeeze(-1)
        pred = pred + F.mse_loss(model.decode(z, theta), X[:, k])
        lin = lin + F.mse_loss(z, model.encode(X[:, k], theta))
    return pred / H, lin / H


def koopman_loss(model, X, theta, w):
    K_fwd, K_bwd = model.operators(theta)
    x0 = X[:, 0]

    rec = F.mse_loss(model.decode(model.encode(x0, theta), theta), x0)

    pred_f, lin_f = rollout_losses(model, X, theta, K_fwd)
    pred_b, lin_b = rollout_losses(model, X.flip(1), theta, K_bwd)

    eye = torch.eye(model.dz, device=X.device).expand_as(K_fwd)
    cons = F.mse_loss(torch.bmm(K_fwd, K_bwd), eye)
    reg = (K_fwd ** 2).sum(dim=(1, 2)).mean() + (K_bwd ** 2).sum(dim=(1, 2)).mean()

    total = (w["rec"] * rec
             + w["pred"] * (pred_f + pred_b)
             + w["lin"] * (lin_f + lin_b)
             + w["cons"] * cons
             + w["reg"] * reg)
    parts = dict(rec=rec.item(), pred=0.5 * (pred_f + pred_b).item(),
                 lin=0.5 * (lin_f + lin_b).item(), cons=cons.item())
    return total, parts


def train(model, X, theta, epochs=3000, batch_size=128, lr=1e-3, weights=None,
          clip=None, log_every=50, device=None):
    """Train on trajectories X (N, H+1, n) with parameters theta (N, p)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))

    model.to(device)
    loader = DataLoader(TensorDataset(X.to(device), theta.to(device)),
                        batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=100)

    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        sums = dict(total=0.0, rec=0.0, pred=0.0, lin=0.0, cons=0.0)
        for xb, tb in loader:
            opt.zero_grad()
            loss, parts = koopman_loss(model, xb, tb, w)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            sums["total"] += loss.item()
            for key in parts:
                sums[key] += parts[key]

        avg = {k: v / len(loader) for k, v in sums.items()}
        avg["lr"] = opt.param_groups[0]["lr"]
        sched.step(avg["total"])
        history.append(avg)

        if epoch % log_every == 0:
            print(f"epoch {epoch:5d}  lr {avg['lr']:.1e}  loss {avg['total']:.5f}  "
                  f"pred {avg['pred']:.5f}  lin {avg['lin']:.5f}  ({time.time() - t0:.0f}s)",
                  flush=True)

    model.eval()
    return model, history
