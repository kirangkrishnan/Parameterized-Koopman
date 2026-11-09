"""Deep Koopman autoencoder in the style of Lusch et al. with one fixed K.

The model never sees theta. It is trained on the same mixed-theta trajectories
as PKNN, so it has to fit one linear model to the whole closed-loop family.
Column-vector convention z_{k+1} = K z_k.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DeepKoopman(nn.Module):
    def __init__(self, n, dz=10, hidden=64):
        super().__init__()
        self.n, self.dz, self.hidden = n, dz, hidden
        self.encoder = nn.Sequential(nn.Linear(n, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                     nn.Tanh(), nn.Linear(hidden, dz))
        self.decoder = nn.Sequential(nn.Linear(dz, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                     nn.Tanh(), nn.Linear(hidden, n))
        self.K = nn.Parameter(0.9 * torch.eye(dz) + 0.01 * torch.randn(dz, dz))


@torch.no_grad()
def predict(model, x0, steps):
    """x0: (batch, n). Returns predictions of x_1..x_steps, shape (batch, steps, n)."""
    z = model.encoder(x0)
    preds = []
    for _ in range(steps):
        z = torch.mm(z, model.K.T)
        preds.append(model.decoder(z))
    return torch.stack(preds, dim=1)


def loss_fn(model, X):
    z = model.encoder(X[:, 0])
    loss = nn.functional.mse_loss(model.decoder(z), X[:, 0])
    for k in range(1, X.shape[1]):
        z = torch.mm(z, model.K.T)
        loss = loss + 8.0 * nn.functional.mse_loss(model.decoder(z), X[:, k])
        loss = loss + 4.0 * nn.functional.mse_loss(z, model.encoder(X[:, k]))
    return loss + 1e-4 * (model.K ** 2).mean()


def train_deep_koopman(X, epochs=2000, batch_size=256, lr=1e-3, seed=8, **model_kw):
    """X: trajectories (N, H+1, n); the loss rolls out over all H steps."""
    torch.manual_seed(seed)
    model = DeepKoopman(X.shape[-1], **model_kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.5)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for (bx,) in loader:
            opt.zero_grad()
            loss = loss_fn(model, bx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step(total / len(loader))
        if epoch % 500 == 0:
            print(f"  deep koopman epoch {epoch:4d}  loss {total / len(loader):.5f}", flush=True)
    return model.eval()


def save_deep_koopman(model, path):
    config = dict(n=model.n, dz=model.dz, hidden=model.hidden)
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


def load_deep_koopman(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DeepKoopman(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
