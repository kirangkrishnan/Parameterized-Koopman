"""Parametric Koopman neural network (PKNN), following the code of Guo et al.

The lifted state is Psi(x) = [1, x, g(x)] with a learned dictionary g, and a
second network maps theta to the Koopman matrix K(theta). There is no decoder:
the state is read off the lifted vector directly. The model is trained on
one-step pairs with the loss || Psi(x_{k+1}) - Psi(x_k) K(theta) ||^2.
Row-vector convention as in Guo et al.'s implementation, Psi_next = Psi K.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PKNN(nn.Module):
    def __init__(self, n, p, n_psi=10, hidden=32, layers=2):
        super().__init__()
        self.n, self.p, self.n_psi = n, p, n_psi
        self.hidden, self.layers = hidden, layers

        # dictionary: bias-free input layer, residual tanh layers, linear output
        self.dict_input = nn.Linear(n, hidden, bias=False)
        self.dict_hidden = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.dict_output = nn.Linear(hidden, n_psi - n - 1)

        # theta -> the free entries of K; the first column is fixed to e_1 so
        # that the constant observable stays constant
        k_layers, n_in = [], p
        for _ in range(layers):
            k_layers += [nn.Linear(n_in, hidden), nn.Tanh()]
            n_in = hidden
        self.k_hidden = nn.Sequential(*k_layers)
        self.k_output = nn.Linear(n_in, n_psi * (n_psi - 1))

    def lift(self, x):
        h = self.dict_input(x)
        for layer in self.dict_hidden:
            h = h + torch.tanh(layer(h))
        ones = torch.ones((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
        return torch.cat([ones, x, self.dict_output(h)], dim=-1)

    def K(self, theta):
        free = self.k_output(self.k_hidden(theta)).view(-1, self.n_psi, self.n_psi - 1)
        first = torch.zeros((theta.shape[0], self.n_psi, 1), dtype=theta.dtype, device=theta.device)
        first[:, 0, 0] = 1.0
        return torch.cat([first, free], dim=2)

    @staticmethod
    def step(psi, K):
        return torch.bmm(psi.unsqueeze(1), K).squeeze(1)

    def state(self, psi):
        return psi[..., 1:1 + self.n]


@torch.no_grad()
def predict_pure(model, x0, theta, steps):
    """Lift x0 once and propagate the whole lifted vector with K(theta).
    This is the rollout reported as PKNN in the paper. Returns (batch, steps, n)."""
    K = model.K(theta)
    psi = model.lift(x0)
    preds = []
    for _ in range(steps):
        psi = model.step(psi, K)
        preds.append(model.state(psi))
    return torch.stack(preds, dim=1)


@torch.no_grad()
def predict_relift(model, x0, theta, steps):
    """Re-lift the predicted state before every step (the rollout used in Guo et al.'s code)."""
    K = model.K(theta)
    x = x0
    preds = []
    for _ in range(steps):
        x = model.state(model.step(model.lift(x), K))
        preds.append(x)
    return torch.stack(preds, dim=1)


def train_pknn(X, theta, epochs=500, batch_size=200, lr=1e-3, seed=8, **model_kw):
    """X: trajectories (N, H+1, n), theta: (N, p). Every consecutive pair is one sample."""
    torch.manual_seed(seed)
    n, p = X.shape[-1], theta.shape[-1]
    model = PKNN(n, p, **model_kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.1, min_lr=1e-10)

    H = X.shape[1] - 1
    x_now = X[:, :-1].reshape(-1, n)
    x_next = X[:, 1:].reshape(-1, n)
    th = theta[:, None, :].expand(-1, H, -1).reshape(-1, p)
    loader = DataLoader(TensorDataset(x_now, x_next, th), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        total = 0.0
        for bx, by, bt in loader:
            opt.zero_grad()
            psi_x, psi_y = model.lift(bx), model.lift(by)
            loss = nn.functional.mse_loss(model.step(psi_x, model.K(bt)), psi_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step(total / len(loader))
        if epoch == 1 or epoch % 100 == 0:
            print(f"  pknn epoch {epoch:4d}  loss {total / len(loader):.6f}", flush=True)
    return model.eval()


def save_pknn(model, path):
    config = dict(n=model.n, p=model.p, n_psi=model.n_psi, hidden=model.hidden, layers=model.layers)
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


def load_pknn(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PKNN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
