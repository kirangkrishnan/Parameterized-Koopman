import torch
import torch.nn as nn


def mlp(n_in, n_out, hidden, act):
    layers = []
    for h in hidden:
        layers += [nn.Linear(n_in, h), act()]
        n_in = h
    layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


class PolicyKoopman(nn.Module):
    """Koopman model whose coordinates and operator both depend on theta.

    encoder  phi(x, theta)  : R^(n+p)  -> R^dz
    decoder  psi(z, theta)  : R^(dz+p) -> R^n
    operator K(theta)       : R^p      -> R^(dz x dz)

    The operator network outputs two blocks, K_fwd (used for prediction) and
    K_bwd (only used during training, for the backward rollout and the
    consistency term K_fwd K_bwd = I).

    conditioning:
      "full"     - encoder, decoder and operator see theta (the method in the paper)
      "operator" - only the operator sees theta, fixed dictionary (ablation B)
    """

    def __init__(self, n, p, dz, hidden=(32,), conditioning="full", small_init=False,
                 theta_scale=None):
        super().__init__()
        self.n, self.p, self.dz = n, p, dz
        self.hidden = tuple(hidden)
        self.conditioning = conditioning

        # theta is divided by this before it enters any network (only the
        # manipulator uses it, its gains have very different ranges)
        scale = torch.ones(p) if theta_scale is None else torch.as_tensor(theta_scale, dtype=torch.float32)
        self.register_buffer("theta_scale", scale)

        extra = p if conditioning == "full" else 0
        self.encoder = mlp(n + extra, dz, hidden, nn.Tanh)
        self.decoder = mlp(dz + extra, n, hidden, nn.Tanh)
        self.operator = mlp(p, 2 * dz * dz, hidden, nn.ReLU)

        if small_init:
            # start with K close to zero, helps on the cartpole runs
            self.operator[-1].weight.data.uniform_(-0.01, 0.01)
            self.operator[-1].bias.data.uniform_(-0.01, 0.01)

    def _net_input(self, v, theta):
        if self.conditioning == "full":
            return torch.cat([v, theta / self.theta_scale], dim=-1)
        return v

    # phi(0, theta) = 0 and psi(0, theta) = 0 by construction, so the
    # equilibrium at the origin is kept for every theta
    def encode(self, x, theta):
        zero = torch.zeros_like(x)
        return self.encoder(self._net_input(x, theta)) - self.encoder(self._net_input(zero, theta))

    def decode(self, z, theta):
        zero = torch.zeros_like(z)
        return self.decoder(self._net_input(z, theta)) - self.decoder(self._net_input(zero, theta))

    def operators(self, theta):
        out = self.operator(theta / self.theta_scale)
        d2 = self.dz * self.dz
        K_fwd = out[..., :d2].reshape(-1, self.dz, self.dz)
        K_bwd = out[..., d2:].reshape(-1, self.dz, self.dz)
        return K_fwd, K_bwd

    def K(self, theta):
        return self.operators(theta)[0]

    def predict(self, x0, theta, steps):
        """Roll the latent state forward. Returns x_hat of shape (batch, steps, n),
        where x_hat[:, k-1] is the prediction of x_k."""
        K = self.K(theta)
        z = self.encode(x0, theta)
        preds = []
        for _ in range(steps):
            z = torch.bmm(K, z.unsqueeze(-1)).squeeze(-1)
            preds.append(self.decode(z, theta))
        return torch.stack(preds, dim=1)

    def latent_rollout(self, x0, theta, steps):
        """Same as predict but returns the latent states z_1..z_steps."""
        K = self.K(theta)
        z = self.encode(x0, theta)
        zs = []
        for _ in range(steps):
            z = torch.bmm(K, z.unsqueeze(-1)).squeeze(-1)
            zs.append(z)
        return torch.stack(zs, dim=1)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PolicyKoopman(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def save_model(model, path, **extra):
    config = dict(n=model.n, p=model.p, dz=model.dz, hidden=model.hidden,
                  conditioning=model.conditioning)
    torch.save({"config": config, "state_dict": model.state_dict(), **extra}, path)
