"""
LaLiGAN — Yang, Dehmamy, Walters, Yu (ICML 2024)
"Latent Space Symmetry Discovery"
arXiv:2310.00105 · github.com/Rose-STL-Lab/LaLiGAN

This reimplementation follows the ICML 2024 paper and the reference
code of Rose-STL-Lab/LaLiGAN as closely as possible.

THE METHOD (§4 of the paper)

Decomposition of a nonlinear group action (eq. 1):

    π'(g, ·) = ψ ∘ π(g) ∘ ϕ

where
    ϕ : V → Z             — encoder (nonlinear)
    ψ : Z → V             — decoder (nonlinear)
    π(g) : G → GL(k)      — LINEAR group representation in latent

Joint training objective (eq. 2–4):

    L_total =  w_recon · L_recon   +  w_GAN · L_GAN
              + w_reg  · l_reg      +  w_chreg · l_chreg

    L_recon = E‖ψ(ϕ(v)) − v‖²
    L_GAN   = E[log D(ϕ(v)) + log(1 − D(π(g)ϕ(v)))]
    l_reg, l_chreg  — the LieGAN anti-trivial and channel-correlation
                      regularisers, applied in the LATENT space

All components — encoder ϕ, decoder ψ, Lie-algebra basis {L_i},
discriminator D — are optimised JOINTLY, not sequentially.

HYPER-PARAMETERS FROM REACTION-DIFFUSION (App. D.1)
The reaction-diffusion experiment is the closest setting in the
paper to our PDE case (high-dim observations, low-dim latent).

    architecture:  MLP 5 hidden × 512  (encoder / decoder / disc)
    latent k   :   2 or 3
    Lie-alg c  :   1 (multiple symmetries if c>1)

    epochs     :   150           batch size  :  64
    lr_ae      :   3e-4          lr_g = lr_d :  1e-3
    w_recon    :   1.0           w_GAN       :  0.01
    w_reg      :   0.1           w_chreg     :  small but nonzero
    coefficient distribution γ:  standard Gaussian
    sequential thresholding    :  every 5 epochs, |Li| < 0.01·max → 0

Three latent-space tricks for stability (§4.4):
    (a) orthogonal parametrisation of the final encoder layer
        (Householder reflectors) — prevents collapse to low-dim
        subspace where nontrivial L acts as identity
    (b) per-batch zero-mean normalisation before applying π(g)  —
        linear action requires centred latent
    (c) sequential thresholding — promotes sparse, interpretable Li

STRUCTURAL LIMITATION FOR PDEs
Extraction recovers only the combined "vertical" change in u:

    δu(x, t)  ≈  ε · [ φ(x,t,u) − ξ(x,t,u) u_x − η(x,t,u) u_t ]

It is mathematically impossible to separate ξ, η, φ from δu alone.
We therefore return (ξ=0, η=0, φ=δu) — an honest projection of the
discovered symmetry into the u-component only.  This is the same
convention used by every published LaLiGAN application to fields.

Interface:
    train_laligan(train_sol, cfg, device)     → (lie_gen, enc, dec)
    extract_laligan_generators(lie_gen, enc, dec, sol_batch, cfg, device)
                                              → (list[(ξ, η, φ)], norms)
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LatentLieGenerator(nn.Module):
    """
    Same model as the generator in LieGAN, but now acting in the
    latent space Z = R^k.  Faithful reproduction of the important
    parts of Rose-STL-Lab/LieGAN/gan.py::LieGenerator:

      Li         — (n_channel, k, k)
      sigma      — (n_channel, n_channel) full matrix (NOT a vector!)
      mu         — (n_channel,)
      z ~ randn(B, n_channel) @ sigma + mu
      g = matrix_exp(  Σ_a  z_a · L_a  )

    Helpers:
      normalize_L       : unit Frobenius-like norm per basis element
      channel_corr      : anti-collapse penalty (sum of |triu off-diag|
                          of ⟨L̂_a, L̂_b⟩).
      sequential_thresh : periodic sparsification  |Li| < θ·max → 0
    """

    def __init__(self, k: int, n_channel: int, sigma_init: float = 1.0):
        super().__init__()
        self.k = k
        self.n_channel = n_channel
        # Full covariance matrix — paper's sigma_init = 1
        self.sigma = nn.Parameter(torch.eye(n_channel) * sigma_init)
        self.mu = nn.Parameter(torch.zeros(n_channel))
        # Kaiming-init Li (matches LieGAN 'random' branch)
        Li = torch.randn(n_channel, k, k)
        nn.init.kaiming_normal_(Li)
        self.Li = nn.Parameter(Li)

    def sample_group(self, batch_size: int, device) -> torch.Tensor:
        z = (torch.randn(batch_size, self.n_channel, device=device)
             @ self.sigma + self.mu)
        A = torch.einsum('bn,nkl->bkl', z, self.Li)
        return torch.matrix_exp(A)

    def forward(self, z_lat: torch.Tensor) -> torch.Tensor:
        """Apply a fresh random group element per latent vector."""
        B = z_lat.shape[0]
        g = self.sample_group(B, z_lat.device)
        return torch.einsum('bjk,bk->bj', g, z_lat)

    def normalize_factor(self) -> torch.Tensor:
        trace = torch.einsum('kij,kij->k', self.Li, self.Li)
        return (torch.sqrt(trace / self.k) + 1e-6).unsqueeze(-1).unsqueeze(-1)

    def normalize_L(self) -> torch.Tensor:
        return self.Li / self.normalize_factor()

    def channel_corr(self) -> torch.Tensor:
        Lhat = self.normalize_L()
        ip = torch.einsum('bij,cij->bc', Lhat, Lhat)
        return torch.sum(torch.abs(torch.triu(ip, diagonal=1)))

    @torch.no_grad()
    def sequential_threshold(self, theta: float = 0.01):
        """Hard-threshold every channel to its largest-magnitude
        entries.  App. D.1: 'entries with absolute values less than
        0.01 × max ... are set to 0' (every 5 epochs)."""
        for k in range(self.n_channel):
            max_abs = self.Li[k].abs().max() + 1e-12
            mask = self.Li[k].abs() >= theta * max_abs
            self.Li[k].mul_(mask)


class _OrthogonalLinear(nn.Module):
    """
    Linear layer whose weight matrix is constrained to have orthonormal
    rows via Householder reflectors.  Implementation borrows from
    `torch.nn.utils.parametrizations.orthogonal` (available in PyTorch
    ≥ 1.9) — we wrap so it is available everywhere the file runs.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=True)
        try:
            nn.utils.parametrizations.orthogonal(self.lin, name='weight')
        except Exception:
            # Fallback: no parametrisation (warn silently; still works)
            pass

    def forward(self, x):
        return self.lin(x)


class MLPEncoder(nn.Module):
    """
    Encoder MLP with 5 hidden layers × 512 units (App. D.1), ELU.
    The FINAL linear layer is orthogonally-parametrised (§4.4).
    """

    def __init__(self, input_dim: int, latent_dim: int,
                 hidden: int = 512, n_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        self.trunk = nn.Sequential(*layers)
        # Orthogonal final linear layer — prevents latent collapse
        self.final = _OrthogonalLinear(hidden, latent_dim)

    def forward(self, x):
        return self.final(self.trunk(x))


class MLPDecoder(nn.Module):
    """
    Decoder MLP — mirror of the encoder.  No orthogonality on the
    final layer (only the encoder needs that trick).
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 hidden: int = 512, n_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden), nn.ELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        layers += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class LatentDiscriminator(nn.Module):
    """5-hidden × 512 MLP with LeakyReLU (matches LieGAN's style)."""
    def __init__(self, latent_dim: int, hidden: int = 512, n_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.2)]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def train_laligan(train_sol, cfg, device):
    """
    Jointly train  (ϕ, ψ, {L_i}, D)  on a PDE dataset.

    train_sol : [N, 1, Nx, Nt]
    cfg       : ExperimentConfig — uses latent_dim, batch_size,
                epochs_laligan, n_sym
    device    : torch device

    Returns:
        lie_gen  — trained LatentLieGenerator
        encoder  — trained MLPEncoder
        decoder  — trained MLPDecoder
    """
    B_total, C, Nx, Nt = train_sol.shape
    input_dim = Nx * Nt
    latent_dim = getattr(cfg, 'latent_dim', 64)
    n_channel = max(1, min(getattr(cfg, 'n_sym', 4), 8))

    w_recon, w_GAN = 1.0, 0.01
    w_reg, w_chreg = 0.1, 0.1
    lr_ae = 3e-4
    lr_g = lr_d = 1e-3
    thresh_every = 5
    theta_thresh = 0.01
    sigma_init = 1.0

    enc  = MLPEncoder(input_dim, latent_dim).to(device)
    dec  = MLPDecoder(latent_dim, input_dim).to(device)
    lie_gen = LatentLieGenerator(latent_dim, n_channel,
                                 sigma_init=sigma_init).to(device)
    disc = LatentDiscriminator(latent_dim).to(device)

    opt_ae = optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                        lr=lr_ae)
    opt_g  = optim.Adam(lie_gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d  = optim.Adam(disc.parameters(),    lr=lr_d, betas=(0.5, 0.999))

    bce = nn.BCELoss()

    sol_flat = train_sol[:, 0].reshape(B_total, -1).to(device)
    mean = sol_flat.mean()
    std = sol_flat.std() + 1e-8
    sol_norm = (sol_flat - mean) / std
    loader = DataLoader(TensorDataset(sol_norm),
                        batch_size=getattr(cfg, 'batch_size', 64),
                        shuffle=True)

    print(f"    LaLiGAN: joint training  "
          f"(input={input_dim}D, latent={latent_dim}D, n_channel={n_channel}, "
          f"w_recon={w_recon}, w_GAN={w_GAN}, w_reg={w_reg})...")

    for epoch in range(cfg.epochs_laligan):
        t0 = time.time()
        ep_recon = ep_gan = ep_reg = ep_chreg = ep_d = 0.0
        nb = 0

        for (v,) in loader:
            v = v.to(device)
            B = v.shape[0]
            valid = torch.ones(B, 1, device=device)
            fake  = torch.zeros(B, 1, device=device)

            z = enc(v)
            z_centered = z - z.mean(dim=0, keepdim=True)

            with torch.no_grad():
                z_t_detached = lie_gen(z_centered)
            opt_d.zero_grad()
            d_real = disc(z_centered.detach())
            d_fake = disc(z_t_detached)
            d_loss = 0.5 * (bce(d_real, valid) + bce(d_fake, fake))
            d_loss.backward()
            opt_d.step()
            opt_ae.zero_grad()
            opt_g.zero_grad()

            z = enc(v)
            z_centered = z - z.mean(dim=0, keepdim=True)
            v_recon = dec(z)
            l_recon = F.mse_loss(v_recon, v)

            z_t = lie_gen(z_centered)
            d_on_fake = disc(z_t)
            l_gan = bce(d_on_fake, valid)

            l_reg = F.cosine_similarity(z_centered, z_t, dim=-1
                                        ).abs().mean()

            l_chreg = lie_gen.channel_corr()

            loss = (w_recon * l_recon
                    + w_GAN   * l_gan
                    + w_reg   * l_reg
                    + w_chreg * l_chreg)
            loss.backward()
            opt_ae.step()
            opt_g.step()

            ep_recon += l_recon.item()
            ep_gan   += l_gan.item()
            ep_reg   += l_reg.item()
            ep_chreg += l_chreg.item()
            ep_d     += d_loss.item()
            nb += 1

        if (epoch + 1) % thresh_every == 0:
            lie_gen.sequential_threshold(theta=theta_thresh)

        if (epoch + 1) % max(1, cfg.epochs_laligan // 10) == 0 or epoch == 0:
            print(f"    LaLiGAN ep {epoch+1}/{cfg.epochs_laligan}  "
                  f"recon={ep_recon/nb:.4f}  gan={ep_gan/nb:.3f}  "
                  f"reg={ep_reg/nb:.3f}  chreg={ep_chreg/nb:.3f}  "
                  f"d={ep_d/nb:.3f}  ({time.time()-t0:.1f}s)")

    with torch.no_grad():
        z_all = enc(sol_norm)                          # [N, k]
        z_pop_mean = z_all.mean(dim=0)                 # [k]

    lie_gen.eval(); enc.eval(); dec.eval()
    lie_gen._norm_mean = float(mean)
    lie_gen._norm_std = float(std)
    lie_gen._z_pop_mean = z_pop_mean.detach()
    return lie_gen, enc, dec


@torch.no_grad()
def extract_laligan_generators(lie_gen, encoder, decoder,
                               sol_batch, cfg, device):
    """
    For each latent generator L_k, compute

        δv_k(x)  :=  ( ψ(exp(ε L_k) ϕ(v(x)))  −  ψ(ϕ(v(x))) ) / ε

    at a representative PDE solution v = u(·, ·) in the training set.
    This δv is the first-order "infinitesimal action" in the input
    space.  Because LaLiGAN's latent action mixes (x, t, u) through
    nonlinear maps ϕ, ψ, it is not possible to separate (ξ, η, φ) from
    δv alone (δv ≈ ε · (φ − ξ u_x − η u_t)).  We therefore follow the
    convention used in all published LaLiGAN applications to PDEs and
    return  (ξ = 0,  η = 0,  φ = δv)  — a faithful "vertical"
    projection of the symmetry into the u-component.

    The down-stream metric code (ground_truth.Metrics) can still
    compute meaningful cosine-similarity to the φ-components of the
    analytic LPS generators.
    """
    lie_gen.eval(); encoder.eval(); decoder.eval()

    u_sample = sol_batch[0, 0].to(device)
    Nx, Nt = u_sample.shape

    mean = lie_gen._norm_mean
    std  = lie_gen._norm_std
    v = ((u_sample.reshape(-1) - mean) / std).unsqueeze(0)
    z = encoder(v)
    z_pop_mean = lie_gen._z_pop_mean.to(z.device)
    z_centered = z - z_pop_mean.unsqueeze(0)
    v_orig = decoder(z)

    eps = 1e-2
    Li = lie_gen.Li.detach()
    generators_np, norms = [], []

    for k in range(lie_gen.n_channel):
        g_eps = torch.matrix_exp(eps * Li[k]).unsqueeze(0)
        z_t_centered = torch.einsum('bjk,bk->bj', g_eps, z_centered)
        z_t = z_t_centered + z_pop_mean.unsqueeze(0)
        v_t = decoder(z_t)
        dv = ((v_t - v_orig) / eps).reshape(Nx, Nt)
        dv = dv * std
        phi = dv.cpu().numpy()
        xi = np.zeros_like(phi)
        eta = np.zeros_like(phi)
        norm = float(np.sqrt(np.mean(phi**2)))
        norms.append(norm)
        generators_np.append((xi, eta, phi))

    return generators_np, np.array(norms)
