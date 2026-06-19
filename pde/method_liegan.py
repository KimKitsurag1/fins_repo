"""
LieGAN — Yang, Walters, Dehmamy, Yu (ICML 2023)
"Generative Adversarial Symmetry Discovery"
arXiv:2302.00236 · github.com/Rose-STL-Lab/LieGAN

This file is a reverse-engineered reimplementation aligned with the
official code of the Rose-STL-Lab repository (gan.py, train.py).
Everything follows the paper / repo unless explicitly marked as an
adaptation for the PDE domain.

STRUCTURE OF LIEGAN (verbatim from gan.py)

    Li     : nn.Parameter(n_channel, n_dim, n_dim)       — algebra basis
    sigma  : nn.Parameter(n_channel, n_channel)          — covariance
                                                           for z sampling
    mu     : nn.Parameter(n_channel)                     — mean for z
    mask   : zero-out last row when g_init == 'affine'   — homogeneous
                                                           coord trick

    z   ~  N(mu, sigma²)           (sigma is a full matrix)
    g   =  matrix_exp( Σ_k z_k · (L_k · mask) )
    x̃  =  g · x                   (batched, per-sample g)

GENERATOR TOTAL LOSS (verbatim from train.py)

    g_loss =  BCE(D(x̃, ỹ), valid)                   adversarial
            + lamda  * reg_anti_trivial            anti-identity
            + mu     * ||Li||_p                    sparsity
            + eta    * channel_corr                anti-collapse
                                                   (off-diag ⟨L_a, L_b⟩)

    reg ∈ {'cosine', 'rel_diff', 'Li_norm', 'fourier'}
    channel_corr normalises Li to unit  ||L_k||² = n_dim  first.

    Hyper-parameters for top-tagging (closest setting to PDE):
        --lamda 1  --eta 0.1  --sigma_init 1
    Hyper-parameters for 2-body trajectory:
        --lamda 1  --sigma_init 1  --num_epochs 100

ADAPTATION FOR PDEs (X × U space)

LieGAN is structurally **affine-only**.  On the (x, t, u) product
space  X × U ⊆ ℝ³  of a 1-D evolution PDE, we represent the action
using homogeneous coordinates:

      point      (x, t, u, 1)   ∈ ℝ⁴
      algebra    L_k ∈ ℝ^{4×4}, last row zeroed by mask
      group      g = exp(Σ z_k L_k · mask) ∈ Aff(3)

This is the paper's 'affine' g_init applied to 3-D space.  Each
"sample" is a point-cloud {(x_i, t_j, u_{ij})} from one PDE solution
(sub-sampled for memory — GAN on 64×128 points per instance is
impractical).  The discriminator sees a flattened point-cloud and
returns the real/fake probability.

Expected recoveries:
    Heat    (6 GT):  ∂_x, ∂_t, u∂_u, x∂_x+2t∂_t       → 4 / 6
                     (Galilean 2t∂_x + xu∂_u and projective contain
                      the non-affine term x·u → not representable.)
    Burgers (5 GT):  ∂_x, ∂_t, t∂_x+∂_u, x∂_x+2t∂_t−u∂_u → 4 / 5
    KdV     (4 GT):  all four (all affine)             → 4 / 4

Interface kept compatible with the rest of the pipeline:
    train_liegan(train_sol, pde_type, x_grid, t_grid, cfg, device)
    extract_liegan_generators(gen, sol_batch, x_grid, t_grid, cfg, device)

The first function returns the trained LieGenerator (not a tuple as
before — the discriminator is internal).  The second returns
    generators_np : list of (ξ, η, φ) fields, each [Nx, Nt]
    norms         : np.ndarray of generator L² norms
matching the convention of all other baselines.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


#  1.  LieGenerator — verbatim structure from Rose-STL-Lab/LieGAN/gan.py
class LieGenerator(nn.Module):
    """
    Infinitesimal generators L_k together with a Gaussian sampler over
    their coefficients.  Following the original code we expose:

        self.Li        : (n_channel, n_dim, n_dim)
        self.sigma     : (n_channel, n_channel)   — full covariance
        self.mu        : (n_channel,)
        self.mask      : (n_dim, n_dim)   zero-out last row in affine mode

        sample_coefficient(B) :  z ~ randn @ sigma + mu
        sample_group(B)       :  g = exp(Σ z_k · L_k · mask)
        forward(x)            :  g·x via einsum('bjk,btk->btj', g, x)

        normalize_factor()    :  sqrt(⟨L_k, L_k⟩ / n_dim)  per channel
        channel_corr()        :  sum|triu off-diagonal ⟨L_k, L_l⟩|
                                 on normalised L (eq. 6 in the paper)
    """

    def __init__(self, n_dim: int = 4, n_channel: int = 4,
                 sigma_init: float = 1.0, affine: bool = True,
                 normalize_Li: bool = False):
        super().__init__()
        self.n_dim = n_dim
        self.n_channel = n_channel
        self.normalize_Li = normalize_Li

        self.sigma = nn.Parameter(torch.eye(n_channel) * sigma_init)
        self.mu = nn.Parameter(torch.zeros(n_channel))

        Li = torch.randn(n_channel, n_dim, n_dim)
        nn.init.kaiming_normal_(Li)
        self.Li = nn.Parameter(Li)

        mask = torch.ones(n_dim, n_dim)
        if affine:
            mask[-1, :] = 0
        self.register_buffer('mask', mask)

    def getLi(self) -> torch.Tensor:
        """Effective (masked) Lie algebra basis."""
        return self.Li * self.mask

    def normalize_factor(self) -> torch.Tensor:
        trace = torch.einsum('kdf,kdf->k', self.Li, self.Li)
        return (torch.sqrt(trace / self.n_dim) + 1e-6).unsqueeze(-1).unsqueeze(-1)

    def normalize_L(self) -> torch.Tensor:
        return self.Li / self.normalize_factor()

    def channel_corr(self, killing: bool = False) -> torch.Tensor:
        """
        Channel-correlation regulariser (eq. 6 in the paper).
        Normalise every basis element to unit Frobenius-like norm, then
        sum absolute values of the upper-triangular off-diagonal inner
        products ⟨L_a, L_b⟩.  Encourages linear independence of bases.
        """
        Li = self.normalize_L()
        if not killing:
            ip = torch.einsum('bij,cij->bc', Li, Li)
            return torch.sum(torch.abs(torch.triu(ip, diagonal=1)))
        else:
            trxy = torch.triu(torch.einsum('bij,cji->bc', Li, Li),
                              diagonal=1)
            trx = torch.einsum('kdd->k', Li)
            trx_try = torch.triu(torch.einsum('b,c->bc', trx, trx),
                                 diagonal=1)
            return torch.sum(
                torch.abs(trxy - trx_try / self.n_dim))

    def sample_coefficient(self, batch_size: int, device) -> torch.Tensor:
        return (torch.randn(batch_size, self.n_channel, device=device)
                @ self.sigma + self.mu)

    def sample_group(self, batch_size: int, device) -> torch.Tensor:
        z = self.sample_coefficient(batch_size, device)
        Li_eff = (self.normalize_L() if self.normalize_Li else self.Li)
        A = torch.einsum('bj,jkl->bkl', z, Li_eff * self.mask)
        return torch.matrix_exp(A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, n_dim] — B point-clouds of T homogeneous points.
        Returns x̃ of identical shape, with each cloud transformed by
        its own random group element.
        """
        B = x.shape[0]
        g = self.sample_group(B, x.device)
        return torch.einsum('bjk,btk->btj', g, x)


class LieDiscriminator(nn.Module):
    """
    Standard MLP taking the concatenation of x (transformed) and y
    (scalar label / unused).  We follow the paper's LieDiscriminator
    with 512-512 hidden and sigmoid output.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1), nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x_flat = x.reshape(x.shape[0], -1)
        if y is not None:
            y_flat = y.reshape(y.shape[0], -1)
            xy = torch.cat([x_flat, y_flat], dim=1)
        else:
            xy = x_flat
        return self.model(xy)


def _pde_point_cloud(sol: torch.Tensor, x_grid: np.ndarray,
                     t_grid: np.ndarray, n_points: int,
                     normalize: bool = True
                     ) -> tuple[torch.Tensor, dict]:
    """
    Turn a batch of PDE solutions into homogeneous point clouds.

    sol        : [B, Nx, Nt] PDE values
    x_grid     : [Nx] physical x coords
    t_grid     : [Nt] physical t coords
    n_points   : how many random (i, j) points to draw per instance
    normalize  : if True, rescale (x, t, u) each to zero-mean unit-std
                 — GANs are very sensitive to feature scale.

    Returns
    -------
    pts        : [B, n_points, 4]  homogeneous points (x̂, t̂, û, 1)
    stats      : {'x_mean', 'x_std', 't_mean', 't_std',
                  'u_mean', 'u_std'} — needed to map Li back to
                 physical coordinates at extraction time.
    """
    device = sol.device
    B, Nx, Nt = sol.shape

    xg = torch.tensor(x_grid, dtype=torch.float32, device=device)
    tg = torch.tensor(t_grid, dtype=torch.float32, device=device)

    ii = torch.randint(0, Nx, (B, n_points), device=device)
    jj = torch.randint(0, Nt, (B, n_points), device=device)

    x_vals = xg[ii]                                  # [B, K]
    t_vals = tg[jj]                                  # [B, K]
    u_vals = sol[torch.arange(B, device=device).unsqueeze(1), ii, jj]

    if normalize:
        x_m, x_s = xg.mean().item(), xg.std().item() + 1e-8
        t_m, t_s = tg.mean().item(), tg.std().item() + 1e-8
        u_m = float(sol.mean())
        u_s = float(sol.std()) + 1e-8
        x_n = (x_vals - x_m) / x_s
        t_n = (t_vals - t_m) / t_s
        u_n = (u_vals - u_m) / u_s
    else:
        x_m = t_m = u_m = 0.0
        x_s = t_s = u_s = 1.0
        x_n, t_n, u_n = x_vals, t_vals, u_vals

    ones = torch.ones_like(x_n)
    pts = torch.stack([x_n, t_n, u_n, ones], dim=-1)     # [B, K, 4]
    stats = dict(x_mean=x_m, x_std=x_s,
                 t_mean=t_m, t_std=t_s,
                 u_mean=u_m, u_std=u_s)
    return pts, stats


def train_liegan(train_sol, pde_type, x_grid, t_grid, cfg, device):
    """
    Train LieGAN on a PDE dataset.

    Parameters
    ----------
    train_sol : torch.Tensor [N, 1, Nx, Nt] — N solutions on a Nx×Nt grid
    pde_type  : str (unused here — adversarial objective is PDE-agnostic)
    x_grid    : np.ndarray [Nx]
    t_grid    : np.ndarray [Nt]
    cfg       : ExperimentConfig (uses batch_size, epochs_liegan,
                lr_liegan, n_sym)
    device    : torch device

    Returns
    -------
    generator : trained LieGenerator. Carries stats (._data_stats) so
                extract_liegan_generators can map Li back to physical
                coordinates.
    """
    B_total, C, Nx, Nt = train_sol.shape
    sol = train_sol[:, 0].to(device)

    n_channel = min(getattr(cfg, 'n_sym', 4), 6)
    n_dim = 4                       # (x, t, u, 1) homogeneous
    sigma_init = 1.0
    lamda = 1.0                     # anti-trivial weight
    mu_sparse = 0.05                # sparsity weight (paper: 0 or small)
    eta = 0.1                       # channel-corr weight
    p_norm = 2                      # L2 for sparsity reg
    lr_g = getattr(cfg, 'lr_liegan', 2e-4)
    lr_d = lr_g                     # paper uses same lr for G and D
    n_points = min(256, Nx * Nt)    # sub-sampled point-cloud size
    reg_type = 'cosine'             # paper's choice for vector data

    gen = LieGenerator(n_dim=n_dim, n_channel=n_channel,
                       sigma_init=sigma_init, affine=True).to(device)
    disc = LieDiscriminator(input_size=n_points * n_dim).to(device)

    opt_g = optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCELoss(reduction='mean')

    bs = getattr(cfg, 'batch_size', 4)
    idx_loader = DataLoader(TensorDataset(torch.arange(B_total)),
                            batch_size=bs, shuffle=True)

    print(f"    LieGAN: training in homogeneous ℝ⁴  "
          f"(n_channel={n_channel}, λ={lamda}, η={eta}, μ_sp={mu_sparse},"
          f" σ_init={sigma_init}, K={n_points})...")

    _, stats = _pde_point_cloud(sol, x_grid, t_grid, n_points=1,
                                 normalize=True)

    for epoch in range(cfg.epochs_liegan):
        t0 = time.time()
        d_real_acc = d_fake_acc = g_adv_acc = 0.0
        reg_acc = sp_acc = chreg_acc = 0.0
        nb = 0

        for (batch_idx,) in idx_loader:
            sol_batch = sol[batch_idx]
            pts_real, _ = _pde_point_cloud(sol_batch, x_grid, t_grid,
                                           n_points=n_points,
                                           normalize=True)
            valid = torch.ones(pts_real.shape[0], 1, device=device)
            fake  = torch.zeros(pts_real.shape[0], 1, device=device)

            opt_g.zero_grad()
            pts_fake = gen(pts_real)

            g_adv = bce(disc(pts_fake), valid)

            if reg_type == 'cosine':
                g_reg = torch.abs(
                    F.cosine_similarity(pts_fake, pts_real, dim=-1).mean())
            elif reg_type == 'rel_diff':
                g_reg = -torch.minimum(
                    torch.abs((pts_fake - pts_real)
                              / (pts_real.abs() + 1e-6)).mean(),
                    torch.tensor(1.0, device=device))
            elif reg_type == 'Li_norm':
                g_reg = -torch.minimum(
                    torch.norm(gen.getLi(), p=2),
                    torch.tensor(float(n_dim * n_channel), device=device))
            else:
                raise ValueError(reg_type)
            g_reg = lamda * g_reg

            g_sp = mu_sparse * torch.norm(gen.getLi(), p=p_norm)

            g_chreg = eta * gen.channel_corr(killing=False)

            g_loss = g_adv + g_reg + g_sp + g_chreg
            g_loss.backward()
            opt_g.step()

            opt_d.zero_grad()
            pts_fake2 = gen(pts_real).detach()
            real_loss = bce(disc(pts_real), valid)
            fake_loss = bce(disc(pts_fake2), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            opt_d.step()

            d_real_acc += real_loss.item()
            d_fake_acc += fake_loss.item()
            g_adv_acc  += g_adv.item()
            reg_acc    += g_reg.item() / max(lamda, 1e-6)
            sp_acc     += g_sp.item() / max(mu_sparse, 1e-6)
            chreg_acc  += g_chreg.item() / max(eta, 1e-6)
            nb += 1

        if (epoch + 1) % max(1, cfg.epochs_liegan // 10) == 0 or epoch == 0:
            print(f"    LieGAN ep {epoch + 1}/{cfg.epochs_liegan}  "
                  f"D_r={d_real_acc/nb:.3f} D_f={d_fake_acc/nb:.3f} "
                  f"G_adv={g_adv_acc/nb:.3f} reg={reg_acc/nb:.3f} "
                  f"sp={sp_acc/nb:.3f} chreg={chreg_acc/nb:.3f} "
                  f"({time.time()-t0:.1f}s)")

    gen.eval()
    gen._data_stats = stats
    return gen


@torch.no_grad()
def extract_liegan_generators(gen: LieGenerator, sol_batch,
                              x_grid, t_grid, cfg, device):
    """
    Convert each learned Li (acting on homogeneous (x̂, t̂, û, 1))
    into a vector-field (ξ, η, φ) defined on the _physical_ grid of
    sol_batch[0].  Output matches the format used by every other
    baseline in the pipeline.

    For an affine generator  L_k ∈ ℝ^{4×4}  (last row = 0 by mask),
    the associated infinitesimal vector field is

          v_k(p̂) = L_k · p̂ ,       p̂ = (x̂, t̂, û, 1)ᵀ.

    We then undo the per-axis standardisation so that (ξ, η, φ) are
    expressed in the same units as the analytic ground-truth fields
    produced by ground_truth.py.

    Chain rule, since data was standardised (x̂ = (x - x_m)/x_s etc.):

          ∂x̂/∂x = 1/x_s          ∂x/∂x̂ = x_s
          ∂t̂/∂t = 1/t_s          ∂t/∂t̂ = t_s
          ∂û/∂u = 1/u_s          ∂u/∂û = u_s

    So if v̂ = (ξ̂, η̂, φ̂) is the normalised vector field, the
    physical field is
          ξ_phys = x_s · ξ̂,  η_phys = t_s · η̂,  φ_phys = u_s · φ̂.

    And we must evaluate v̂ at (x̂_i, t̂_j, û_{ij}) first.
    """
    sol_sample = sol_batch[0, 0].to(device)
    Nx, Nt = sol_sample.shape
    stats = gen._data_stats
    x_m, x_s = stats['x_mean'], stats['x_std']
    t_m, t_s = stats['t_mean'], stats['t_std']
    u_m, u_s = stats['u_mean'], stats['u_std']

    x_phys = torch.tensor(x_grid, dtype=torch.float32, device=device)
    t_phys = torch.tensor(t_grid, dtype=torch.float32, device=device)
    X_phys, T_phys = torch.meshgrid(x_phys, t_phys, indexing='ij')

    X_n = (X_phys - x_m) / x_s
    T_n = (T_phys - t_m) / t_s
    U_n = (sol_sample - u_m) / u_s
    ones = torch.ones_like(X_n)
    pts = torch.stack([X_n, T_n, U_n, ones], dim=-1)

    Li_eff = gen.getLi()

    generators_np, norms = [], []
    for k in range(Li_eff.shape[0]):
        v_n = torch.einsum('ij,xyj->xyi', Li_eff[k], pts)
        xi_phys  = (x_s * v_n[..., 0]).cpu().numpy()
        eta_phys = (t_s * v_n[..., 1]).cpu().numpy()
        phi_phys = (u_s * v_n[..., 2]).cpu().numpy()

        norm = float(np.sqrt(
            np.mean(xi_phys**2) + np.mean(eta_phys**2)
            + np.mean(phi_phys**2)))
        norms.append(norm)
        generators_np.append((xi_phys, eta_phys, phi_phys))

    return generators_np, np.array(norms)
