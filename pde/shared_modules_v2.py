"""
Translation-equivariant variant of FNOSurrogate.

Key change vs shared_modules.FNOSurrogate:
  - Input channels: (u0_broadcast, t_coord)  — only 2 channels
  - NO x_coord channel, so by FNO construction the map is equivariant
    under spatial translation of u0 modulo boundary effects:
        S_θ(u_0(· − a)) = S_θ(u_0)(· − a, t)
  - Multi-domain-length datasets (e.g. heat with varying L) are still
    supported because different L gives different t range through
    the T ∝ L² scaling, and because dx is handled outside the FNO
    (inside the equivariance_loss via spectral_dx with physical dx).

This is the surrogate recommended for FINS discovery.
FNOSurrogate with explicit x_coord channel causes ∂_x and scaling
generators to be unrecoverable because their equivariance residual
becomes dominated by the mismatch between shifted u and unshifted
x_coord input.

Drop-in replacement:
    from shared_modules_v2 import FNOSurrogateEquivariant as FNOSurrogate
"""

import torch
import torch.nn as nn
from shared_modules import make_fno


class FNOSurrogateEquivariant(nn.Module):
    """
    Translation-equivariant FNO solution operator.

    Input:  u_0 : [B, 1, Nx]   — initial condition
            t_coords : [B, 1, Nt]  — temporal grid (physical times)
    Output: u(x, t) : [B, 1, Nx, Nt]

    The FNO receives 2 channels: u_0 broadcast across time, and t_coord
    broadcast across space. No x_coord channel. This preserves the
    architectural translation-equivariance of FNO that the explicit
    coordinate breaks.
    """
    def __init__(self, cfg=None):
        super().__init__()
        from config import ExperimentConfig
        cfg = cfg or ExperimentConfig()
        self.Nt = cfg.time_steps
        # Two channels: u0_broadcast + t_coord
        self.fno = make_fno(2, 1, cfg)

    def forward(self, u0, x_coords=None, t_coords=None):
        """
        x_coords is accepted for API compatibility but IGNORED.
        u0: [B, 1, Nx]
        t_coords: [B, 1, Nt] (broadcast over space), or None
        """
        B, C, Nx = u0.shape
        Nt = self.Nt

        u0_bc = u0.unsqueeze(-1).expand(B, 1, Nx, Nt)

        if t_coords is None:
            t_coord = torch.linspace(0, 1, Nt, device=u0.device) \
                           .view(1, 1, 1, Nt).expand(B, 1, Nx, Nt)
        elif t_coords.dim() == 3:
            t_coord = t_coords.unsqueeze(2).expand(B, 1, Nx, Nt)
        else:
            t_coord = t_coords

        inp = torch.cat([u0_bc, t_coord], dim=1)
        return self.fno(inp)
