"""Shared neural network modules: FNO surrogate, spectral conv, grids."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ExperimentConfig

try:
    from neuralop.models import FNO as NeuralOpFNO
    HAS_NEURALOP = True
    print("NeuralOp FNO loaded successfully.")
except ImportError:
    HAS_NEURALOP = False

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_ch, x.size(-2), x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy",
                                                x_ft[:, :, :m1, :m2],
                                                torch.view_as_complex(self.weights1))
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy",
                                                 x_ft[:, :, -m1:, :m2],
                                                 torch.view_as_complex(self.weights2))
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class SimpleFNO2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes=16, width=32, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                SpectralConv2d(width, width, modes, modes),
                nn.Conv2d(width, width, 1),
            ]))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, out_ch, 1))

    def forward(self, x):
        x = self.lift(x)
        for spec, w in self.layers:
            x = F.gelu(spec(x) + w(x)) + x
        return self.proj(x)


def make_fno_1d(in_ch=1, out_ch=1, cfg=None):
    cfg = cfg or ExperimentConfig()
    if HAS_NEURALOP:
        from neuralop.models import FNO as NeuralOpFNOBase
        class NeuralOpFNO1d(nn.Module):
            def __init__(self, in_ch, out_ch, modes, width, n_layers):
                super().__init__()
                self.fno = NeuralOpFNOBase(n_modes=(modes,), hidden_channels=width,
                                           in_channels=in_ch, out_channels=out_ch,
                                           n_layers=n_layers)
            def forward(self, x):
                return self.fno(x)
        return NeuralOpFNO1d(in_ch, out_ch, cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers)
    else:
        raise ImportError("neuralop is not installed. Please install neuralop to use FNO.")


class LocalGeneratorFNO(nn.Module):
    """Local PDE Generator N_theta[u](x, t) ≈ u_t parameterized by 1D FNO."""
    def __init__(self, cfg=None, integrator="rk2"):
        super().__init__()
        self.cfg = cfg or ExperimentConfig()
        self.integrator = integrator
        self.Nt = self.cfg.time_steps
        # self.fno = make_fno_1d(in_ch=3, out_ch=1, cfg=self.cfg)
        self.fno = make_fno_1d(in_ch=1, out_ch=1, cfg=self.cfg)
    # def generator_rhs(self, u_slice, x_slice, t_slice):
    #     """Evaluate N_theta at one time slice. Inputs: [B, 1, Nx]."""
    #     inp = torch.cat([u_slice, x_slice, t_slice], dim=1) # [B, 3, Nx]
    #     return self.fno(inp)
    def generator_rhs(self, u_slice, x_slice, t_slice):
        return self.fno(u_slice)

    # def generator_field(self, u_full, x_grid, t_grid):
    #     """Evaluate N_theta on full space-time field. Inputs: [B, 1, Nx, Nt]."""
    #     # FNO1d operates on 1D spatial domain, so we treat Nt as part of the batch dimension
    #     B, C, Nx, Nt = u_full.shape
    #     u_reshaped = u_full.permute(0, 3, 1, 2).reshape(B*Nt, C, Nx)
    #     x_reshaped = x_grid.permute(0, 3, 1, 2).reshape(B*Nt, 1, Nx)
    #     t_reshaped = t_grid.permute(0, 3, 1, 2).reshape(B*Nt, 1, Nx)
        
    #     inp = torch.cat([u_reshaped, x_reshaped, t_reshaped], dim=1) # [B*Nt, 3, Nx]
    #     out_reshaped = self.fno(inp) # [B*Nt, 1, Nx]
        
    #     # Reshape back to [B, 1, Nx, Nt]
    #     return out_reshaped.reshape(B, Nt, 1, Nx).permute(0, 2, 3, 1)
    def generator_field(self, u_full, x_grid, t_grid):
        B, C, Nx, Nt = u_full.shape
        u_reshaped = u_full.permute(0, 3, 1, 2).reshape(B*Nt, C, Nx)
        out = self.fno(u_reshaped)  # только u, без x и t
        return out.reshape(B, Nt, 1, Nx).permute(0, 2, 3, 1)

    def rollout(self, u0, x_coords, t_coords):
        """Integrate u_t = N_theta[u] over explicit saved time grid."""
        B, C, Nx = u0.shape
        
        # Build 1D x_slice and full t_coord sequence
        if x_coords is None:
            x_slice = torch.linspace(0, 1, Nx, device=u0.device).view(1, 1, Nx).expand(B, 1, Nx)
        else:
            x_slice = x_coords.view(B, 1, Nx)
            
        if t_coords is None:
            Nt = self.Nt
            t_vec = torch.linspace(0, 1, Nt, device=u0.device)
            t_seq = t_vec.view(1, 1, 1).expand(B, 1, Nt) # Need expanding per slice
        else:
            Nt = t_coords.shape[-1]
            t_vec = t_coords[0, 0, :] # assuming uniform across batch for dt calculation, or we handle B-specific
            t_seq = t_coords.view(B, 1, Nt)
            
        u_curr = u0
        rollout_states = [u_curr]
        
        for k in range(Nt - 1):
            t_k = t_seq[:, :, k:k+1].expand(B, 1, Nx)
            
            # Per-step dt (handles arbitrary but monotonic t_coords)
            dt = t_seq[:, 0, k+1] - t_seq[:, 0, k]
            dt = dt.view(B, 1, 1)

            if self.integrator == "euler":
                rhs = self.generator_rhs(u_curr, x_slice, t_k)
                u_next = u_curr + dt * rhs
            elif self.integrator == "rk2":
                k1 = self.generator_rhs(u_curr, x_slice, t_k)
                u_mid = u_curr + 0.5 * dt * k1
                t_mid = t_k + 0.5 * dt.expand(B, 1, Nx)
                k2 = self.generator_rhs(u_mid, x_slice, t_mid)
                u_next = u_curr + dt * k2
            elif self.integrator == "rk4":
                k1 = self.generator_rhs(u_curr, x_slice, t_k)
                k2 = self.generator_rhs(u_curr + 0.5*dt*k1, x_slice, t_k + 0.5*dt.expand(B, 1, Nx))
                k3 = self.generator_rhs(u_curr + 0.5*dt*k2, x_slice, t_k + 0.5*dt.expand(B, 1, Nx))
                k4 = self.generator_rhs(u_curr + dt*k3, x_slice, t_k + dt.expand(B, 1, Nx))
                u_next = u_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown integrator {self.integrator}")
                
            u_curr = u_next
            rollout_states.append(u_curr)
            
        return torch.stack(rollout_states, dim=-1) # [B, 1, Nx, Nt]

    def forward(self, u0, x_coords=None, t_coords=None):
        """Compatibility shim allowing u0 to be full field or initial slice."""
        if u0.dim() == 4:
            u0 = u0[:, :, :, 0] # Extract initial slice
        return self.rollout(u0, x_coords, t_coords)


def make_fno(in_ch=1, out_ch=1, cfg=None):
    cfg = cfg or ExperimentConfig()
    if HAS_NEURALOP:
        return NeuralOpFNO(
            n_modes=(cfg.fno_modes, cfg.fno_modes),
            hidden_channels=cfg.fno_hidden,
            in_channels=in_ch, out_channels=out_ch,
            n_layers=cfg.fno_layers,
            channel_mlp_dropout=0.1
        )
    return SimpleFNO2d(in_ch, out_ch, cfg.fno_modes, cfg.fno_hidden, cfg.fno_layers)


class FNOSurrogate(nn.Module):
    """FNO PDE solver: (u0, x_coords) → u(x, t) with spatial + temporal coordinates."""
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or ExperimentConfig()
        self.Nt = cfg.time_steps
        self.fno = make_fno(3, 1, cfg)  # 3 channels: u0 broadcast + x coord + t coord

    def forward(self, u0, x_coords=None, t_coords=None):
        """
        u0: [B, 1, Nx]
        x_coords: [B, 1, Nx] or [B, 1, Nx, Nt] — physical x positions
        t_coords: [B, 1, Nt] or [B, 1, Nx, Nt] — physical t positions
        """
        B, C, Nx = u0.shape
        Nt = self.Nt

        u0_broadcast = u0.unsqueeze(-1).expand(B, 1, Nx, Nt)
        
        if t_coords is None:
            t_coord = torch.linspace(0, 1, Nt, device=u0.device).view(1, 1, 1, Nt).expand(B, 1, Nx, Nt)
        elif t_coords.dim() == 3:
            t_coord = t_coords.unsqueeze(2).expand(B, 1, Nx, Nt)
        else:
            t_coord = t_coords

        if x_coords is None:
            x_coord = torch.linspace(0, 1, Nx, device=u0.device).view(1, 1, Nx, 1).expand(B, 1, Nx, Nt)
        elif x_coords.dim() == 3:
            x_coord = x_coords.unsqueeze(-1).expand(B, 1, Nx, Nt)
        else:
            x_coord = x_coords

        return self.fno(torch.cat([u0_broadcast, x_coord, t_coord], dim=1))


def create_grids(B, Nx, Nt, device):
    x = torch.linspace(0, 1, Nx, device=device)
    t = torch.linspace(0, 1, Nt, device=device)
    xg, tg = torch.meshgrid(x, t, indexing='ij')
    return (xg.view(1, 1, Nx, Nt).expand(B, 1, Nx, Nt),
            tg.view(1, 1, Nx, Nt).expand(B, 1, Nx, Nt))


def apply_symmetry_transform(xi, eta, phi, u_tensor, tau, device):
    """Apply infinitesimal symmetry as grid deformation + value shift."""
    B, C, Nx, Nt = u_tensor.shape
    gen_Nx, gen_Nt = xi.shape

    xi_t = torch.tensor(xi, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    eta_t = torch.tensor(eta, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    phi_t = torch.tensor(phi, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    if gen_Nx != Nx or gen_Nt != Nt:
        xi_t = F.interpolate(xi_t, size=(Nx, Nt), mode='bilinear', align_corners=True)
        eta_t = F.interpolate(eta_t, size=(Nx, Nt), mode='bilinear', align_corners=True)
        phi_t = F.interpolate(phi_t, size=(Nx, Nt), mode='bilinear', align_corners=True)

    x_base = torch.linspace(-1, 1, Nt, device=device).view(1, 1, 1, -1).expand(B, 1, Nx, -1)
    y_base = torch.linspace(-1, 1, Nx, device=device).view(1, 1, -1, 1).expand(B, 1, -1, Nt)

    grid = torch.stack([
        (x_base + tau * eta_t.expand(B, 1, Nx, Nt)).squeeze(1),
        (y_base + tau * xi_t.expand(B, 1, Nx, Nt)).squeeze(1)
    ], dim=-1)

    u_warped = F.grid_sample(u_tensor, grid, mode='bilinear',
                              padding_mode='border', align_corners=True)
    return torch.clamp(u_warped + tau * phi_t.expand(B, C, Nx, Nt), min=-10, max=10)
