"""Central configuration for all experiments."""
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # PDE parameters
    grid_size: int = 64
    time_steps: int = 128
    n_train: int = 1024
    n_test: int = 128
    alpha_heat: float = 0.01
    nu_burgers: float = 0.01
    T_final: float = 1.0

    # Model parameters
    n_sym: int = 6
    fno_modes: int = 64
    fno_hidden: int = 48
    fno_layers: int = 4

    # Training
    epochs_surrogate: int = 500
    epochs_symmetry: int = 50
    epochs_liegan: int = 100
    epochs_laligan: int = 100
    epochs_lig: int = 80
    epochs_augerino: int = 80
    batch_size: int = 8
    lr_surrogate: float = 1e-4
    lr_symmetry: float = 1e-3
    lr_liegan: float = 2e-4
    lr_laligan: float = 2e-4
    lr_lig: float = 1e-3
    lr_augerino: float = 1e-3

    # Loss weights
    lambda_ortho: float = 1e-0
    lambda_lips: float = 1e-2

    # Evaluation
    n_seeds: int = 5
    device: str = 'cuda'

    # Latent dim for LaLiGAN
    latent_dim: int = 64

    # LieNLSD
    lienlsd_hidden: int = 200
    lienlsd_layers: int = 3
    lienlsd_epochs: int = 200
