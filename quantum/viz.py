import math
from functools import lru_cache

import numpy as np
import torch
from scipy.special import eval_hermitenorm

from quantum.config import POSITION_X_MAX
from quantum.physics import evolution_step, hermitian, reduced_entropy_from_psi


def ideal_s(n):
    return 0.5 * math.log(n)


def basin_threshold_s(n):
    return 0.1 * math.log(n)


def joint_amplitude_grid(psi_AB, n):
    return psi_AB.reshape(n, n)


@lru_cache(maxsize=8)
def _position_basis(n, x_max):
    x = np.linspace(-x_max, x_max, n)
    phi = np.zeros((n, n), dtype=np.float64)
    for level in range(n):
        H = eval_hermitenorm(level, x)
        norm = 1.0 / np.sqrt((2**level) * math.factorial(level) * math.sqrt(math.pi))
        phi[:, level] = norm * H * np.exp(-0.5 * x**2)
        column_norm = np.linalg.norm(phi[:, level])
        if column_norm > 0:
            phi[:, level] /= column_norm
    return phi


def fock_grid_to_position(grid, n, x_max=POSITION_X_MAX):
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    phi = _position_basis(n, x_max)
    if np.iscomplexobj(grid):
        return np.einsum("ab,xa,yb->xy", grid, phi, phi, optimize=True)
    return np.einsum("ab,xa,yb->xy", grid, phi, phi, optimize=True)


def subsystem_push_grids(M_A, M_B, psi_AB, n):
    H_A = hermitian(M_A.cpu())
    H_B = hermitian(M_B.cpu())
    psi_ij = joint_amplitude_grid(psi_AB.cpu(), n)
    push_A = H_A @ psi_ij
    push_B = psi_ij @ H_B
    return psi_ij, push_A, push_B


def entropy_trajectory(M_A, M_B, psi_AB, n, steps, g, dt, device):
    s_a = []
    s_b = []
    state = psi_AB.clone()
    for _ in range(steps):
        _, state = evolution_step(M_A, M_B, state, n, g, dt, device)
        s_a.append(reduced_entropy_from_psi(state, n, "A").item())
        s_b.append(reduced_entropy_from_psi(state, n, "B").item())
    return s_a, s_b


def probability_grid_numpy(psi_AB, n, x_max=POSITION_X_MAX):
    psi_xy = fock_grid_to_position(joint_amplitude_grid(psi_AB, n), n, x_max)
    return np.abs(psi_xy) ** 2


def push_drive_grids_numpy(M_A, M_B, psi_AB, n, x_max=POSITION_X_MAX):
    _, push_A, push_B = subsystem_push_grids(M_A, M_B, psi_AB, n)
    drive_A = np.abs(fock_grid_to_position(push_A, n, x_max)) ** 2
    drive_B = np.abs(fock_grid_to_position(push_B, n, x_max)) ** 2
    return drive_A, drive_B, drive_A + drive_B


def energy_organization_rgb_numpy(M_A, M_B, psi_AB, n, x_max=POSITION_X_MAX):
    drive_A, drive_B, _ = push_drive_grids_numpy(M_A, M_B, psi_AB, n, x_max)
    a_max = float(drive_A.max())
    b_max = float(drive_B.max())
    if a_max < 1e-12:
        a_max = 1.0
    if b_max < 1e-12:
        b_max = 1.0
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    rgb[:, :, 0] = drive_A / a_max
    rgb[:, :, 2] = drive_B / b_max
    return rgb
