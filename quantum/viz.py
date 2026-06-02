import math
from functools import lru_cache

import numpy as np
import torch
from scipy.special import eval_hermitenorm

from quantum.config import POSITION_X_MAX
from quantum.physics import (
    apply_evolution_step,
    critical_entropy_target,
    evolution_step,
    hermitian,
    reduced_entropy_from_psi,
    volume_law_entropy_bound,
)


def critical_s(n):
    return critical_entropy_target(n)


def area_law_s():
    return 0.0


def volume_law_s(n):
    return volume_law_entropy_bound(n)


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


def fock_vector_to_position(vec, n, x_max=POSITION_X_MAX):
    if hasattr(vec, "detach"):
        vec = vec.detach().cpu().numpy()
    phi = _position_basis(n, x_max)
    return np.einsum("a,xa->x", vec, phi, optimize=True)


def fock_grid_to_position(grid, n, x_max=POSITION_X_MAX):
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    phi = _position_basis(n, x_max)
    if np.iscomplexobj(grid):
        return np.einsum("ab,xa,yb->xy", grid, phi, phi, optimize=True)
    return np.einsum("ab,xa,yb->xy", grid, phi, phi, optimize=True)


def subsystem_push_grids(M, psi_AB, n):
    H = hermitian(M.cpu())
    psi_ij = joint_amplitude_grid(psi_AB.cpu(), n)
    push_A = H @ psi_ij
    push_B = psi_ij @ H
    return psi_ij, push_A, push_B


def entropy_trajectory(M, psi_AB, n, steps, g, dt, device):
    s_a = []
    s_b = []
    state = psi_AB.clone()
    for _ in range(steps):
        _, state = evolution_step(M, state, n, g, dt, device)
        s_a.append(reduced_entropy_from_psi(state, n, "A").item())
        s_b.append(reduced_entropy_from_psi(state, n, "B").item())
    return s_a, s_b


def _excitation_probabilities_from_psi(psi_AB, n):
    probs = torch.abs(psi_AB.reshape(n, n)) ** 2
    p_A = probs.sum(dim=1)
    p_B = probs.sum(dim=0)
    exc_A = 1.0 - p_A[0]
    exc_B = 1.0 - p_B[0]
    return exc_A.item(), exc_B.item()


def excitation_probability_trajectory(M, psi_AB, n, steps, g, dt, device):
    exc_a = []
    exc_b = []
    state = psi_AB.clone()
    exc_A, exc_B = _excitation_probabilities_from_psi(state, n)
    exc_a.append(exc_A)
    exc_b.append(exc_B)
    for _ in range(steps):
        state = apply_evolution_step(M, state, n, g, dt, device)
        exc_A, exc_B = _excitation_probabilities_from_psi(state, n)
        exc_a.append(exc_A)
        exc_b.append(exc_B)
    return exc_a, exc_b


def fock_probability_grid_numpy(psi_AB, n):
    if hasattr(psi_AB, "detach"):
        grid = psi_AB.reshape(n, n).detach().cpu().numpy()
    else:
        grid = psi_AB.reshape(n, n)
    return np.abs(grid) ** 2


def probability_grid_numpy(psi_AB, n, x_max=POSITION_X_MAX):
    psi_xy = fock_grid_to_position(joint_amplitude_grid(psi_AB, n), n, x_max)
    return np.abs(psi_xy) ** 2


def joint_probability_grid_numpy(psi_AB, n, vis_mode, x_max=POSITION_X_MAX):
    if vis_mode == "fock":
        return fock_probability_grid_numpy(psi_AB, n)
    if vis_mode == "spatial":
        return probability_grid_numpy(psi_AB, n, x_max)
    raise ValueError(f"unknown VIS_MODE: {vis_mode}")


def fock_marginal_rgb_numpy(psi_AB, n):
    probs = fock_probability_grid_numpy(psi_AB, n)
    p_A = probs.sum(axis=1)
    p_B = probs.sum(axis=0)
    a_max = float(p_A.max())
    b_max = float(p_B.max())
    if a_max < 1e-12:
        a_max = 1.0
    if b_max < 1e-12:
        b_max = 1.0
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    rgb[:, :, 0] = p_A[:, np.newaxis] / a_max
    rgb[:, :, 2] = p_B[np.newaxis, :] / b_max
    return rgb


def fock_drive_rgb_numpy(M, psi_AB, n):
    _, push_A, push_B = subsystem_push_grids(M, psi_AB, n)
    drive_A = np.abs(push_A) ** 2
    drive_B = np.abs(push_B) ** 2
    a_max = float(drive_A.max())
    b_max = float(drive_B.max())
    if a_max < 1e-12 and b_max < 1e-12:
        return fock_marginal_rgb_numpy(psi_AB, n)
    if a_max < 1e-12:
        a_max = 1.0
    if b_max < 1e-12:
        b_max = 1.0
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    rgb[:, :, 0] = drive_A / a_max
    rgb[:, :, 2] = drive_B / b_max
    return rgb


def push_drive_grids_numpy(M, psi_AB, n, x_max=POSITION_X_MAX):
    _, push_A, push_B = subsystem_push_grids(M, psi_AB, n)
    drive_A = np.abs(fock_grid_to_position(push_A, n, x_max)) ** 2
    drive_B = np.abs(fock_grid_to_position(push_B, n, x_max)) ** 2
    return drive_A, drive_B, drive_A + drive_B


def marginal_mode_rgb_numpy(psi_AB, n, x_max=POSITION_X_MAX):
    if hasattr(psi_AB, "detach"):
        probs = torch.abs(psi_AB.reshape(n, n)) ** 2
        p_A = probs.sum(dim=1).detach().cpu().numpy()
        p_B = probs.sum(dim=0).detach().cpu().numpy()
    else:
        probs = np.abs(psi_AB.reshape(n, n)) ** 2
        p_A = probs.sum(axis=1)
        p_B = probs.sum(axis=0)
    pos_A = np.abs(fock_vector_to_position(p_A, n, x_max)) ** 2
    pos_B = np.abs(fock_vector_to_position(p_B, n, x_max)) ** 2
    a_max = float(pos_A.max())
    b_max = float(pos_B.max())
    if a_max < 1e-12:
        a_max = 1.0
    if b_max < 1e-12:
        b_max = 1.0
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    rgb[:, :, 0] = pos_A[:, np.newaxis] / a_max
    rgb[:, :, 2] = pos_B[np.newaxis, :] / b_max
    return rgb


def energy_organization_rgb_numpy(M, psi_AB, n, x_max=POSITION_X_MAX):
    drive_A, drive_B, _ = push_drive_grids_numpy(M, psi_AB, n, x_max)
    a_max = float(drive_A.max())
    b_max = float(drive_B.max())
    if a_max < 1e-12 and b_max < 1e-12:
        return marginal_mode_rgb_numpy(psi_AB, n, x_max)
    if a_max < 1e-12:
        a_max = 1.0
    if b_max < 1e-12:
        b_max = 1.0
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    rgb[:, :, 0] = drive_A / a_max
    rgb[:, :, 2] = drive_B / b_max
    return rgb


def joint_energy_rgb_numpy(M, psi_AB, n, vis_mode, x_max=POSITION_X_MAX):
    if vis_mode == "fock":
        return fock_drive_rgb_numpy(M, psi_AB, n)
    if vis_mode == "spatial":
        return energy_organization_rgb_numpy(M, psi_AB, n, x_max)
    raise ValueError(f"unknown VIS_MODE: {vis_mode}")


def prob_panel_labels(vis_mode):
    if vis_mode == "fock":
        return "|ψ(i,j)|² Fock", "disorder B", "vacuum A"
    return "|ψ(x,y)|²", "y", "x"
