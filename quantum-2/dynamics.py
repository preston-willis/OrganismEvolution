import numpy as np
from scipy.linalg import expm

from dissipation import damping_ops_on_a, rk4_lindblad_step
from hamiltonian import build_hamiltonian, sum_pauli_x_b


def vacuum_state(n_qubits):
    psi = np.zeros(2**n_qubits, dtype=complex)
    psi[0] = 1.0
    return psi


def random_state(n_qubits, seed):
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2**n_qubits) + 1j * rng.standard_normal(2**n_qubits)
    norm = np.linalg.norm(psi)
    return psi / norm


def initial_state(n_qubits, kind, seed):
    if kind == "vacuum":
        return vacuum_state(n_qubits)
    if kind == "random":
        return random_state(n_qubits, seed)
    raise ValueError(f"unknown initial state: {kind}")


def grid_from_rho(rho, n_a):
    n_qubits = int(np.log2(rho.shape[0]))
    dim_a = 2**n_a
    dim_b = 2**n_qubits // dim_a
    grid = np.zeros((dim_a, dim_b), dtype=float)
    for a in range(dim_a):
        for b in range(dim_b):
            idx = a * dim_b + b
            grid[a, b] = float(np.real(rho[idx, idx]))
    return grid


def transfer_metrics_from_grid(grid, lambda_a):
    p_b_exc = float(1.0 - grid[:, 0].sum())
    p_a_exc = float(grid[1:, :].sum())
    transfer = p_b_exc - lambda_a * p_a_exc
    return {
        "transfer": transfer,
        "p_b_exc": p_b_exc,
        "p_a_exc": p_a_exc,
    }


def partial_trace_b(rho, n_a):
    n_qubits = int(np.log2(rho.shape[0]))
    dim_a = 2**n_a
    dim_b = 2**n_qubits // dim_a
    rho_t = rho.reshape(dim_a, dim_b, dim_a, dim_b)
    rho_a = np.trace(rho_t, axis1=1, axis2=3)
    return rho_a


def bipartition_entropy_rho(rho, n_a):
    rho_a = partial_trace_b(rho, n_a)
    eigenvalues = np.linalg.eigvalsh(rho_a)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def psi_to_rho(psi):
    return np.outer(psi, psi.conj())


def propagate_open(H, H_drive, psi0, dt, n_steps, drive_amp, drive_omega, gamma, n_a):
    jump_ops = damping_ops_on_a(int(np.log2(psi0.size)), n_a)
    rho = psi_to_rho(psi0)
    rhos = [rho.copy()]
    for step in range(n_steps):
        t = step * dt
        H_eff = H + drive_amp * np.cos(drive_omega * t) * H_drive
        U = expm(-1j * H_eff * dt)
        rho = U @ rho @ U.conj().T
        rho = rk4_lindblad_step(rho, np.zeros_like(H), jump_ops, gamma, dt)
        rhos.append(rho)
    return rhos


def run_rollout(
    genome,
    n_qubits,
    dt,
    n_steps,
    initial,
    seed,
    drive_amp,
    drive_omega,
    dissipation_gamma,
):
    H = build_hamiltonian(genome, n_qubits)
    psi0 = initial_state(n_qubits, initial, seed)
    n_a = n_qubits // 2
    H_drive = sum_pauli_x_b(n_qubits, n_a)
    rhos = propagate_open(
        H,
        H_drive,
        psi0,
        dt,
        n_steps,
        drive_amp,
        drive_omega,
        dissipation_gamma,
        n_a,
    )
    entropies = [bipartition_entropy_rho(rho, n_a) for rho in rhos]
    grids = [grid_from_rho(rho, n_a) for rho in rhos]
    return {
        "H": H,
        "rhos": rhos,
        "entropies": entropies,
        "grids": grids,
        "n_a": n_a,
        "dt": dt,
        "n_steps": n_steps,
    }
