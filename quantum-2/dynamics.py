import numpy as np
from scipy.linalg import expm

from hamiltonian import build_hamiltonian, sum_pauli_x


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


def propagate(H, psi0, dt, n_steps):
    states = [psi0.copy()]
    psi = psi0.copy()
    for _ in range(n_steps):
        U = expm(-1j * H * dt)
        psi = U @ psi
        states.append(psi)
    return states


def propagate_driven(H, H_drive, psi0, dt, n_steps, drive_amp, drive_omega):
    states = [psi0.copy()]
    psi = psi0.copy()
    for step in range(n_steps):
        t = step * dt
        H_eff = H + drive_amp * np.cos(drive_omega * t) * H_drive
        U = expm(-1j * H_eff * dt)
        psi = U @ psi
        states.append(psi)
    return states


def _n_qubits_from_psi(psi):
    n = int(np.log2(psi.size))
    if 2**n != psi.size:
        raise ValueError("psi length must be a power of 2")
    return n


def probability_grid(psi, n_a):
    n_qubits = _n_qubits_from_psi(psi)
    n_b = n_qubits - n_a
    dim_a = 2**n_a
    dim_b = 2**n_b
    amp = psi.reshape(dim_a, dim_b)
    return np.abs(amp) ** 2


def transfer_metrics(psi, n_a, lambda_a):
    grid = probability_grid(psi, n_a)
    p_b_exc = float(1.0 - grid[:, 0].sum())
    p_a_exc = float(grid[1:, :].sum())
    transfer = p_b_exc - lambda_a * p_a_exc
    return {
        "transfer": transfer,
        "p_b_exc": p_b_exc,
        "p_a_exc": p_a_exc,
    }


def bipartition_entropy(psi, n_a):
    n_qubits = _n_qubits_from_psi(psi)
    n_b = n_qubits - n_a
    dim_a = 2**n_a
    dim_b = 2**n_b
    amp = psi.reshape(dim_a, dim_b)
    rho_a = amp @ amp.conj().T
    eigenvalues = np.linalg.eigvalsh(rho_a)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def run_rollout(
    genome,
    n_qubits,
    dt,
    n_steps,
    initial,
    seed,
    drive_amp=0.0,
    drive_omega=0.0,
):
    H = build_hamiltonian(genome, n_qubits)
    psi0 = initial_state(n_qubits, initial, seed)
    if drive_amp != 0.0:
        H_drive = sum_pauli_x(n_qubits)
        states = propagate_driven(
            H, H_drive, psi0, dt, n_steps, drive_amp, drive_omega
        )
    else:
        H_drive = None
        states = propagate(H, psi0, dt, n_steps)
    n_a = n_qubits // 2
    entropies = [bipartition_entropy(psi, n_a) for psi in states]
    grids = [probability_grid(psi, n_a) for psi in states]
    return {
        "H": H,
        "H_drive": H_drive,
        "states": states,
        "entropies": entropies,
        "grids": grids,
        "n_a": n_a,
        "dt": dt,
        "n_steps": n_steps,
        "drive_amp": drive_amp,
        "drive_omega": drive_omega,
    }
