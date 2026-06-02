import numpy as np


def lowering_operator(n_qubits, qubit):
    """Jump operator |0><1| on one qubit (decay |1> -> |0>)."""
    dim = 2**n_qubits
    L = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        if (idx >> qubit) & 1:
            target = idx - (1 << qubit)
            L[target, idx] = 1.0
    return L


def damping_ops_on_a(n_qubits, n_a):
    return [lowering_operator(n_qubits, q) for q in range(n_a)]


def lindblad_rhs(rho, H, jump_ops, gamma):
    drho = -1j * (H @ rho - rho @ H)
    for L in jump_ops:
        Ld = L.conj().T
        LdL = Ld @ L
        drho = drho + gamma * (2.0 * L @ rho @ Ld - LdL @ rho - rho @ LdL)
    return drho


def trace_normalize(rho):
    tr = np.trace(rho)
    if tr > 1e-12:
        return rho / tr
    return rho


def rk4_lindblad_step(rho, H, jump_ops, gamma, dt):
    def f(r):
        return lindblad_rhs(r, H, jump_ops, gamma)

    k1 = f(rho)
    k2 = f(rho + 0.5 * dt * k1)
    k3 = f(rho + 0.5 * dt * k2)
    k4 = f(rho + dt * k3)
    rho_next = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    rho_next = 0.5 * (rho_next + rho_next.conj().T)
    return trace_normalize(rho_next)
