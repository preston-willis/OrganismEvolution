import numpy as np

from config import R_TARGET
from hamiltonian import build_hamiltonian


def level_spacing_ratio(H):
    eigenvalues = np.sort(np.linalg.eigvalsh(H))
    deltas = np.diff(eigenvalues)
    deltas = deltas[deltas > 1e-10]
    if len(deltas) < 2:
        return None

    ratios = np.minimum(deltas[:-1], deltas[1:]) / np.maximum(deltas[:-1], deltas[1:])
    if len(ratios) == 0:
        return None
    return float(np.mean(ratios))


def fitness(genome, n_qubits):
    H = build_hamiltonian(genome, n_qubits)
    r = level_spacing_ratio(H)
    if r is None:
        return 0.0, None
    score = 1.0 / (abs(r - R_TARGET) + 1e-6)
    return score, r


def spacing_ratios(H):
    eigenvalues = np.sort(np.linalg.eigvalsh(H))
    deltas = np.diff(eigenvalues)
    deltas = deltas[deltas > 1e-10]
    if len(deltas) < 2:
        return np.array([])
    return np.minimum(deltas[:-1], deltas[1:]) / np.maximum(deltas[:-1], deltas[1:])
