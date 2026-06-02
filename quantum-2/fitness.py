import numpy as np

from config import (
    DISORDER_SEED,
    DT,
    DRIVE_OMEGA,
    R_TARGET,
    TRAIN_INITIAL,
    TRAIN_ROLLOUT_STEPS,
    TRANSFER_LAMBDA,
    W_SPACING,
    W_TRANSFER,
)
from dynamics import run_rollout, transfer_metrics
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


def spacing_fitness(genome, n_qubits):
    H = build_hamiltonian(genome, n_qubits)
    r = level_spacing_ratio(H)
    if r is None:
        return 0.0, None
    score = 1.0 / (abs(r - R_TARGET) + 1e-6)
    return score, r


def transfer_fitness(
    genome,
    n_qubits,
    dt=DT,
    n_steps=TRAIN_ROLLOUT_STEPS,
    initial=TRAIN_INITIAL,
    seed=DISORDER_SEED,
    drive_amp=0.0,
    drive_omega=DRIVE_OMEGA,
    lambda_a=TRANSFER_LAMBDA,
):
    rollout = run_rollout(
        genome,
        n_qubits,
        dt,
        n_steps,
        initial,
        seed,
        drive_amp=drive_amp,
        drive_omega=drive_omega,
    )
    psi = rollout["states"][-1]
    metrics = transfer_metrics(psi, rollout["n_a"], lambda_a)
    score = max(0.0, metrics["transfer"])
    return score, metrics


def evaluate_genome(
    genome,
    n_qubits,
    task,
    drive_amp=0.0,
    drive_omega=DRIVE_OMEGA,
):
    metrics = {}
    score = 0.0

    if task in ("spacing", "combined"):
        spacing_score, r = spacing_fitness(genome, n_qubits)
        metrics["r"] = r
        metrics["spacing_fitness"] = spacing_score
        if task == "spacing":
            return spacing_score, metrics

    if task in ("transfer", "combined"):
        transfer_score, t_metrics = transfer_fitness(
            genome,
            n_qubits,
            drive_amp=drive_amp,
            drive_omega=drive_omega,
        )
        metrics.update(t_metrics)
        metrics["transfer_fitness"] = transfer_score
        if task == "transfer":
            return transfer_score, metrics

    spacing_score = metrics["spacing_fitness"]
    transfer_score = metrics["transfer_fitness"]
    spacing_part = 1.0 / (abs(metrics["r"] - R_TARGET) + 0.05) if metrics["r"] is not None else 0.0
    score = W_TRANSFER * transfer_score + W_SPACING * spacing_part
    metrics["spacing_part"] = spacing_part
    return score, metrics


def fitness(genome, n_qubits):
    return spacing_fitness(genome, n_qubits)


def spacing_ratios(H):
    eigenvalues = np.sort(np.linalg.eigvalsh(H))
    deltas = np.diff(eigenvalues)
    deltas = deltas[deltas > 1e-10]
    if len(deltas) < 2:
        return np.array([])
    return np.minimum(deltas[:-1], deltas[1:]) / np.maximum(deltas[:-1], deltas[1:])
