import numpy as np

from config import (
    DISORDER_SEED,
    DISSIPATION_GAMMA,
    DRIVE_AMP,
    DRIVE_OMEGA,
    DT,
    FLOW_MIN,
    FLOW_PENALTY,
    ROLLOUT_STEPS,
    S_BAND_PENALTY,
    S_HIGH,
    S_LOW,
    S_VAR_PENALTY,
    TRAIN_INITIALS,
    TRANSFER_LAMBDA,
    WINDOW_END,
    WINDOW_START,
)
from dynamics import run_rollout, transfer_metrics_from_grid


def _window_metrics(rollout, lambda_a, window_start, window_end):
    grids = rollout["grids"][window_start : window_end + 1]
    entropies = rollout["entropies"][window_start : window_end + 1]
    transfers = []
    for grid in grids:
        transfers.append(transfer_metrics_from_grid(grid, lambda_a)["transfer"])
    return transfers, entropies


def _s_band_penalty(mean_s):
    if mean_s < S_LOW:
        return (S_LOW - mean_s) ** 2
    if mean_s > S_HIGH:
        return (mean_s - S_HIGH) ** 2
    return 0.0


def _score_rollout(transfers, entropies):
    mean_transfer = float(np.mean(transfers))
    mean_s = float(np.mean(entropies))
    var_s = float(np.var(entropies))

    score = mean_transfer
    score -= S_BAND_PENALTY * _s_band_penalty(mean_s)
    score -= S_VAR_PENALTY * var_s
    if mean_transfer < FLOW_MIN:
        score -= FLOW_PENALTY * (FLOW_MIN - mean_transfer)

    return score, {
        "mean_transfer": mean_transfer,
        "var_transfer": float(np.var(transfers)),
        "mean_s": mean_s,
        "var_s": var_s,
    }


def regulated_fitness(
    genome,
    n_qubits,
    dt=DT,
    n_steps=ROLLOUT_STEPS,
    initial="vacuum",
    seed=DISORDER_SEED,
    drive_amp=DRIVE_AMP,
    drive_omega=DRIVE_OMEGA,
    gamma=DISSIPATION_GAMMA,
    lambda_a=TRANSFER_LAMBDA,
    window_start=WINDOW_START,
    window_end=WINDOW_END,
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
        dissipation_gamma=gamma,
    )
    transfers, entropies = _window_metrics(
        rollout, lambda_a, window_start, window_end
    )
    if len(transfers) == 0:
        return -1.0, {
            "initial": initial,
            "transfer": 0.0,
            "p_b_exc": 0.0,
            "p_a_exc": 0.0,
            "mean_s": 0.0,
            "var_s": 0.0,
            "mean_transfer": 0.0,
        }

    score, window = _score_rollout(transfers, entropies)
    final = transfer_metrics_from_grid(rollout["grids"][-1], lambda_a)
    metrics = {
        "initial": initial,
        "transfer": final["transfer"],
        "p_b_exc": final["p_b_exc"],
        "p_a_exc": final["p_a_exc"],
        **window,
    }
    return score, metrics


def evaluate_genome(genome, n_qubits, drive_amp=DRIVE_AMP, dissipation_gamma=DISSIPATION_GAMMA):
    per_initial = {}
    scores = []
    for initial in TRAIN_INITIALS:
        score, metrics = regulated_fitness(
            genome,
            n_qubits,
            initial=initial,
            drive_amp=drive_amp,
            gamma=dissipation_gamma,
        )
        per_initial[initial] = metrics
        scores.append(score)

    worst = min(scores)
    vacuum = per_initial["vacuum"]
    random_m = per_initial["random"]
    combined = {
        "fitness": worst,
        "regulated_score": max(0.0, worst),
        "mean_s": float(np.mean([vacuum["mean_s"], random_m["mean_s"]])),
        "var_s": float(np.mean([vacuum["var_s"], random_m["var_s"]])),
        "mean_transfer": float(np.min([vacuum["mean_transfer"], random_m["mean_transfer"]])),
        "flow_vacuum": vacuum["mean_transfer"],
        "flow_random": random_m["mean_transfer"],
        "s_vacuum": vacuum["mean_s"],
        "s_random": random_m["mean_s"],
        "transfer": float(np.min([vacuum["transfer"], random_m["transfer"]])),
        "p_b_exc": vacuum["p_b_exc"],
        "p_a_exc": vacuum["p_a_exc"],
        "p_b_exc_random": random_m["p_b_exc"],
        "p_a_exc_random": random_m["p_a_exc"],
        "per_initial": per_initial,
    }
    return worst, combined

