import math

import torch

from quantum.config import (
    BASIN_N_GENERATIONS,
    BASIN_N_TRIALS,
    BASIN_N_VALUES,
    DT,
    G,
    STEPS_PER_EVAL,
)
from quantum.device import get_device, get_dtype
from quantum.evolution import run_evolution


def threshold(n):
    return 0.1 * math.log(n)


def basin_size_experiment(
    n_values=None,
    n_trials=BASIN_N_TRIALS,
    n_generations=BASIN_N_GENERATIONS,
    device=None,
    dtype=None,
):
    if n_values is None:
        n_values = BASIN_N_VALUES
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype()

    results = {}

    for n in n_values:
        converged = 0
        for _ in range(n_trials):
            M_A = torch.randn(n, n, dtype=dtype, device=device) * 0.1
            M_B = torch.randn(n, n, dtype=dtype, device=device) * 0.1

            final_complexity = run_evolution(
                M_A,
                M_B,
                n,
                n_generations,
                population_size=2,
                steps_per_eval=STEPS_PER_EVAL,
                g=G,
                dt=DT,
                device=device,
                dtype=dtype,
            )

            if final_complexity["entanglement"] > threshold(n):
                converged += 1

        results[n] = converged / n_trials
        print(f"n={n}: {results[n]:.2%} converged to organized state")

    return results
