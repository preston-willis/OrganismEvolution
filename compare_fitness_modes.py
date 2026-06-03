#!/usr/bin/env python3
"""Compare one CNN under all fitness objectives (same physics, different scores)."""
import argparse

import torch

from config import CNN_TRAINING_MAX_TIME
from main import (
    CNN_FITNESS_MODES,
    EnergyDistributionCNN,
    Simulation,
    cnn_fitness_mode_label,
    device,
    load_latest_cnn,
    run_cnn_fitness_rollout,
)


def evaluate_cnn_all_modes(cnn, max_time):
    rows = []
    for mode in CNN_FITNESS_MODES:
        sim = Simulation(enable_debug=False)
        sim.organism_manager.energy_distribution_cnn = cnn
        fitness, cumulative_cells, _ = run_cnn_fitness_rollout(sim, max_time, fitness_mode=mode)
        destroyed = sim.organism_manager.destroyed_energy
        destroyed_s = sim.organism_manager.destroyed_entropy
        final_cells = torch.sum(sim.organism_manager.topology_matrix).item()
        rows.append((mode, fitness, cumulative_cells, destroyed, destroyed_s, final_cells))
        del sim
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare CNN fitness across all CNN_FITNESS_MODES")
    parser.add_argument("--ticks", type=int, default=CNN_TRAINING_MAX_TIME, help="Rollout length per mode")
    parser.add_argument("--load", action="store_true", help="Use latest saved CNN from data/")
    args = parser.parse_args()

    if args.load:
        cnn = load_latest_cnn()
        if cnn is None:
            raise SystemExit("No saved CNN found in data/")
    else:
        cnn = EnergyDistributionCNN(device)

    print(f"Rollout ticks: {args.ticks}")
    print(
        f"{'mode':<22} {'label':<20} {'fitness':>12} {'sum_cells':>12} "
        f"{'destroyed_E':>12} {'destroyed_S':>12} {'final_cells':>12}"
    )
    for mode, fitness, cumulative_cells, destroyed, destroyed_s, final_cells in evaluate_cnn_all_modes(
        cnn, args.ticks
    ):
        print(
            f"{mode:<22} {cnn_fitness_mode_label(mode):<20} "
            f"{fitness:12.4f} {cumulative_cells:12.4f} {destroyed:12.4f} "
            f"{destroyed_s:12.4f} {final_cells:12.0f}"
        )


if __name__ == "__main__":
    main()
