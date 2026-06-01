import os

import torch

from quantum.checkpoint import clear_saved_checkpoints, load_latest_checkpoint, save_checkpoint
from quantum.config import (
    DT,
    G,
    INITIAL_STATE,
    LOG_INTERVAL,
    MUTATION_RATE,
    N,
    N_GENERATIONS,
    POPULATION_SIZE,
    STEPS_PER_EVAL,
)
from quantum.device import get_device, get_dtype
from quantum.physics import (
    initial_product_state,
    rollout_fitness,
    rollout_fitness_batch,
    track_complexity,
)


def init_population(population_size, n, dtype, device):
    return [
        (
            torch.randn(n, n, dtype=dtype, device=device) * 0.1,
            torch.randn(n, n, dtype=dtype, device=device) * 0.1,
        )
        for _ in range(population_size)
    ]


def _mutate_organism(M_A, M_B, mutation_rate):
    noise_A = torch.randn_like(M_A) * mutation_rate
    noise_B = torch.randn_like(M_B) * mutation_rate
    return (M_A + noise_A, M_B + noise_B)


def _refill_from_survivors(survivors, target_size, mutation_rate):
    new_population = list(survivors)
    i = 0
    while len(new_population) < target_size:
        M_A, M_B = survivors[i % len(survivors)]
        new_population.append(_mutate_organism(M_A, M_B, mutation_rate))
        i += 1
    return new_population


def resize_population(
    population,
    target_size,
    n,
    steps_per_eval,
    g,
    dt,
    device,
    dtype,
    mutation_rate,
):
    if len(population) == target_size:
        return population

    psi_initial = initial_product_state(n, dtype, device, INITIAL_STATE)
    scores, _ = evaluate_population(
        population, psi_initial, n, steps_per_eval, g, dt, device
    )
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if len(population) > target_size:
        return [population[i] for i in ranked[:target_size]]

    survivor_count = max(len(population) // 2, 1)
    survivors = [population[i] for i in ranked[:survivor_count]]
    return _refill_from_survivors(survivors, target_size, mutation_rate)


def evaluate_organism(M_A, M_B, psi_AB, n, steps, g, dt, device):
    return rollout_fitness(M_A, M_B, psi_AB, n, steps, g, dt, device)


def evaluate_population(population, psi_AB, n, steps, g, dt, device):
    M_A = torch.stack([pair[0] for pair in population])
    M_B = torch.stack([pair[1] for pair in population])
    batch_size = M_A.shape[0]
    psi = psi_AB.unsqueeze(0).expand(batch_size, -1).clone()
    fitness, psi = rollout_fitness_batch(M_A, M_B, psi, n, steps, g, dt, device)
    scores = fitness.tolist()
    states = [psi[i] for i in range(batch_size)]
    return scores, states


def best_organism_from_population(population, psi_AB, n, steps, g, dt, device):
    scores, states = evaluate_population(population, psi_AB, n, steps, g, dt, device)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return population[best_idx], states[best_idx], scores[best_idx]


def run_generation(population, n, steps_per_eval, g, dt, device, dtype):
    psi_initial = initial_product_state(n, dtype, device, INITIAL_STATE)
    scores, states = evaluate_population(
        population, psi_initial, n, steps_per_eval, g, dt, device
    )

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_M_A, best_M_B = population[best_idx]
    psi_final = states[best_idx]
    complexity = track_complexity(psi_final, best_M_A, n)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    survivors = [population[i] for i in ranked[: len(population) // 2]]

    new_population = list(survivors)
    for M_A, M_B in survivors:
        noise_A = torch.randn_like(M_A) * MUTATION_RATE
        noise_B = torch.randn_like(M_B) * MUTATION_RATE
        new_population.append((M_A + noise_A, M_B + noise_B))

    return (
        new_population,
        scores[best_idx],
        complexity,
        best_M_A,
        best_M_B,
        psi_initial,
        psi_final,
    )


def run_evolution(
    M_A,
    M_B,
    n,
    n_generations,
    population_size=2,
    steps_per_eval=STEPS_PER_EVAL,
    g=G,
    dt=DT,
    device=None,
    dtype=None,
):
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype()

    if n_generations == 0:
        return None

    population = init_population(population_size, n, dtype, device)
    population[0] = (M_A, M_B)

    for _ in range(n_generations):
        population, _, complexity, _, _, _, _ = run_generation(
            population, n, steps_per_eval, g, dt, device, dtype
        )

    return complexity


def train(
    n=N,
    population_size=POPULATION_SIZE,
    n_generations=N_GENERATIONS,
    steps_per_eval=STEPS_PER_EVAL,
    g=G,
    dt=DT,
    device=None,
    dtype=None,
    grapher=None,
    load=False,
    save_checkpoints=None,
):
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype()

    if save_checkpoints is None:
        save_checkpoints = "PYTEST_CURRENT_TEST" not in os.environ

    start_generation = 0
    if load:
        ckpt = load_latest_checkpoint(device, dtype)
        if ckpt["n"] != n:
            raise ValueError(
                f"checkpoint n={ckpt['n']} does not match train n={n}"
            )
        population = ckpt["population"]
        history = ckpt["history"]
        start_generation = ckpt["generation"] + 1
        ckpt_pop = len(population)
        if ckpt_pop != population_size:
            print(
                f"Resizing population {ckpt_pop} -> {population_size} "
                f"(checkpoint recorded size {ckpt['population_size']})"
            )
            population = resize_population(
                population,
                population_size,
                n,
                steps_per_eval,
                g,
                dt,
                device,
                dtype,
                MUTATION_RATE,
            )
        print(f"Loaded checkpoint: {ckpt['path']}")
        print(f"Resuming at generation {start_generation}")
    else:
        if save_checkpoints:
            clear_saved_checkpoints()
        population = init_population(population_size, n, dtype, device)
        history = []

    for generation in range(start_generation, n_generations):
        (
            population,
            best_fitness,
            complexity,
            best_M_A,
            best_M_B,
            psi_initial,
            psi_final,
        ) = run_generation(population, n, steps_per_eval, g, dt, device, dtype)
        history.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                **complexity,
            }
        )

        if generation % LOG_INTERVAL == 0:
            print(
                f"Gen {generation:5d} | fitness: {best_fitness:.4f} | "
                f"entanglement: {complexity['entanglement']:.4f} | "
                f"PR: {complexity['participation_ratio']:.2f} | "
                f"structure: {complexity['hamiltonian_structure']:.4f}"
            )

        if save_checkpoints:
            save_checkpoint(
                generation,
                best_fitness,
                population,
                history,
                best_M_A,
                best_M_B,
                n,
                population_size,
                steps_per_eval,
                g,
                dt,
            )

        if grapher is not None:
            grapher.update_generation(
                generation,
                best_fitness,
                complexity["entanglement"],
                history,
                best_M_A,
                best_M_B,
                psi_initial,
                psi_final,
                steps_per_eval,
                g,
                dt,
                device,
            )

    return history, population
