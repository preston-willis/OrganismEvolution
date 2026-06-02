import os

import torch

from quantum.checkpoint import clear_saved_checkpoints, load_latest_checkpoint, save_checkpoint
from quantum.config import (
    DISORDER_SEED,
    DT,
    ELITISM,
    ELITISM_FREE_GENERATIONS,
    G,
    LOG_INTERVAL,
    MUTATION_RATE,
    N,
    N_GENERATIONS,
    POPULATION_SIZE,
    STEPS_PER_EVAL,
)
from quantum.device import get_device, get_dtype
from quantum.physics import (
    critical_edge_initial_state,
    reduced_entropy_from_psi,
    rollout_fitness,
    rollout_fitness_batch,
    track_criticality,
)


def training_psi_initial(n, dtype, device):
    return critical_edge_initial_state(n, dtype, device, DISORDER_SEED)


def init_population(population_size, n, dtype, device):
    return [
        torch.randn(n, n, dtype=dtype, device=device) * 0.1
        for _ in range(population_size)
    ]


def _mutate_matrix(M, mutation_rate):
    return M + torch.randn_like(M) * mutation_rate


def _next_population(population, scores, mutation_rate, elitism):
    population_size = len(population)
    ranked = sorted(range(population_size), key=lambda i: scores[i], reverse=True)
    survivors = [population[i] for i in ranked[: population_size // 2]]
    best = population[ranked[0]]

    if not elitism:
        new_population = list(survivors)
        for M in survivors:
            new_population.append(_mutate_matrix(M, mutation_rate))
        return new_population

    new_population = [best.clone()]
    for M in survivors:
        if len(new_population) >= population_size:
            break
        new_population.append(M)
    for M in survivors:
        if len(new_population) >= population_size:
            break
        new_population.append(_mutate_matrix(M, mutation_rate))
    return new_population


def _refill_from_survivors(survivors, target_size, mutation_rate):
    new_population = list(survivors)
    i = 0
    while len(new_population) < target_size:
        new_population.append(_mutate_matrix(survivors[i % len(survivors)], mutation_rate))
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
    psi_initial=None,
):
    if len(population) == target_size:
        return population

    if psi_initial is None:
        psi_initial = training_psi_initial(n, dtype, device)
    scores, _ = evaluate_population(
        population, psi_initial, n, steps_per_eval, g, dt, device
    )
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if len(population) > target_size:
        return [population[i] for i in ranked[:target_size]]

    survivor_count = max(len(population) // 2, 1)
    survivors = [population[i] for i in ranked[:survivor_count]]
    return _refill_from_survivors(survivors, target_size, mutation_rate)


def evaluate_organism(M, psi_AB, n, steps, g, dt, device):
    return rollout_fitness(M, psi_AB, n, steps, g, dt, device)


def evaluate_population(population, psi_AB, n, steps, g, dt, device):
    M_batch = torch.stack(population)
    psi = psi_AB.unsqueeze(0).expand(M_batch.shape[0], -1).clone()
    fitness, psi_out = rollout_fitness_batch(M_batch, psi, n, steps, g, dt, device)
    scores = fitness.tolist()
    states = [psi_out[i] for i in range(M_batch.shape[0])]
    return scores, states


def run_generation(
    population,
    n,
    steps_per_eval,
    g,
    dt,
    device,
    dtype,
    psi_initial=None,
    generation=0,
):
    if psi_initial is None:
        psi_initial = training_psi_initial(n, dtype, device)
    scores, states = evaluate_population(
        population, psi_initial, n, steps_per_eval, g, dt, device
    )

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_M = population[best_idx]
    psi_final = states[best_idx]
    metrics = track_criticality(psi_final, best_M, n, dtype, device)
    metrics["entanglement_initial"] = reduced_entropy_from_psi(psi_initial, n).item()

    use_elitism = ELITISM and generation >= ELITISM_FREE_GENERATIONS
    new_population = _next_population(population, scores, MUTATION_RATE, use_elitism)

    return (
        new_population,
        scores[best_idx],
        metrics,
        best_M,
        psi_initial,
        psi_final,
    )


def run_evolution(M, n, n_generations, population_size=2, steps_per_eval=STEPS_PER_EVAL, g=G, dt=DT, device=None, dtype=None):
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = get_dtype()

    if n_generations == 0:
        return None

    population = init_population(population_size, n, dtype, device)
    population[0] = M

    for gen in range(n_generations):
        population, _, metrics, _, _, _ = run_generation(
            population, n, steps_per_eval, g, dt, device, dtype, generation=gen
        )

    return metrics


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
    psi_training = training_psi_initial(n, dtype, device)
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
                psi_training,
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
            metrics,
            best_M,
            psi_initial,
            psi_final,
        ) = run_generation(
            population,
            n,
            steps_per_eval,
            g,
            dt,
            device,
            dtype,
            psi_training,
            generation,
        )
        history.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                **metrics,
            }
        )

        if generation % LOG_INTERVAL == 0:
            print(
                f"Gen {generation:5d} | fitness: {best_fitness:.4f} | "
                f"F_static: {metrics['f_static']:.4f} | "
                f"F_dynamic: {metrics['f_dynamic']:.4f} | "
                f"r: {metrics['r_mean']:.4f} | "
                f"r*: {metrics['r_star']:.4f} | "
                f"f_ord: {metrics['f_order_static']:.3f} | "
                f"f_chaos: {metrics['f_chaos_static']:.3f} | "
                f"S: {metrics['entanglement']:.4f}"
            )

        if save_checkpoints:
            save_checkpoint(
                generation,
                best_fitness,
                population,
                history,
                best_M,
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
                metrics["entanglement"],
                history,
                best_M,
                psi_initial,
                psi_final,
                steps_per_eval,
                g,
                dt,
                device,
            )

    return history, population
