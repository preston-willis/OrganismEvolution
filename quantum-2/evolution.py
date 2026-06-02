import random

import numpy as np

from checkpoint import clone_genome, save_genome_checkpoint
from config import (
    DISSIPATION_GAMMA,
    DRIVE_AMP,
    MUTATION_RATE,
    N_GENERATIONS,
    N_QUBITS,
    N_TERMS,
    OUTPUT_DIR,
    POPULATION_SIZE,
    SAVE_EVERY_GENERATION,
)
from fitness import evaluate_genome


def random_pauli_string(n_qubits):
    return "".join(random.choice("IXYZ") for _ in range(n_qubits))


def random_genome(n_qubits, n_terms):
    return [
        (np.random.uniform(-1, 1), random_pauli_string(n_qubits))
        for _ in range(n_terms)
    ]


def mutate(genome, n_qubits, rate=MUTATION_RATE):
    new_genome = []
    for coeff, pstring in genome:
        if random.random() < rate:
            coeff += np.random.normal(0, 0.1)
        if random.random() < rate:
            plist = list(pstring)
            idx = random.randint(0, len(plist) - 1)
            plist[idx] = random.choice("IXYZ")
            pstring = "".join(plist)
        new_genome.append((coeff, pstring))

    if random.random() < 0.1:
        new_genome.append((np.random.uniform(-1, 1), random_pauli_string(n_qubits)))
    if random.random() < 0.1 and len(new_genome) > 2:
        new_genome.pop(random.randint(0, len(new_genome) - 1))

    return new_genome


def crossover(g1, g2):
    n = min(len(g1), len(g2))
    if n <= 1:
        return clone_genome(g1 if random.random() < 0.5 else g2)
    cut = random.randint(1, n - 1)
    return g1[:cut] + g2[cut:]


def _format_gen_line(gen, gen_score, metrics):
    return (
        f"Gen {gen}: fitness={gen_score:.4f}, "
        f"S(v/r)={metrics['s_vacuum']:.3f}/{metrics['s_random']:.3f}, "
        f"flow(v/r)={metrics['flow_vacuum']:.3f}/{metrics['flow_random']:.3f}, "
        f"min_flow={metrics['mean_transfer']:.3f}"
    )


def evolve(
    n_qubits=N_QUBITS,
    pop_size=POPULATION_SIZE,
    generations=N_GENERATIONS,
    n_terms=N_TERMS,
    out_dir=OUTPUT_DIR,
    save_every_gen=SAVE_EVERY_GENERATION,
    dissipation_gamma=DISSIPATION_GAMMA,
):
    population = [random_genome(n_qubits, n_terms) for _ in range(pop_size)]
    best_score = float("-inf")
    best_genome = clone_genome(population[0])
    best_metrics = {}

    for gen in range(generations):
        scored = []
        for g in population:
            score, metrics = evaluate_genome(
                g,
                n_qubits,
                drive_amp=DRIVE_AMP,
                dissipation_gamma=dissipation_gamma,
            )
            scored.append((score, metrics, g))

        ranked = sorted(scored, reverse=True, key=lambda x: x[0])
        gen_score, gen_metrics, gen_genome = ranked[0]

        if gen_score > best_score:
            best_score = gen_score
            best_metrics = dict(gen_metrics)
            best_genome = clone_genome(gen_genome)

        print(_format_gen_line(gen, gen_score, gen_metrics))

        if save_every_gen and out_dir:
            save_genome_checkpoint(
                best_genome,
                n_qubits,
                out_dir,
                "regulated",
                best_score,
                best_metrics,
                gen,
            )
            print(f"  → saved checkpoint (best so far) gen {gen}")

        survivors = [
            g for score, _, g in ranked if np.isfinite(score)
        ][: pop_size // 5]
        if not survivors:
            survivors = [random_genome(n_qubits, n_terms) for _ in range(pop_size // 5)]

        new_pop = [clone_genome(best_genome)]
        new_pop.extend(survivors[: pop_size // 5])
        while len(new_pop) < pop_size:
            if len(survivors) >= 2:
                p1, p2 = random.sample(survivors, 2)
            else:
                p1 = random_genome(n_qubits, n_terms)
                p2 = random_genome(n_qubits, n_terms)
            child = mutate(crossover(p1, p2), n_qubits)
            new_pop.append(child)

        population = new_pop[:pop_size]

    return best_score, best_metrics, best_genome
