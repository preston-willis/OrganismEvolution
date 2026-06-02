import random

import numpy as np

from checkpoint import clone_genome, save_genome_checkpoint
from config import (
    DRIVE_AMP,
    DRIVE_OMEGA,
    MUTATION_RATE,
    N_GENERATIONS,
    N_QUBITS,
    N_TERMS,
    OUTPUT_DIR,
    POPULATION_SIZE,
    SAVE_EVERY_GENERATION,
    TASK,
    TRAIN_DRIVE,
)
from fitness import evaluate_genome, level_spacing_ratio
from hamiltonian import build_hamiltonian


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


def _format_gen_line(gen, task, gen_score, metrics):
    if task == "transfer":
        return (
            f"Gen {gen}: fitness={gen_score:.4f}, "
            f"transfer={metrics['transfer']:.4f}, "
            f"p_B={metrics['p_b_exc']:.4f}, p_A={metrics['p_a_exc']:.4f}"
        )
    if task == "spacing":
        r = metrics.get("r")
        if r is None:
            return f"Gen {gen}: fitness={gen_score:.2f}, r=invalid"
        return f"Gen {gen}: fitness={gen_score:.2f}, r={r:.4f}"
    r = metrics.get("r")
    r_str = f"{r:.4f}" if r is not None else "invalid"
    return (
        f"Gen {gen}: fitness={gen_score:.4f}, "
        f"transfer={metrics.get('transfer', 0):.4f}, r={r_str}"
    )


def evolve(
    n_qubits=N_QUBITS,
    pop_size=POPULATION_SIZE,
    generations=N_GENERATIONS,
    n_terms=N_TERMS,
    task=TASK,
    train_drive=TRAIN_DRIVE,
    out_dir=OUTPUT_DIR,
    save_every_gen=SAVE_EVERY_GENERATION,
):
    population = [random_genome(n_qubits, n_terms) for _ in range(pop_size)]
    best_score = 0.0
    best_genome = clone_genome(population[0])
    best_metrics = {}

    drive_amp = DRIVE_AMP if train_drive and task in ("transfer", "combined") else 0.0

    for gen in range(generations):
        scored = []
        for g in population:
            score, metrics = evaluate_genome(
                g, n_qubits, task, drive_amp=drive_amp, drive_omega=DRIVE_OMEGA
            )
            scored.append((score, metrics, g))

        ranked = sorted(scored, reverse=True, key=lambda x: x[0])
        gen_score, gen_metrics, gen_genome = ranked[0]

        if gen_score > best_score:
            best_score = gen_score
            best_metrics = dict(gen_metrics)
            best_genome = clone_genome(gen_genome)

        print(_format_gen_line(gen, task, gen_score, gen_metrics))

        if save_every_gen and out_dir:
            save_genome_checkpoint(
                best_genome,
                n_qubits,
                out_dir,
                task,
                best_score,
                best_metrics,
                gen,
            )
            print(f"  → saved checkpoint (best so far) gen {gen}")

        survivors = [
            g for score, _, g in ranked if score > 0 and np.isfinite(score)
        ][: pop_size // 5]
        if not survivors:
            survivors = [random_genome(n_qubits, n_terms) for _ in range(pop_size // 5)]

        new_pop = [clone_genome(best_genome)]
        new_pop.extend(survivors[: pop_size // 5])
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = mutate(crossover(p1, p2), n_qubits)
            new_pop.append(child)

        population = new_pop[:pop_size]

    if task == "spacing" and best_metrics.get("r") is None:
        best_metrics["r"] = level_spacing_ratio(
            build_hamiltonian(best_genome, n_qubits)
        )

    return best_score, best_metrics, best_genome
