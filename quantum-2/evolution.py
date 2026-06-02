import random

import numpy as np

from config import (
    MUTATION_RATE,
    N_GENERATIONS,
    N_QUBITS,
    N_TERMS,
    POPULATION_SIZE,
)
from fitness import fitness, level_spacing_ratio
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
        return g1 if random.random() < 0.5 else g2
    cut = random.randint(1, n - 1)
    return g1[:cut] + g2[cut:]


def evolve(
    n_qubits=N_QUBITS,
    pop_size=POPULATION_SIZE,
    generations=N_GENERATIONS,
    n_terms=N_TERMS,
):
    population = [random_genome(n_qubits, n_terms) for _ in range(pop_size)]
    best_score = 0.0
    best_genome = population[0]
    best_r = 0.0

    for gen in range(generations):
        scores = [fitness(g, n_qubits) for g in population]
        ranked = sorted(
            zip(scores, population),
            reverse=True,
            key=lambda x: x[0][0],
        )

        gen_score, gen_r = ranked[0][0]
        gen_genome = ranked[0][1]
        if gen_score > best_score:
            best_score = gen_score
            best_r = gen_r
            best_genome = gen_genome

        if gen_r is None:
            print(f"Gen {gen}: fitness={gen_score:.2f}, r=invalid")
        else:
            print(f"Gen {gen}: fitness={gen_score:.2f}, r={gen_r:.4f}")

        survivors = [
            g
            for (s, _), g in ranked
            if s > 0 and np.isfinite(s)
        ][: pop_size // 5]
        if not survivors:
            survivors = [random_genome(n_qubits, n_terms) for _ in range(pop_size // 5)]

        new_pop = [best_genome]
        new_pop.extend(survivors[: pop_size // 5])
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = mutate(crossover(p1, p2), n_qubits)
            new_pop.append(child)

        population = new_pop[:pop_size]

    return best_score, best_r, best_genome
