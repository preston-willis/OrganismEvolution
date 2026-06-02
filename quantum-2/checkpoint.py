import json
import os


def clone_genome(genome):
    return [(float(c), str(s)) for c, s in genome]


def save_genome_checkpoint(
    genome,
    n_qubits,
    out_dir,
    task,
    fitness,
    metrics,
    generation,
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "critical_hamiltonian.json")
    payload = {
        "generation": generation,
        "n_qubits": n_qubits,
        "task": task,
        "fitness": fitness,
        "genome": [[float(c), s] for c, s in genome],
        "training": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
