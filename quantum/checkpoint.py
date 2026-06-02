import glob
import os

import torch

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_PACKAGE_DIR, "data")


def _under_pytest():
    return "PYTEST_CURRENT_TEST" in os.environ


def _checkpoint_glob():
    return os.path.join(DATA_DIR, "quantum_gen*.pt")


def checkpoint_path(generation, best_fitness):
    return os.path.join(
        DATA_DIR, f"quantum_gen{generation + 1}_{best_fitness:.6f}.pt"
    )


def _generation_from_path(path):
    base = os.path.basename(path)
    marker = "quantum_gen"
    start = base.index(marker) + len(marker)
    end = base.index("_", start)
    return int(base[start:end])


def _matrix_population_to_cpu(population):
    return [M.detach().cpu() for M in population]


def _matrix_population_to_device(population, device, dtype):
    return [M.to(device=device, dtype=dtype) for M in population]


def clear_saved_checkpoints():
    files = glob.glob(_checkpoint_glob())
    for path in files:
        os.remove(path)
    if files and not _under_pytest():
        print(f"Cleared {len(files)} quantum checkpoint file(s)")


def save_checkpoint(
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
):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = checkpoint_path(generation, best_fitness)
    torch.save(
        {
            "generation": generation,
            "best_M": best_M.detach().cpu(),
            "population": _matrix_population_to_cpu(population),
            "history": history,
            "n": n,
            "population_size": population_size,
            "steps_per_eval": steps_per_eval,
            "g": g,
            "dt": dt,
        },
        filename,
    )


def load_latest_checkpoint(device, dtype):
    files = glob.glob(_checkpoint_glob())
    if not files:
        raise FileNotFoundError(
            f"No saved checkpoints found in {DATA_DIR}/ (expected quantum_gen*.pt)"
        )
    files.sort(key=_generation_from_path, reverse=True)
    path = files[0]
    state = torch.load(path, map_location="cpu")

    if "best_M" in state:
        best_M = state["best_M"].to(device=device, dtype=dtype)
    elif "best_M_A" in state:
        best_M = state["best_M_A"].to(device=device, dtype=dtype)
    else:
        raise KeyError(f"Checkpoint {path} has no best_M or best_M_A")

    if "population" in state and state["population"]:
        first = state["population"][0]
        if isinstance(first, tuple):
            population = [
                pair[0].to(device=device, dtype=dtype) for pair in state["population"]
            ]
        else:
            population = _matrix_population_to_device(
                state["population"], device, dtype
            )
    else:
        raise KeyError(f"Checkpoint {path} has no population")

    return {
        "path": path,
        "generation": state["generation"],
        "best_M": best_M,
        "population": population,
        "history": state["history"],
        "n": state["n"],
        "population_size": state["population_size"],
        "steps_per_eval": state["steps_per_eval"],
        "g": state["g"],
        "dt": state["dt"],
    }
