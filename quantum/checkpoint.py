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


def _population_to_cpu(population):
    return [
        (M_A.detach().cpu(), M_B.detach().cpu())
        for M_A, M_B in population
    ]


def _population_to_device(population, device, dtype):
    return [
        (
            M_A.to(device=device, dtype=dtype),
            M_B.to(device=device, dtype=dtype),
        )
        for M_A, M_B in population
    ]


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
    best_M_A,
    best_M_B,
    n,
    population_size,
    steps_per_eval,
    g,
    dt,
):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = checkpoint_path(generation, best_fitness)
    state = {
        "generation": generation,
        "population": _population_to_cpu(population),
        "best_M_A": best_M_A.detach().cpu(),
        "best_M_B": best_M_B.detach().cpu(),
        "history": history,
        "n": n,
        "population_size": population_size,
        "steps_per_eval": steps_per_eval,
        "g": g,
        "dt": dt,
    }
    torch.save(state, filename)


def load_latest_checkpoint(device, dtype):
    files = glob.glob(_checkpoint_glob())
    if not files:
        raise FileNotFoundError(
            f"No saved checkpoints found in {DATA_DIR}/ (expected quantum_gen*.pt)"
        )
    files.sort(key=_generation_from_path, reverse=True)
    path = files[0]
    state = torch.load(path, map_location="cpu")
    if "best_M_A" not in state or "best_M_B" not in state:
        raise KeyError(
            f"Checkpoint {path} has no best_M_A/best_M_B; "
            "re-save from a newer training run"
        )
    population = _population_to_device(state["population"], device, dtype)
    best_M_A = state["best_M_A"].to(device=device, dtype=dtype)
    best_M_B = state["best_M_B"].to(device=device, dtype=dtype)
    return {
        "path": path,
        "generation": state["generation"],
        "population": population,
        "best_M_A": best_M_A,
        "best_M_B": best_M_B,
        "history": state["history"],
        "n": state["n"],
        "population_size": state["population_size"],
        "steps_per_eval": state["steps_per_eval"],
        "g": state["g"],
        "dt": state["dt"],
    }
