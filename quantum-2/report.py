import json
import os

from config import DRIVE_AMP, DISSIPATION_GAMMA, S_HIGH, S_LOW, S_TARGET, TRAIN_INITIALS
from fitness import evaluate_genome, regulated_fitness
from hamiltonian import pad_genome


def hamiltonian_string(genome):
    terms = sorted(genome, key=lambda x: abs(x[0]), reverse=True)
    parts = []
    for coeff, pstring in terms:
        if coeff >= 0:
            parts.append(f"+{coeff:.4f}·{pstring}")
        else:
            parts.append(f"{coeff:.4f}·{pstring}")
    return "H = " + " ".join(parts)


def build_report(genome, n_qubits, out_dir, train_metrics=None):
    os.makedirs(out_dir, exist_ok=True)
    genome = pad_genome(genome, n_qubits)
    json_path = os.path.join(out_dir, "critical_hamiltonian.json")

    _, eval_metrics = evaluate_genome(
        genome,
        n_qubits,
        drive_amp=DRIVE_AMP,
        dissipation_gamma=DISSIPATION_GAMMA,
    )
    per_initial = {}
    for initial in TRAIN_INITIALS:
        _, m = regulated_fitness(
            genome,
            n_qubits,
            initial=initial,
            drive_amp=DRIVE_AMP,
            gamma=DISSIPATION_GAMMA,
        )
        per_initial[initial] = m

    payload = {
        "hamiltonian": hamiltonian_string(genome),
        "n_qubits": n_qubits,
        "genome": [[float(c), s] for c, s in genome],
        "s_target": S_TARGET,
        "s_band": [S_LOW, S_HIGH],
        "eval": eval_metrics,
        "per_initial": per_initial,
        "training": train_metrics,
        "environment": {
            "drive_amp": DRIVE_AMP,
            "gamma_on_a": DISSIPATION_GAMMA,
            "pump": "sum X_i on B",
            "sink": "amplitude damping on A",
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 60)
    print("REGULATED FLOW REPORT")
    print("=" * 60)
    print(payload["hamiltonian"])
    print()
    print(f"  fitness (min over ICs) = {eval_metrics['fitness']:.4f}")
    print(
        f"  entanglement  S vac/rnd = {eval_metrics['s_vacuum']:.4f} / "
        f"{eval_metrics['s_random']:.4f}  (band {S_LOW:.2f}–{S_HIGH:.2f})"
    )
    print(
        f"  A→B flow      T vac/rnd = {eval_metrics['flow_vacuum']:.4f} / "
        f"{eval_metrics['flow_random']:.4f}  (min {eval_metrics['mean_transfer']:.4f})"
    )
    print()
    print(f"Wrote {json_path}")
    print("=" * 60)

    return payload
