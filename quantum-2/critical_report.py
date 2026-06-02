import json
import os

import matplotlib.pyplot as plt
import numpy as np

from analyze import analyze_hamiltonian
from config import R_POISSON, R_TARGET, R_TOLERANCE, R_WIGNER
from fitness import level_spacing_ratio, spacing_ratios
from hamiltonian import build_hamiltonian
from known import KNOWN_CRITICAL, matches_known


def pad_genome(genome, n_qubits):
    padded = []
    for coeff, pstring in genome:
        if len(pstring) > n_qubits:
            continue
        if len(pstring) < n_qubits:
            pstring = pstring + "I" * (n_qubits - len(pstring))
        padded.append((coeff, pstring))
    return padded


def genome_stats(genome, n_qubits):
    genome = pad_genome(genome, n_qubits)
    if not genome:
        return None
    H = build_hamiltonian(genome, n_qubits)
    r_mean = level_spacing_ratio(H)
    ratios = spacing_ratios(H)
    if r_mean is None:
        return None

    lo = min(R_POISSON, R_WIGNER)
    hi = max(R_POISSON, R_WIGNER)
    in_band = float(np.mean((ratios >= lo) & (ratios <= hi))) if len(ratios) else 0.0

    return {
        "n_qubits": n_qubits,
        "r_mean": r_mean,
        "r_std": float(np.std(ratios)) if len(ratios) else 0.0,
        "n_levels": int(2**n_qubits),
        "n_ratios": int(len(ratios)),
        "frac_ratios_in_poisson_goe_band": in_band,
        "distance_to_target": abs(r_mean - R_TARGET),
    }


def reference_stats(n_qubits):
    rows = []
    for name, terms in KNOWN_CRITICAL.items():
        stats = genome_stats(pad_genome(terms, n_qubits), n_qubits)
        if stats is None:
            continue
        rows.append({"name": name, **stats})
    return rows


def scaling_study(genome, base_n_qubits, sizes):
    rows = []
    for n in sizes:
        if n < base_n_qubits:
            continue
        stats = genome_stats(genome, n)
        if stats is not None:
            rows.append(stats)
    return rows


def is_critical(stats, tol=R_TOLERANCE):
    if stats is None:
        return False
    return stats["distance_to_target"] <= tol


def hamiltonian_string(genome):
    terms = sorted(genome, key=lambda x: abs(x[0]), reverse=True)
    parts = []
    for coeff, pstring in terms:
        if coeff >= 0:
            parts.append(f"+{coeff:.4f}·{pstring}")
        else:
            parts.append(f"{coeff:.4f}·{pstring}")
    return "H = " + " ".join(parts)


def plot_scaling(scaling_rows, save_path):
    ns = [row["n_qubits"] for row in scaling_rows]
    rs = [row["r_mean"] for row in scaling_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, rs, "o-", linewidth=2, markersize=8)
    ax.axhline(R_POISSON, color="b", linestyle="--", label="Poisson (0.386)")
    ax.axhline(R_TARGET, color="g", linestyle="--", label="Target (0.458)")
    ax.axhline(R_WIGNER, color="r", linestyle="--", label="Wigner-Dyson (0.530)")
    ax.set_xlabel("qubits")
    ax.set_ylabel("mean r")
    ax.set_title("Finite-size scaling (padded embedding)")
    ax.legend()
    ax.set_xticks(ns)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def build_report(genome, n_qubits, scale_sizes, out_dir, show_plots, train_metrics=None):
    os.makedirs(out_dir, exist_ok=True)

    sizes = sorted({n_qubits} | {n for n in scale_sizes if n >= n_qubits})
    stats = genome_stats(genome, n_qubits)
    scaling_rows = scaling_study(genome, n_qubits, sizes)
    refs = reference_stats(n_qubits)
    known_hits = matches_known(genome)

    primary_critical = is_critical(stats)
    scaling_critical = all(is_critical(row) for row in scaling_rows) if scaling_rows else False

    analyze_path = os.path.join(out_dir, "spectrum_analysis.png")
    scaling_path = os.path.join(out_dir, "finite_size_scaling.png")
    json_path = os.path.join(out_dir, "critical_hamiltonian.json")

    analyze_hamiltonian(
        pad_genome(genome, n_qubits),
        n_qubits,
        show=show_plots,
        save_path=analyze_path,
    )
    if len(scaling_rows) > 1:
        plot_scaling(scaling_rows, scaling_path)

    verdict_parts = []
    if primary_critical:
        verdict_parts.append(
            f"Mean r={stats['r_mean']:.4f} is within {R_TOLERANCE} of critical target {R_TARGET:.3f}."
        )
    else:
        verdict_parts.append(
            f"Mean r={stats['r_mean']:.4f} is NOT within tolerance of target {R_TARGET:.3f}."
        )
    if scaling_critical:
        verdict_parts.append("Finite-size padding preserves critical mean r at all checked sizes.")
    elif scaling_rows:
        verdict_parts.append("Mean r drifts under finite-size embedding — interpret scaling cautiously.")
    if known_hits:
        names = ", ".join(name for name, _ in known_hits)
        verdict_parts.append(f"Dominant terms overlap known models: {names}.")
    else:
        verdict_parts.append("Dominant terms do not match built-in Ising/XXZ references.")

    from config import DRIVE_AMP, DRIVE_OMEGA, ROLLOUT_STEPS
    from fitness import spacing_fitness, transfer_fitness

    _, transfer_eval = transfer_fitness(
        pad_genome(genome, n_qubits),
        n_qubits,
        n_steps=ROLLOUT_STEPS,
        drive_amp=DRIVE_AMP,
        drive_omega=DRIVE_OMEGA,
    )
    _, r_eval = spacing_fitness(pad_genome(genome, n_qubits), n_qubits)

    payload = {
        "critical": primary_critical and scaling_critical,
        "primary_critical": primary_critical,
        "scaling_critical": scaling_critical,
        "verdict": " ".join(verdict_parts),
        "hamiltonian": hamiltonian_string(pad_genome(genome, n_qubits)),
        "n_qubits": n_qubits,
        "genome": [[float(c), s] for c, s in pad_genome(genome, n_qubits)],
        "r_target": R_TARGET,
        "r_poisson": R_POISSON,
        "r_wigner": R_WIGNER,
        "spacing": stats,
        "finite_size_scaling": scaling_rows,
        "reference_models": refs,
        "known_overlap": [{"name": name, "overlap": overlap} for name, overlap in known_hits],
        "transfer_eval": transfer_eval,
        "r_eval": r_eval,
        "training": train_metrics,
        "artifacts": {
            "spectrum_plot": analyze_path,
            "scaling_plot": scaling_path if len(scaling_rows) > 1 else None,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 60)
    print("CRITICAL HAMILTONIAN REPORT")
    print("=" * 60)
    print(payload["hamiltonian"])
    print()
    print(f"  mean r       = {stats['r_mean']:.4f}  (target {R_TARGET:.3f})")
    print(f"  r std        = {stats['r_std']:.4f}")
    print(f"  in-band frac = {stats['frac_ratios_in_poisson_goe_band']:.0%} of ratios between Poisson and GOE")
    print(
        f"  A→B transfer (eval) = {transfer_eval['transfer']:.4f}  "
        f"(p_B={transfer_eval['p_b_exc']:.4f}, p_A={transfer_eval['p_a_exc']:.4f})"
    )
    if r_eval is not None:
        print(f"  spacing r (eval)  = {r_eval:.4f}")
    print()
    print("Finite-size scaling (extra qubits padded with I):")
    for row in scaling_rows:
        flag = "ok" if is_critical(row) else "drift"
        print(f"  N={row['n_qubits']}: r={row['r_mean']:.4f}  [{flag}]")
    print()
    print("Reference models at same N:")
    for ref in refs:
        print(f"  {ref['name']}: r={ref['r_mean']:.4f}")
    print()
    print(f"Verdict: {payload['verdict']}")
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {analyze_path}")
    if len(scaling_rows) > 1:
        print(f"Wrote {scaling_path}")
    print("=" * 60)

    return payload
