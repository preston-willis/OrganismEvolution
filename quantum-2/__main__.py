import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ANIM_FRAME_MS,
    DISORDER_SEED,
    DRIVE_AMP,
    DRIVE_OMEGA,
    DT,
    INITIAL_STATE,
    N_GENERATIONS,
    N_QUBITS,
    N_TERMS,
    OUTPUT_DIR,
    POPULATION_SIZE,
    ROLLOUT_STEPS,
    SCALE_SIZES,
)
from animate import run_animation
from critical_report import build_report
from evolution import evolve


def load_genome(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    genome = [(float(c), s) for c, s in data["genome"]]
    n_qubits = data["n_qubits"]
    return genome, n_qubits


def default_genome_path(root):
    return os.path.join(root, OUTPUT_DIR, "critical_hamiltonian.json")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(root, OUTPUT_DIR)
    default_json = default_genome_path(root)

    parser = argparse.ArgumentParser(
        description="Evolve and certify a critical Pauli-string Hamiltonian",
    )
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--pop-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--n-qubits", type=int, default=N_QUBITS)
    parser.add_argument("--n-terms", type=int, default=N_TERMS)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=default_out,
        help="Directory for critical_hamiltonian.json and plots",
    )
    parser.add_argument(
        "--load-genome",
        type=str,
        default="",
        help="Skip evolution; verify an existing genome JSON",
    )
    parser.add_argument(
        "--scale-sizes",
        type=str,
        default="",
        help="Comma-separated qubit counts for finite-size check (default: 4,6,8)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Open matplotlib windows (plots are always saved to --out-dir)",
    )
    parser.add_argument(
        "--evolve-only",
        action="store_true",
        help="Run GA only, skip criticality report",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Time-evolve H and show field + entanglement animation",
    )
    parser.add_argument(
        "--animate-only",
        action="store_true",
        help="Skip GA/report; animate from --load-genome or output/critical_hamiltonian.json",
    )
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument(
        "--initial",
        choices=["vacuum", "random"],
        default=INITIAL_STATE,
    )
    parser.add_argument("--seed", type=int, default=DISORDER_SEED)
    parser.add_argument(
        "--save-animation",
        type=str,
        default="",
        help="Save animation to .gif (e.g. output/field.gif)",
    )
    parser.add_argument("--frame-ms", type=int, default=ANIM_FRAME_MS)
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Add AC drive: H(t)=H_crit + A cos(ωt) sum_i X_i",
    )
    parser.add_argument("--drive-amp", type=float, default=DRIVE_AMP)
    parser.add_argument("--drive-omega", type=float, default=DRIVE_OMEGA)
    args = parser.parse_args()

    if args.scale_sizes:
        scale_sizes = [int(x.strip()) for x in args.scale_sizes.split(",")]
    else:
        scale_sizes = list(SCALE_SIZES)

    genome_path = args.load_genome
    if args.animate_only and not genome_path:
        genome_path = default_json

    best_genome = None
    n_qubits = args.n_qubits

    if args.animate_only:
        if not os.path.isfile(genome_path):
            parser.error(
                f"--animate-only needs {genome_path}; run full pipeline first or pass --load-genome"
            )
        best_genome, n_qubits = load_genome(genome_path)
        print(f"Loaded genome from {genome_path} (N={n_qubits})")
    elif genome_path:
        best_genome, n_qubits = load_genome(genome_path)
        print(f"Loaded genome from {genome_path} (N={n_qubits})")
    else:
        best_score, best_r, best_genome = evolve(
            n_qubits=args.n_qubits,
            pop_size=args.pop_size,
            generations=args.generations,
            n_terms=args.n_terms,
        )
        n_qubits = args.n_qubits
        if best_r is None:
            print(f"\nBest: fitness={best_score:.2f}, r=invalid")
        else:
            print(f"\nBest: fitness={best_score:.2f}, r={best_r:.4f}")
        genome_path = os.path.join(args.out_dir, "critical_hamiltonian.json")

    if not args.animate_only and not args.evolve_only:
        build_report(
            best_genome,
            n_qubits,
            scale_sizes=scale_sizes,
            out_dir=args.out_dir,
            show_plots=args.show_plots,
        )

    if args.animate or args.animate_only:
        if best_genome is None:
            parser.error("--animate requires a genome (run evolution or --load-genome)")
        save_anim = args.save_animation
        if not save_anim and args.animate_only:
            save_anim = os.path.join(args.out_dir, "field.gif")
        show = args.show_plots or not save_anim
        drive_amp = args.drive_amp if args.drive else 0.0
        drive_omega = args.drive_omega
        if args.drive:
            print(f"Drive: A={drive_amp} omega={drive_omega} (sum_i X_i)")
        run_animation(
            best_genome,
            n_qubits,
            args.dt,
            args.steps,
            args.initial,
            args.seed,
            show=show,
            save_path=save_anim if save_anim else "",
            frame_ms=args.frame_ms,
            drive_amp=drive_amp,
            drive_omega=drive_omega,
        )


if __name__ == "__main__":
    main()
