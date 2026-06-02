import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ANIM_FRAME_MS,
    DISORDER_SEED,
    DISSIPATION_GAMMA,
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
    SAVE_EVERY_GENERATION,
)
from animate import run_animation
from checkpoint import save_genome_checkpoint
from evolution import evolve
from report import build_report


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
        description="Evolve H for regulated S and A→B flow (robust to vacuum & random init)",
    )
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--pop-size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--n-qubits", type=int, default=N_QUBITS)
    parser.add_argument("--n-terms", type=int, default=N_TERMS)
    parser.add_argument(
        "--no-save-every-gen",
        action="store_true",
        help="Only save checkpoint at end of training",
    )
    parser.add_argument("--out-dir", type=str, default=default_out)
    parser.add_argument("--load-genome", type=str, default="")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--evolve-only", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--animate-only", action="store_true")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--initial", choices=["vacuum", "random"], default=INITIAL_STATE)
    parser.add_argument("--seed", type=int, default=DISORDER_SEED)
    parser.add_argument("--save-animation", type=str, default="")
    parser.add_argument("--frame-ms", type=int, default=ANIM_FRAME_MS)
    parser.add_argument("--drive-amp", type=float, default=DRIVE_AMP)
    parser.add_argument("--drive-omega", type=float, default=DRIVE_OMEGA)
    parser.add_argument("--gamma", type=float, default=DISSIPATION_GAMMA)
    args = parser.parse_args()

    genome_path = args.load_genome
    if args.animate_only and not genome_path:
        genome_path = default_json

    print(
        f"B-pump A={args.drive_amp} omega={args.drive_omega} | "
        f"gamma={args.gamma} on A"
    )

    best_genome = None
    n_qubits = args.n_qubits
    best_metrics = {}
    best_score = 0.0

    if args.animate_only:
        if not os.path.isfile(genome_path):
            parser.error(f"--animate-only needs {genome_path}")
        best_genome, n_qubits = load_genome(genome_path)
        print(f"Loaded genome from {genome_path} (N={n_qubits})")
    elif genome_path:
        best_genome, n_qubits = load_genome(genome_path)
        print(f"Loaded genome from {genome_path} (N={n_qubits})")
    else:
        save_every_gen = SAVE_EVERY_GENERATION and not args.no_save_every_gen
        best_score, best_metrics, best_genome = evolve(
            n_qubits=args.n_qubits,
            pop_size=args.pop_size,
            generations=args.generations,
            n_terms=args.n_terms,
            out_dir=args.out_dir,
            save_every_gen=save_every_gen,
            dissipation_gamma=args.gamma,
        )
        n_qubits = args.n_qubits
        print(
            f"\nBest: fitness={best_score:.4f}, "
            f"S(v/r)={best_metrics['s_vacuum']:.3f}/{best_metrics['s_random']:.3f}, "
            f"flow(v/r)={best_metrics['flow_vacuum']:.3f}/{best_metrics['flow_random']:.3f}"
        )
        if not save_every_gen:
            save_genome_checkpoint(
                best_genome,
                n_qubits,
                args.out_dir,
                "regulated",
                best_score,
                best_metrics,
                args.generations - 1,
            )
            print(f"Wrote {os.path.join(args.out_dir, 'critical_hamiltonian.json')}")

    if not args.animate_only and not args.evolve_only:
        build_report(
            best_genome,
            n_qubits,
            out_dir=args.out_dir,
            train_metrics=best_metrics if best_metrics else None,
        )

    if args.animate or args.animate_only:
        if best_genome is None:
            parser.error("--animate requires a genome")
        save_anim = args.save_animation
        if not save_anim and args.animate_only:
            save_anim = os.path.join(args.out_dir, "field.gif")
        show = args.show_plots or not save_anim
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
            drive_amp=args.drive_amp,
            drive_omega=args.drive_omega,
            dissipation_gamma=args.gamma,
        )


if __name__ == "__main__":
    main()
