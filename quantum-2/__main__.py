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
    TASK,
    TRAIN_DRIVE,
    SAVE_EVERY_GENERATION,
)
from animate import run_animation
from checkpoint import save_genome_checkpoint
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


def _print_best(task, best_score, best_metrics):
    if task == "transfer":
        print(
            f"\nBest: fitness={best_score:.4f}, "
            f"transfer={best_metrics['transfer']:.4f}, "
            f"p_B={best_metrics['p_b_exc']:.4f}, p_A={best_metrics['p_a_exc']:.4f}"
        )
        return
    if task == "spacing":
        r = best_metrics.get("r")
        if r is None:
            print(f"\nBest: fitness={best_score:.2f}, r=invalid")
        else:
            print(f"\nBest: fitness={best_score:.2f}, r={r:.4f}")
        return
    r = best_metrics.get("r")
    r_str = f"{r:.4f}" if r is not None else "invalid"
    print(
        f"\nBest: fitness={best_score:.4f}, "
        f"transfer={best_metrics.get('transfer', 0):.4f}, r={r_str}"
    )


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
        "--task",
        choices=["spacing", "transfer", "combined"],
        default=TASK,
        help="spacing=r only; transfer=A→B excitation; combined=both",
    )
    parser.add_argument(
        "--no-train-drive",
        action="store_true",
        help="Disable AC drive during transfer/combined training rollouts",
    )
    parser.add_argument(
        "--no-save-every-gen",
        action="store_true",
        help="Only save checkpoint at end of training (not each generation)",
    )
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
        help="Add AC drive in animation (and training unless --no-train-drive)",
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

    train_drive = TRAIN_DRIVE and not args.no_train_drive
    if args.task in ("transfer", "combined") and train_drive:
        print(f"Training task={args.task} with drive A={DRIVE_AMP} omega={DRIVE_OMEGA}")

    best_genome = None
    n_qubits = args.n_qubits
    best_metrics = {}
    best_score = 0.0

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
        save_every_gen = SAVE_EVERY_GENERATION and not args.no_save_every_gen
        best_score, best_metrics, best_genome = evolve(
            n_qubits=args.n_qubits,
            pop_size=args.pop_size,
            generations=args.generations,
            n_terms=args.n_terms,
            task=args.task,
            train_drive=train_drive,
            out_dir=args.out_dir,
            save_every_gen=save_every_gen,
        )
        n_qubits = args.n_qubits
        _print_best(args.task, best_score, best_metrics)
        if not save_every_gen:
            save_genome_checkpoint(
                best_genome,
                n_qubits,
                args.out_dir,
                args.task,
                best_score,
                best_metrics,
                args.generations - 1,
            )
            print(f"Wrote {os.path.join(args.out_dir, 'critical_hamiltonian.json')}")

    if not args.animate_only and not args.evolve_only:
        train_metrics = {
            "task": args.task,
            "train_drive": train_drive,
            **best_metrics,
        }
        build_report(
            best_genome,
            n_qubits,
            scale_sizes=scale_sizes,
            out_dir=args.out_dir,
            show_plots=args.show_plots,
            train_metrics=train_metrics if best_metrics else None,
        )

    if args.animate or args.animate_only:
        if best_genome is None:
            parser.error("--animate requires a genome (run evolution or --load-genome)")
        save_anim = args.save_animation
        if not save_anim and args.animate_only:
            save_anim = os.path.join(args.out_dir, "field.gif")
        show = args.show_plots or not save_anim
        use_drive = args.drive
        if args.task in ("transfer", "combined") and train_drive and not args.drive:
            use_drive = True
        drive_amp = args.drive_amp if use_drive else 0.0
        drive_omega = args.drive_omega
        if use_drive:
            print(f"Animation drive: A={drive_amp} omega={drive_omega}")
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
