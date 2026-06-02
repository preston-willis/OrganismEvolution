import argparse

from quantum.config import N, N_GENERATIONS, TRAIN_HEADLESS
from quantum.evolution import train
from quantum.grapher import QuantumGrapher
from quantum.sim import run_loaded_simulation, run_physics_demo


def main():
    parser = argparse.ArgumentParser(
        description="Evolve Hamiltonians at the quantum critical edge (vacuum ⊗ disorder)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  --train              Evolve H to sustain entanglement between area and volume law
  --graph              Live matplotlib during --train
  --load               Resume training, or animate best checkpoint
  --demo               Rabi oscillation sanity check (fixed physics)
        """,
    )
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--graph", action="store_true", help="Show graphs during --train")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load latest checkpoint; with --train resume, else rollout viewer",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Animate Rabi demo (fixed physics, not the training objective)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=N_GENERATIONS,
        help="training generations",
    )
    args = parser.parse_args()

    if args.demo:
        run_physics_demo()
        return

    if args.load and not args.train:
        run_loaded_simulation()
        return

    if args.train:
        use_graph = (not TRAIN_HEADLESS) or args.graph
        if TRAIN_HEADLESS and not args.graph:
            print("Headless training (quantum critical edge)")
        elif args.graph:
            print("Training with graphs")
        grapher = QuantumGrapher(N) if use_graph else None
        train(n_generations=args.generations, grapher=grapher, load=args.load)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
