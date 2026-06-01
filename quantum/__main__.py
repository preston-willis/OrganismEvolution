import argparse

from quantum.basin import basin_size_experiment
from quantum.config import BASIN_N_VALUES, N, N_GENERATIONS, TRAIN_HEADLESS
from quantum.evolution import train
from quantum.grapher import QuantumGrapher
from quantum.sim import run_loaded_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Quantum coevolution simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  --train              Run neuroevolution training
  --graph              Show matplotlib graphs during --train
  --load               With --train: resume training. Alone: animate loaded best organism
  --basin              Basin-of-attraction scaling experiment
        """,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run neuroevolution training",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Show matplotlib graphs during --train",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load latest checkpoint; with --train resume, without --train run rollout viewer",
    )
    parser.add_argument(
        "--basin",
        action="store_true",
        help="Run basin-size experiment",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=N_GENERATIONS,
        help="training generations",
    )
    args = parser.parse_args()

    if args.basin:
        basin_size_experiment(n_values=BASIN_N_VALUES)
        return

    if args.load and not args.train:
        run_loaded_simulation()
        return

    if args.train:
        use_graph = (not TRAIN_HEADLESS) or args.graph
        if TRAIN_HEADLESS and not args.graph:
            print("Headless training mode (no matplotlib graphs)")
        elif args.graph:
            print("Training with matplotlib graphs")
        grapher = QuantumGrapher(N) if use_graph else None
        train(n_generations=args.generations, grapher=grapher, load=args.load)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
