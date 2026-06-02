import json
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from dynamics import run_rollout
from hamiltonian import build_hamiltonian
from critical_report import pad_genome


def load_genome(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    genome = [(float(c), s) for c, s in data["genome"]]
    return genome, data["n_qubits"]


def run_animation(
    genome,
    n_qubits,
    dt,
    n_steps,
    initial,
    seed,
    show,
    save_path,
    frame_ms,
    drive_amp=0.0,
    drive_omega=0.0,
):
    genome = pad_genome(genome, n_qubits)
    rollout = run_rollout(
        genome,
        n_qubits,
        dt,
        n_steps,
        initial,
        seed,
        drive_amp=drive_amp,
        drive_omega=drive_omega,
    )
    grids = rollout["grids"]
    entropies = rollout["entropies"]
    n_a = rollout["n_a"]
    n_b = n_qubits - n_a
    steps = list(range(len(entropies)))

    vmax = max(np.max(g) for g in grids)
    if vmax < 1e-12:
        vmax = 1.0

    fig, (ax_field, ax_s) = plt.subplots(1, 2, figsize=(10, 4))
    if drive_amp != 0.0:
        title = (
            f"Critical H + drive  "
            f"A={drive_amp} cos({drive_omega}t) sum X_i"
        )
    else:
        title = "Critical Hamiltonian — field self-organization"
    fig.suptitle(title, fontsize=11)

    im = ax_field.imshow(
        grids[0],
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    ax_field.set_xlabel(f"subsystem B ({n_b} qubits)")
    ax_field.set_ylabel(f"subsystem A ({n_a} qubits)")
    ax_field.set_title("|ψ|² on A⊗B")
    fig.colorbar(im, ax=ax_field, fraction=0.046, label="|ψ|²")

    (line_s,) = ax_s.plot([], [], color="C1", linewidth=2)
    ax_s.set_xlim(0, n_steps)
    s_max = max(entropies)
    ax_s.set_ylim(0, s_max * 1.1 if s_max > 1e-12 else 1.0)
    ax_s.set_xlabel("step")
    ax_s.set_ylabel("S (entanglement)")
    ax_s.set_title("Bipartite entanglement entropy")
    ax_s.grid(True, alpha=0.3)

    time_text = ax_field.text(
        0.02,
        0.98,
        "",
        transform=ax_field.transAxes,
        va="top",
        fontsize=9,
        color="white",
    )

    def update(frame):
        im.set_data(grids[frame])
        line_s.set_data(steps[: frame + 1], entropies[: frame + 1])
        time_text.set_text(f"t = {frame * dt:.3f}")
        return im, line_s, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(grids),
        interval=frame_ms,
        repeat=True,
        blit=False,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ani.save(save_path, writer=animation.PillowWriter(fps=1000 // frame_ms))
        print(f"Wrote animation {save_path}")

    if show:
        print("Close the window to exit.")
        plt.show()
    else:
        plt.close(fig)

    return rollout


def main_from_cli(genome_path, dt, n_steps, initial, seed, show, save_path, frame_ms):
    genome, n_qubits = load_genome(genome_path)
    genome = pad_genome(genome, n_qubits)
    eigs = np.linalg.eigvalsh(build_hamiltonian(genome, n_qubits))
    span = float(eigs[-1] - eigs[0])
    print(f"Loaded {genome_path} | N={n_qubits} | spectral span ≈ {span:.3f}")
    print(f"Rollout: dt={dt} steps={n_steps} initial={initial}")
    run_animation(
        genome,
        n_qubits,
        dt,
        n_steps,
        initial,
        seed,
        show,
        save_path,
        frame_ms,
    )
