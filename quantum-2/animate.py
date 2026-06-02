import json
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from config import S_TARGET, TRANSFER_LAMBDA
from dynamics import run_rollout, transfer_metrics_from_grid
from hamiltonian import pad_genome


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
    drive_amp,
    drive_omega,
    dissipation_gamma,
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
        dissipation_gamma=dissipation_gamma,
    )
    grids = rollout["grids"]
    entropies = rollout["entropies"]
    n_a = rollout["n_a"]
    n_b = n_qubits - n_a
    steps = list(range(len(entropies)))

    transfers = [
        transfer_metrics_from_grid(g, TRANSFER_LAMBDA)["transfer"] for g in grids
    ]

    if initial == "vacuum" and len(grids) > 1:
        pooled = np.concatenate([g.ravel() for g in grids[1:]])
        vmax = float(np.percentile(pooled, 99.5))
        vmax = max(vmax, float(np.max(grids[-1])), 1e-12)
    else:
        vmax = max(float(np.max(g)) for g in grids)
    if vmax < 1e-12:
        vmax = 1.0

    final_tm = transfer_metrics_from_grid(grids[-1], TRANSFER_LAMBDA)
    print(
        f"Heatmap vmax={vmax:.4f} | final S={entropies[-1]:.4f} | "
        f"p_B={final_tm['p_b_exc']:.4f} p_A={final_tm['p_a_exc']:.4f} "
        f"transfer={final_tm['transfer']:.4f}"
    )

    fig, (ax_field, ax_s) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"H + B-pump A={drive_amp} cos({drive_omega}t) | gamma={dissipation_gamma} on A",
        fontsize=11,
    )

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
    ax_field.set_title("|ρ|² on A⊗B")
    fig.colorbar(im, ax=ax_field, fraction=0.046, label="diag ρ")

    (line_s,) = ax_s.plot([], [], color="C1", linewidth=2, label="S")
    (line_t,) = ax_s.plot([], [], color="C0", linewidth=1.5, label="transfer")
    ax_s.axhline(S_TARGET, color="C1", linestyle="--", alpha=0.5, linewidth=1)
    ax_s.set_xlim(0, n_steps)
    y_hi = max(max(entropies), max(transfers), S_TARGET) * 1.1
    if y_hi < 1e-12:
        y_hi = 1.0
    ax_s.set_ylim(0, y_hi)
    ax_s.set_xlabel("step")
    ax_s.set_ylabel("S / transfer")
    ax_s.set_title("Entanglement & transfer")
    ax_s.legend(loc="lower right", fontsize=8)
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
        line_t.set_data(steps[: frame + 1], transfers[: frame + 1])
        time_text.set_text(f"t = {frame * dt:.3f}")
        return im, line_s, line_t, time_text

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
