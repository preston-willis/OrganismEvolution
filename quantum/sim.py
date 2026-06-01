import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from quantum.checkpoint import load_latest_checkpoint
from quantum.config import INITIAL_STATE, SIM_FRAME_MS, STEPS_PER_EVAL
from quantum.device import get_device, get_dtype
from quantum.physics import evolution_step, initial_product_state, reduced_entropy_from_psi
from quantum.viz import (
    basin_threshold_s,
    energy_organization_rgb_numpy,
    ideal_s,
    probability_grid_numpy,
)


class QuantumSimGrapher:
    def __init__(self, n, frame_ms=SIM_FRAME_MS):
        self.n = n
        self.frame_ms = frame_ms
        self.s_star = ideal_s(n)
        self.s_max = math.log(n)
        self.s_thresh = basin_threshold_s(n)

        self.fig = plt.figure(num="Quantum simulation", figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
        self.ax_prob = self.fig.add_subplot(gs[0, 0])
        self.ax_energy = self.fig.add_subplot(gs[0, 1])
        self.ax_rollout = self.fig.add_subplot(gs[1, :])

        self.frames_prob = []
        self.frames_rgb = []
        self.s_vals_a = []
        self.s_vals_b = []
        self.steps = 0
        self._line_s_a = None
        self._line_s_b = None
        self._cursor = None

    def _compute_trajectory(self, M_A, M_B, psi_initial, steps, g, dt, device):
        states = []
        s_vals_a = []
        s_vals_b = []
        state = psi_initial.clone()
        for step in range(steps + 1):
            states.append(state.detach().cpu())
            s_vals_a.append(reduced_entropy_from_psi(state, self.n, "A").item())
            s_vals_b.append(reduced_entropy_from_psi(state, self.n, "B").item())
            if step == steps:
                break
            _, state = evolution_step(M_A, M_B, state, self.n, g, dt, device)

        frames_prob = [probability_grid_numpy(psi, self.n) for psi in states]
        frames_rgb = [
            energy_organization_rgb_numpy(M_A, M_B, psi, self.n) for psi in states
        ]
        return frames_prob, frames_rgb, s_vals_a, s_vals_b

    def _setup_rollout_axes(self):
        self.ax_rollout.clear()
        self.ax_rollout.set_title("S(t) subsystems A & B")
        self.ax_rollout.set_xlabel("step")
        self.ax_rollout.set_ylabel("S")
        self.ax_rollout.axhline(self.s_star, color="tab:green", linestyle="--")
        self.ax_rollout.axhline(self.s_thresh, color="tab:orange", linestyle=":")
        self.ax_rollout.axhline(self.s_max, color="tab:red", linestyle=":", alpha=0.4)

        self._line_s_a, = self.ax_rollout.plot(
            [],
            [],
            color="tab:blue",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="S(A)",
        )
        self._line_s_b, = self.ax_rollout.plot(
            [],
            [],
            color="tab:red",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="S(B)",
        )
        self.ax_rollout.legend(loc="best")

        all_s = self.s_vals_a + self.s_vals_b
        s_min = min(all_s)
        s_max = max(all_s)
        pad = max(0.05 * (s_max - s_min), 0.05)
        y_lo = s_min - pad
        y_hi = s_max + pad if s_max > s_min else s_max + 0.1
        self.ax_rollout.set_xlim(0, self.steps)
        self.ax_rollout.set_ylim(y_lo, y_hi)

        self._cursor = self.ax_rollout.axvline(
            0, color="tab:gray", linestyle="-", alpha=0.5, zorder=3
        )

    def _animate(self, frame):
        x = range(frame + 1)
        self._line_s_a.set_data(x, self.s_vals_a[: frame + 1])
        self._line_s_b.set_data(x, self.s_vals_b[: frame + 1])
        self._cursor.set_xdata([frame, frame])
        self._cursor.set_ydata(self.ax_rollout.get_ylim())

        prob = self.frames_prob[frame]
        prob_vmax = float(prob.max())
        if prob_vmax < 1e-12:
            prob_vmax = 1.0
        self.ax_prob.clear()
        self.ax_prob.set_title(f"Step {frame}/{self.steps}: |ψ(x,y)|²")
        self.ax_prob.imshow(
            prob, origin="lower", cmap="magma", aspect="equal", vmin=0.0, vmax=prob_vmax
        )
        self.ax_prob.set_xlabel("y")
        self.ax_prob.set_ylabel("x")

        rgb = self.frames_rgb[frame]
        self.ax_energy.clear()
        self.ax_energy.set_title(f"Step {frame}/{self.steps}: A (red) · B (blue) drive")
        self.ax_energy.imshow(rgb, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
        self.ax_energy.set_xlabel("y")
        self.ax_energy.set_ylabel("x")

        return self._line_s_a, self._line_s_b, self._cursor

    def run_rollout(self, M_A, M_B, psi_initial, steps, g, dt, device):
        self.steps = steps
        print("Computing rollout...")
        self.frames_prob, self.frames_rgb, self.s_vals_a, self.s_vals_b = (
            self._compute_trajectory(M_A, M_B, psi_initial, steps, g, dt, device)
        )
        self._setup_rollout_axes()
        self.fig.tight_layout()

        self._ani = animation.FuncAnimation(
            self.fig,
            self._animate,
            frames=len(self.frames_rgb),
            interval=self.frame_ms,
            repeat=True,
            blit=False,
        )
        print("Close the window to exit.")
        plt.show()


def run_loaded_simulation():
    device = get_device()
    dtype = get_dtype()
    ckpt = load_latest_checkpoint(device, dtype)
    n = ckpt["n"]
    g = ckpt["g"]
    dt = ckpt["dt"]
    steps = STEPS_PER_EVAL
    M_A = ckpt["best_M_A"]
    M_B = ckpt["best_M_B"]

    psi_initial = initial_product_state(n, dtype, device, INITIAL_STATE)

    print(f"Loaded checkpoint: {ckpt['path']}")
    print(
        f"Generation {ckpt['generation']} | n={n} | "
        f"same best organism as --train --graph | {steps} rollout steps"
    )

    grapher = QuantumSimGrapher(n)
    grapher.run_rollout(M_A, M_B, psi_initial, steps, g, dt, device)
