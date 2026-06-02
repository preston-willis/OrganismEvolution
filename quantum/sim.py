import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from quantum.checkpoint import load_latest_checkpoint
from quantum import config
from quantum.config import DEMO_DT, DEMO_G, DEMO_STEPS, N, SIM_FRAME_MS, STEPS_PER_EVAL
from quantum.device import get_device, get_dtype
from quantum.evolution import training_psi_initial
from quantum.physics import evolution_step, rabi_excitation_product_state, zero_hamiltonian
from quantum.viz import (
    area_law_s,
    critical_s,
    entropy_trajectory,
    excitation_probability_trajectory,
    joint_energy_rgb_numpy,
    joint_probability_grid_numpy,
    prob_panel_labels,
    volume_law_s,
)


class QuantumSimGrapher:
    def __init__(self, n, frame_ms=SIM_FRAME_MS):
        self.n = n
        self.frame_ms = frame_ms
        self.s_star = critical_s(n)
        self.s_area = area_law_s()
        self.s_volume = volume_law_s(n)

        self.fig = plt.figure(num="Quantum critical edge", figsize=(12, 8))
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
        self.rollout_metric = "entanglement"
        self.energy_title = "H drive"
        self.vis_mode = config.VIS_MODE
        self._prob_title, self._prob_xlabel, self._prob_ylabel = prob_panel_labels(
            self.vis_mode
        )

    def _compute_trajectory(self, M, psi_initial, steps, g, dt, device):
        states = []
        state = psi_initial.clone()
        for step in range(steps + 1):
            states.append(state.detach().cpu())
            if step == steps:
                break
            _, state = evolution_step(M, state, self.n, g, dt, device)

        if self.rollout_metric == "excitation":
            s_vals_a, s_vals_b = excitation_probability_trajectory(
                M, psi_initial, self.n, steps, g, dt, device
            )
        else:
            s_a, s_b = entropy_trajectory(M, psi_initial, self.n, steps, g, dt, device)
            s_vals_a = [0.0] + s_a
            s_vals_b = [0.0] + s_b

        frames_prob = [
            joint_probability_grid_numpy(psi, self.n, self.vis_mode)
            for psi in states
        ]
        frames_rgb = [
            joint_energy_rgb_numpy(M, psi, self.n, self.vis_mode)
            for psi in states
        ]
        return frames_prob, frames_rgb, s_vals_a, s_vals_b

    def _setup_rollout_axes(self):
        self.ax_rollout.clear()
        if self.rollout_metric == "excitation":
            self.ax_rollout.set_title("P(excitation) vacuum A vs disorder B")
            self.ax_rollout.set_ylabel("P(excitation)")
            label_a = "P(exc) A"
            label_b = "P(exc) B"
        else:
            self.ax_rollout.set_title("S(t): area law ↔ critical S* ↔ volume law")
            self.ax_rollout.set_ylabel("S")
            self.ax_rollout.axhline(self.s_area, color="tab:blue", linestyle=":")
            self.ax_rollout.axhline(self.s_star, color="tab:green", linestyle="--")
            self.ax_rollout.axhline(self.s_volume, color="tab:red", linestyle=":")
            label_a = "S(A)"
            label_b = "S(B)"
        self.ax_rollout.set_xlabel("step")

        self._line_s_a, = self.ax_rollout.plot(
            [],
            [],
            color="tab:blue",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label_a,
        )
        self._line_s_b, = self.ax_rollout.plot(
            [],
            [],
            color="tab:red",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label_b,
        )
        self.ax_rollout.legend(loc="best")

        all_s = list(self.s_vals_a) + list(self.s_vals_b)
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
        self.ax_prob.set_title(f"Step {frame}/{self.steps}: {self._prob_title}")
        self.ax_prob.imshow(
            prob, origin="lower", cmap="magma", aspect="equal", vmin=0.0, vmax=prob_vmax
        )
        self.ax_prob.set_xlabel(self._prob_xlabel)
        self.ax_prob.set_ylabel(self._prob_ylabel)

        rgb = self.frames_rgb[frame]
        self.ax_energy.clear()
        self.ax_energy.set_title(f"Step {frame}/{self.steps}: {self.energy_title}")
        self.ax_energy.imshow(rgb, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
        self.ax_energy.set_xlabel(self._prob_xlabel)
        self.ax_energy.set_ylabel(self._prob_ylabel)

        return self._line_s_a, self._line_s_b, self._cursor

    def run_rollout(
        self,
        M,
        psi_initial,
        steps,
        g,
        dt,
        device,
        rollout_metric=None,
        energy_title=None,
        vis_mode=None,
    ):
        if rollout_metric is not None:
            self.rollout_metric = rollout_metric
        if energy_title is not None:
            self.energy_title = energy_title
        if vis_mode is not None:
            self.vis_mode = vis_mode
            self._prob_title, self._prob_xlabel, self._prob_ylabel = prob_panel_labels(
                self.vis_mode
            )
        self.steps = steps
        print("Computing rollout...")
        (
            self.frames_prob,
            self.frames_rgb,
            self.s_vals_a,
            self.s_vals_b,
        ) = self._compute_trajectory(M, psi_initial, steps, g, dt, device)
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
    M = ckpt["best_M"]
    psi_initial = training_psi_initial(n, dtype, device)

    print(f"Loaded checkpoint: {ckpt['path']}")
    print(
        f"Generation {ckpt['generation']} | n={n} | "
        f"VIS_MODE={config.VIS_MODE} | {steps} rollout steps"
    )

    grapher = QuantumSimGrapher(n)
    grapher.run_rollout(M, psi_initial, steps, g, dt, device)


def run_physics_demo():
    device = get_device()
    dtype = get_dtype()
    n = N
    g = DEMO_G
    dt = DEMO_DT
    steps = DEMO_STEPS
    M = zero_hamiltonian(n, dtype, device)
    psi_initial = rabi_excitation_product_state(n, dtype, device)

    print("Physics demo: Rabi |1,0> <-> |0,1> (H=0, beam coupling only)")
    print(f"n={n} G={g} dt={dt} steps={steps} VIS_MODE={config.VIS_MODE}")

    grapher = QuantumSimGrapher(n)
    grapher.run_rollout(
        M,
        psi_initial,
        steps,
        g,
        dt,
        device,
        rollout_metric="excitation",
        energy_title="mode marginals",
    )
