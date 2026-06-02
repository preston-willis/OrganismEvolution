import math

import matplotlib.pyplot as plt

from quantum import config
from quantum.viz import (
    area_law_s,
    critical_s,
    entropy_trajectory,
    joint_energy_rgb_numpy,
    joint_probability_grid_numpy,
    prob_panel_labels,
    volume_law_s,
)


class QuantumGrapher:
    def __init__(self, n):
        plt.ion()
        self.n = n
        self.s_star = critical_s(n)
        self.s_area = area_law_s()
        self.s_volume = volume_law_s(n)
        self.vis_mode = config.VIS_MODE
        self._prob_title, self._prob_xlabel, self._prob_ylabel = prob_panel_labels(
            self.vis_mode
        )

        self.fig = plt.figure(num="Quantum critical edge", figsize=(14, 10))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])
        gs_heatmaps = gs[0, :].subgridspec(1, 2)
        self.ax_prob = self.fig.add_subplot(gs_heatmaps[0, 0])
        self.ax_energy = self.fig.add_subplot(gs_heatmaps[0, 1])
        self.ax_fitness = self.fig.add_subplot(gs[1, 0])
        self.ax_entropy = self.fig.add_subplot(gs[1, 1])
        self.ax_rollout = self.fig.add_subplot(gs[1, 2])

    def update_generation(
        self,
        generation,
        best_fitness,
        entanglement,
        history,
        M,
        psi_initial,
        psi_final,
        steps,
        g,
        dt,
        device,
    ):
        gens = [entry["generation"] for entry in history]

        self.vis_mode = config.VIS_MODE
        self._prob_title, self._prob_xlabel, self._prob_ylabel = prob_panel_labels(
            self.vis_mode
        )

        prob = joint_probability_grid_numpy(psi_final, self.n, self.vis_mode)
        prob_vmax = float(prob.max())
        if prob_vmax < 1e-12:
            prob_vmax = 1.0
        self.ax_prob.clear()
        self.ax_prob.set_title(f"Gen {generation}: {self._prob_title}")
        self.ax_prob.imshow(
            prob, origin="lower", cmap="magma", aspect="equal", vmin=0.0, vmax=prob_vmax
        )
        self.ax_prob.set_xlabel(self._prob_xlabel)
        self.ax_prob.set_ylabel(self._prob_ylabel)

        rgb = joint_energy_rgb_numpy(M, psi_final, self.n, self.vis_mode)
        self.ax_energy.clear()
        self.ax_energy.set_title(f"Gen {generation}: H drive (vacuum A | disorder B)")
        self.ax_energy.imshow(rgb, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
        self.ax_energy.set_xlabel(self._prob_xlabel)
        self.ax_energy.set_ylabel(self._prob_ylabel)

        self.ax_fitness.clear()
        self.ax_fitness.set_title(f"Generation {generation}: criticality fitness")
        self.ax_fitness.set_xlabel("generation")
        self.ax_fitness.set_ylabel("fitness")
        self.ax_fitness.plot(
            gens, [entry["best_fitness"] for entry in history], marker="o"
        )

        self.ax_entropy.clear()
        self.ax_entropy.set_title("S vs generation (area ↔ volume law)")
        self.ax_entropy.set_xlabel("generation")
        self.ax_entropy.set_ylabel("S")
        self.ax_entropy.plot(
            gens,
            [entry["entanglement"] for entry in history],
            marker="o",
            label="S(final)",
        )
        if history and "r_mean" in history[0]:
            self.ax_entropy.plot(
                gens,
                [entry["r_mean"] for entry in history],
                marker="s",
                color="tab:purple",
                label="r(H)",
            )
        if history and "entanglement_initial" in history[0]:
            self.ax_entropy.plot(
                gens,
                [entry["entanglement_initial"] for entry in history],
                marker="x",
                color="tab:gray",
                label="S(t=0)",
            )
        self.ax_entropy.axhline(
            self.s_area,
            color="tab:blue",
            linestyle=":",
            label="area law",
        )
        self.ax_entropy.axhline(
            self.s_star,
            color="tab:green",
            linestyle="--",
            label=rf"$S^*={self.s_star:.3f}$",
        )
        self.ax_entropy.axhline(
            self.s_volume,
            color="tab:red",
            linestyle=":",
            label="volume law",
        )
        self.ax_entropy.legend(loc="best")

        s_vals_a, s_vals_b = entropy_trajectory(
            M, psi_initial, self.n, steps, g, dt, device
        )
        self.ax_rollout.clear()
        self.ax_rollout.set_title(f"Gen {generation}: S(t) rollout")
        self.ax_rollout.set_xlabel("step")
        self.ax_rollout.set_ylabel("S")
        steps_x = range(len(s_vals_a))
        self.ax_rollout.plot(steps_x, s_vals_a, color="tab:blue", label="S(A)")
        self.ax_rollout.plot(steps_x, s_vals_b, color="tab:red", label="S(B)")
        self.ax_rollout.axhline(self.s_area, color="tab:blue", linestyle=":")
        self.ax_rollout.axhline(self.s_star, color="tab:green", linestyle="--")
        self.ax_rollout.axhline(self.s_volume, color="tab:red", linestyle=":")
        self.ax_rollout.legend(loc="best")

        try:
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass
