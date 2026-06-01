import math

import matplotlib.pyplot as plt

from quantum.viz import (
    basin_threshold_s,
    energy_organization_rgb_numpy,
    entropy_trajectory,
    ideal_s,
    probability_grid_numpy,
)


class QuantumGrapher:
    def __init__(self, n):
        plt.ion()
        self.n = n
        self.s_star = ideal_s(n)
        self.s_max = math.log(n)
        self.s_thresh = basin_threshold_s(n)

        self.fig = plt.figure(num="Quantum Coevolution", figsize=(14, 10))
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
        M_A,
        M_B,
        psi_initial,
        psi_final,
        steps,
        g,
        dt,
        device,
    ):
        gens = [entry["generation"] for entry in history]

        prob = probability_grid_numpy(psi_final, self.n)
        prob_vmax = float(prob.max())
        if prob_vmax < 1e-12:
            prob_vmax = 1.0
        self.ax_prob.clear()
        self.ax_prob.set_title(f"Gen {generation}: |ψ(x,y)|²")
        self.ax_prob.imshow(
            prob, origin="lower", cmap="magma", aspect="equal", vmin=0.0, vmax=prob_vmax
        )
        self.ax_prob.set_xlabel("y")
        self.ax_prob.set_ylabel("x")

        rgb = energy_organization_rgb_numpy(M_A, M_B, psi_final, self.n)
        self.ax_energy.clear()
        self.ax_energy.set_title(f"Gen {generation}: A (red) · B (blue) drive")
        self.ax_energy.imshow(rgb, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
        self.ax_energy.set_xlabel("y")
        self.ax_energy.set_ylabel("x")

        self.ax_fitness.clear()
        self.ax_fitness.set_title(f"Generation {generation}: best fitness")
        self.ax_fitness.set_xlabel("generation")
        self.ax_fitness.set_ylabel("fitness")
        self.ax_fitness.plot(
            gens, [entry["best_fitness"] for entry in history], marker="o"
        )

        self.ax_entropy.clear()
        self.ax_entropy.set_title("Best entanglement S")
        self.ax_entropy.set_xlabel("generation")
        self.ax_entropy.set_ylabel("S")
        self.ax_entropy.plot(
            gens, [entry["entanglement"] for entry in history], marker="o"
        )
        self.ax_entropy.axhline(
            self.s_star,
            color="tab:green",
            linestyle="--",
            label=rf"$S^*={self.s_star:.3f}$",
        )
        self.ax_entropy.legend(loc="best")

        s_vals_a, s_vals_b = entropy_trajectory(
            M_A, M_B, psi_initial, self.n, steps, g, dt, device
        )
        self.ax_rollout.clear()
        self.ax_rollout.set_title(f"Gen {generation}: S(t) rollout")
        self.ax_rollout.set_xlabel("step")
        self.ax_rollout.set_ylabel("S")
        steps_x = range(1, len(s_vals_a) + 1)
        self.ax_rollout.plot(steps_x, s_vals_a, color="tab:blue", label="S(A)")
        self.ax_rollout.plot(steps_x, s_vals_b, color="tab:red", label="S(B)")
        self.ax_rollout.legend(loc="best")
        self.ax_rollout.axhline(self.s_star, color="tab:green", linestyle="--")
        self.ax_rollout.axhline(self.s_thresh, color="tab:orange", linestyle=":")
        self.ax_rollout.axhline(self.s_max, color="tab:red", linestyle=":", alpha=0.4)

        try:
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass
