import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DRIVE_AMP, DRIVE_OMEGA, DISSIPATION_GAMMA, DT, S_TARGET
from dissipation import lowering_operator
from dynamics import run_rollout, transfer_metrics_from_grid
from fitness import evaluate_genome, regulated_fitness


class TestRegulated(unittest.TestCase):
    def test_lowering_maps_excited_to_ground(self):
        n = 2
        L = lowering_operator(n, 0)
        one = np.zeros(2**n, dtype=complex)
        one[1] = 1.0
        out = L @ one
        self.assertAlmostEqual(abs(out[0]), 1.0)
        self.assertAlmostEqual(abs(out[1]), 0.0)

    def test_open_rollout_trace_one(self):
        genome = [(0.3, "XXII"), (0.2, "ZZII")]
        rollout = run_rollout(
            genome,
            4,
            DT,
            20,
            "vacuum",
            0,
            drive_amp=DRIVE_AMP,
            drive_omega=DRIVE_OMEGA,
            dissipation_gamma=DISSIPATION_GAMMA,
        )
        self.assertEqual(len(rollout["rhos"]), 21)
        tr = np.trace(rollout["rhos"][-1])
        self.assertAlmostEqual(float(np.real(tr)), 1.0, places=5)

    def test_regulated_fitness_finite(self):
        genome = [(0.5, "XXXX")]
        score, metrics = regulated_fitness(
            genome,
            4,
            n_steps=40,
            window_start=25,
            window_end=40,
            drive_amp=DRIVE_AMP,
        )
        self.assertTrue(np.isfinite(score))
        self.assertIn("mean_s", metrics)
        score, combined = evaluate_genome(genome, 4, drive_amp=DRIVE_AMP)
        self.assertTrue(np.isfinite(score))
        self.assertIn("flow_vacuum", combined)
        self.assertIn("flow_random", combined)

    def test_product_state_zero_entropy(self):
        grid = np.zeros((4, 4))
        grid[0, 1] = 0.5
        grid[0, 2] = 0.5
        m = transfer_metrics_from_grid(grid, 0.5)
        self.assertAlmostEqual(m["p_a_exc"], 0.0)
        self.assertAlmostEqual(m["p_b_exc"], 1.0)


if __name__ == "__main__":
    unittest.main()
