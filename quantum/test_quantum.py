import math
import unittest
from unittest.mock import patch

import torch

from quantum import config
from quantum.device import get_device, get_dtype
from quantum.evolution import init_population, train, training_psi_initial
from quantum.physics import (
    balance_fitness,
    chaos_signal_from_r,
    critical_edge_initial_state,
    critical_entropy_target,
    mean_level_spacing_ratio,
    order_signal_from_r,
    rollout_fitness,
    rollout_fitness_batch,
    r_critical_target,
    static_hamiltonian_fitness,
    track_criticality,
    volume_law_entropy_bound,
)

device = get_device()
dtype = get_dtype()
n = 4
g = 0.1
dt = 0.01


class TestBalanceFitness(unittest.TestCase):
    def test_peaks_when_equal(self):
        self.assertAlmostEqual(balance_fitness(0.5, 0.5), 0.25, places=5)

    def test_penalizes_dominance(self):
        self.assertAlmostEqual(balance_fitness(1.0, 0.0), -1.0, places=5)
        self.assertAlmostEqual(balance_fitness(0.0, 1.0), -1.0, places=5)

    def test_r_signals_at_limits(self):
        self.assertAlmostEqual(order_signal_from_r(config.R_POISSON), 1.0, places=4)
        self.assertAlmostEqual(chaos_signal_from_r(config.R_GOE), 1.0, places=4)

    def test_r_critical_midpoint(self):
        self.assertAlmostEqual(
            r_critical_target(), 0.5 * (config.R_POISSON + config.R_GOE), places=5
        )


class TestLevelSpacingRatio(unittest.TestCase):
    def test_poisson_like_diagonal(self):
        eigs = torch.arange(5, dtype=torch.float64)
        r = mean_level_spacing_ratio(eigs)
        self.assertAlmostEqual(r, 1.0, places=4)


class TestCriticalEdgeInitial(unittest.TestCase):
    def test_product_zero_entanglement(self):
        from quantum.physics import reduced_entropy_from_psi

        psi = critical_edge_initial_state(n, dtype, device, 42)
        self.assertAlmostEqual(reduced_entropy_from_psi(psi, n).item(), 0.0, places=4)

    def test_matches_training_initial(self):
        psi = training_psi_initial(n, dtype, device)
        ref = critical_edge_initial_state(n, dtype, device, config.DISORDER_SEED)
        self.assertTrue(torch.allclose(psi, ref))


class TestRolloutFitness(unittest.TestCase):
    def test_zero_steps_is_static_only(self):
        M = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        psi = critical_edge_initial_state(n, dtype, device, 0)
        total, _ = rollout_fitness(M, psi, n, 0, g, dt, device)
        f_static, _, _, _ = static_hamiltonian_fitness(M, n, dtype, device)
        self.assertAlmostEqual(total, f_static, places=5)

    def test_batch_matches_single(self):
        pop = init_population(2, n, dtype, device)
        psi = critical_edge_initial_state(n, dtype, device, 3)
        M_batch = torch.stack(pop)
        psi_batch = psi.unsqueeze(0).expand(2, -1).clone()
        batch_scores, _ = rollout_fitness_batch(M_batch, psi_batch, n, 2, g, dt, device)
        for i in range(2):
            single, _ = rollout_fitness(pop[i], psi, n, 2, g, dt, device)
            self.assertAlmostEqual(batch_scores[i].item(), single, places=4)


class TestTrackCriticality(unittest.TestCase):
    def test_keys(self):
        M = init_population(1, n, dtype, device)[0]
        psi = critical_edge_initial_state(n, dtype, device, 1)
        metrics = track_criticality(psi, M, n, dtype, device)
        for key in (
            "entanglement",
            "r_mean",
            "r_star",
            "f_static",
            "f_dynamic",
            "f_order_static",
            "f_chaos_static",
        ):
            self.assertIn(key, metrics)


class TestTrain(unittest.TestCase):
    def test_short_run(self):
        with patch("quantum.evolution.print"):
            history, _ = train(
                n=n,
                population_size=4,
                n_generations=2,
                steps_per_eval=5,
                device=device,
                dtype=dtype,
            )
        self.assertEqual(len(history), 2)
        self.assertIn("f_static", history[0])
        self.assertIn("r_mean", history[0])


class TestEntropyTarget(unittest.TestCase):
    def test_s_star_half_log_n(self):
        self.assertAlmostEqual(
            critical_entropy_target(n), 0.5 * volume_law_entropy_bound(n), places=5
        )


if __name__ == "__main__":
    unittest.main()
