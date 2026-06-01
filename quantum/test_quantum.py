import glob
import math
import os
import unittest
from unittest.mock import patch

import torch

from quantum.device import get_device, get_dtype
from quantum.evolution import (
    best_organism_from_population,
    evaluate_organism,
    evaluate_population,
    init_population,
    run_evolution,
    run_generation,
    train,
)
from quantum.physics import (
    coupling_hamiltonian,
    evolution_step,
    evolution_step_batch,
    hermitian,
    initial_product_state,
    ladder_operators,
    matrix_exp_unitary,
    normalize,
    normalize_batch,
    partial_trace,
    parabolic_entropy,
    random_product_state,
    reduced_entropy_from_psi,
    rollout_fitness,
    rollout_fitness_batch,
    total_hamiltonian,
    track_complexity,
    vacuum_product_state,
    von_neumann_entropy,
)

device = get_device()
dtype = get_dtype()
n = 4
g = 0.1
dt = 0.01


def state_norm(psi):
    return torch.sqrt(torch.sum(torch.abs(psi) ** 2)).item()


class TestNormalize(unittest.TestCase):
    def test_normalize_unit_length(self):
        psi = torch.randn(n, dtype=dtype, device=device)
        out = normalize(psi)
        self.assertAlmostEqual(state_norm(out), 1.0, places=5)

    def test_normalize_batch(self):
        psi = torch.randn(3, n * n, dtype=dtype, device=device)
        out = normalize_batch(psi)
        norms = torch.sqrt(torch.sum(torch.abs(out) ** 2, dim=-1))
        self.assertTrue(torch.allclose(norms, torch.ones(3, device=device), atol=1e-5))


class TestLadderOperators(unittest.TestCase):
    def test_shapes(self):
        a, ad = ladder_operators(n, dtype, device)
        self.assertEqual(a.shape, (n, n))
        self.assertEqual(ad.shape, (n, n))

    def test_adjoint_relation(self):
        a, ad = ladder_operators(n, dtype, device)
        self.assertTrue(torch.allclose(ad, a.conj().mT, atol=1e-5))


class TestCouplingHamiltonian(unittest.TestCase):
    def test_shape_and_hermitian(self):
        H = coupling_hamiltonian(n, g, dtype, device)
        self.assertEqual(H.shape, (n * n, n * n))
        self.assertTrue(torch.allclose(H, H.conj().mT, atol=1e-4))


class TestMatrixExpUnitary(unittest.TestCase):
    def test_unitary(self):
        M_A = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        M_B = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        H_total, _ = total_hamiltonian(M_A, M_B, n, g, dtype, device)
        U = matrix_exp_unitary(H_total, dt)
        I = torch.eye(n * n, dtype=dtype, device=device)
        self.assertTrue(torch.allclose(U @ U.conj().mT, I, atol=1e-3))

    def test_scipy_fallback(self):
        M_A = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        M_B = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        H_total, _ = total_hamiltonian(M_A, M_B, n, g, dtype, device)
        with patch("torch.matrix_exp", side_effect=RuntimeError("fail")):
            U = matrix_exp_unitary(H_total, dt)
        self.assertEqual(U.shape, H_total.shape)


class TestPartialTrace(unittest.TestCase):
    def test_shape(self):
        psi_AB = random_product_state(n, dtype, device)
        rho_AB = torch.outer(psi_AB, psi_AB.conj())
        rho_A = partial_trace(rho_AB, n)
        self.assertEqual(rho_A.shape, (n, n))


class TestVonNeumannEntropy(unittest.TestCase):
    def test_pure_state_zero(self):
        psi = torch.randn(n, dtype=dtype, device=device)
        psi = psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2))
        rho = torch.outer(psi, psi.conj())
        self.assertAlmostEqual(von_neumann_entropy(rho).item(), 0.0, places=4)

    def test_mixed_state_positive(self):
        rho = torch.eye(n, dtype=dtype, device=device) / n
        self.assertGreater(von_neumann_entropy(rho).item(), 0.0)

    def test_tiny_eigenvalues_filtered(self):
        rho = torch.zeros(n, n, dtype=dtype, device=device)
        rho[0, 0] = 1.0
        self.assertAlmostEqual(von_neumann_entropy(rho).item(), 0.0, places=4)


class TestEvolutionStep(unittest.TestCase):
    def test_preserves_norm(self):
        M_A = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        M_B = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        psi_AB = random_product_state(n, dtype, device)
        _, psi_new = evolution_step(M_A, M_B, psi_AB, n, g, dt, device)
        self.assertAlmostEqual(state_norm(psi_new), 1.0, places=4)

    def test_returns_float_score(self):
        M_A = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        M_B = torch.randn(n, n, dtype=dtype, device=device) * 0.1
        psi_AB = vacuum_product_state(n, dtype, device)
        score, _ = evolution_step(M_A, M_B, psi_AB, n, g, dt, device)
        self.assertIsInstance(score, float)


class TestRolloutFitness(unittest.TestCase):
    def test_zero_steps(self):
        M_A, M_B = torch.randn(n, n, dtype=dtype, device=device) * 0.1, torch.randn(
            n, n, dtype=dtype, device=device
        ) * 0.1
        psi = vacuum_product_state(n, dtype, device)
        score, final = rollout_fitness(M_A, M_B, psi, n, 0, g, dt, device)
        self.assertEqual(score, 0.0)
        self.assertTrue(torch.allclose(final, psi))

    def test_multi_step_returns_state(self):
        M_A, M_B = torch.randn(n, n, dtype=dtype, device=device) * 0.1, torch.randn(
            n, n, dtype=dtype, device=device
        ) * 0.1
        psi = vacuum_product_state(n, dtype, device)
        score, final = rollout_fitness(M_A, M_B, psi, n, 3, g, dt, device)
        self.assertIsInstance(score, float)
        self.assertEqual(final.shape, psi.shape)


class TestRolloutFitnessBatch(unittest.TestCase):
    def test_batch_matches_single(self):
        pop = init_population(2, n, dtype, device)
        psi = vacuum_product_state(n, dtype, device)
        M_A = torch.stack([p[0] for p in pop])
        M_B = torch.stack([p[1] for p in pop])
        psi_batch = psi.unsqueeze(0).expand(2, -1).clone()
        batch_scores, _ = rollout_fitness_batch(M_A, M_B, psi_batch, n, 2, g, dt, device)
        for i in range(2):
            single_score, _ = rollout_fitness(pop[i][0], pop[i][1], psi, n, 2, g, dt, device)
            self.assertAlmostEqual(batch_scores[i].item(), single_score, places=4)


class TestTrackComplexity(unittest.TestCase):
    def test_product_state_zero_entanglement(self):
        psi = random_product_state(n, dtype, device)
        M_A, _ = init_population(1, n, dtype, device)[0]
        metrics = track_complexity(psi, M_A, n)
        self.assertAlmostEqual(metrics["entanglement"], 0.0, places=4)

    def test_all_keys_present(self):
        psi = vacuum_product_state(n, dtype, device)
        M_A, _ = init_population(1, n, dtype, device)[0]
        metrics = track_complexity(psi, M_A, n)
        for key in (
            "entanglement",
            "participation_ratio",
            "hamiltonian_structure",
            "eigenvalue_spread",
        ):
            self.assertIn(key, metrics)


class TestInitialProductState(unittest.TestCase):
    def test_vacuum(self):
        psi = vacuum_product_state(n, dtype, device)
        self.assertAlmostEqual(torch.abs(psi[0]).item(), 1.0, places=5)

    def test_random_normalized(self):
        psi = random_product_state(n, dtype, device)
        self.assertAlmostEqual(state_norm(psi), 1.0, places=5)

    def test_initial_modes(self):
        psi_v = initial_product_state(n, dtype, device, "vacuum")
        psi_r = initial_product_state(n, dtype, device, "random")
        self.assertAlmostEqual(state_norm(psi_v), 1.0, places=5)
        self.assertAlmostEqual(state_norm(psi_r), 1.0, places=5)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            initial_product_state(n, dtype, device, "invalid")


class TestReducedEntropy(unittest.TestCase):
    def test_vacuum_zero(self):
        psi = vacuum_product_state(n, dtype, device)
        S = reduced_entropy_from_psi(psi, n)
        self.assertAlmostEqual(S.item(), 0.0, places=4)


class TestEvaluateOrganism(unittest.TestCase):
    def test_zero_steps(self):
        M_A, M_B = init_population(1, n, dtype, device)[0]
        psi = vacuum_product_state(n, dtype, device)
        score, state = evaluate_organism(M_A, M_B, psi, n, 0, g, dt, device)
        self.assertEqual(score, 0.0)
        self.assertTrue(torch.allclose(state, psi))

    def test_accumulates_over_steps(self):
        M_A, M_B = init_population(1, n, dtype, device)[0]
        psi = vacuum_product_state(n, dtype, device)
        score, _ = evaluate_organism(M_A, M_B, psi, n, 3, g, dt, device)
        single, _ = evolution_step(M_A, M_B, psi, n, g, dt, device)
        self.assertNotEqual(score, single)


class TestEvaluatePopulation(unittest.TestCase):
    def test_returns_scores_and_states(self):
        pop = init_population(3, n, dtype, device)
        psi = vacuum_product_state(n, dtype, device)
        scores, states = evaluate_population(pop, psi, n, 2, g, dt, device)
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(states), 3)


class TestBestOrganismFromPopulation(unittest.TestCase):
    def test_picks_highest_score(self):
        pop = init_population(3, n, dtype, device)
        psi = vacuum_product_state(n, dtype, device)
        with patch(
            "quantum.evolution.evaluate_population",
            return_value=([0.1, 0.9, 0.2], [psi, psi, psi]),
        ):
            (M_A, M_B), _, score = best_organism_from_population(
                pop, psi, n, 1, g, dt, device
            )
        self.assertEqual(score, 0.9)
        self.assertIs(pop[1][0], M_A)


class TestRunGeneration(unittest.TestCase):
    def test_refills_population(self):
        pop = init_population(4, n, dtype, device)
        new_pop, _, _, _, _, _, _ = run_generation(pop, n, 2, g, dt, device, dtype)
        self.assertEqual(len(new_pop), 4)

    def test_returns_seven_values(self):
        pop = init_population(4, n, dtype, device)
        result = run_generation(pop, n, 1, g, dt, device, dtype)
        self.assertEqual(len(result), 7)

    def test_survivors_are_top_half(self):
        pop = init_population(4, n, dtype, device)
        scores = [0.0, 3.0, 1.0, 2.0]
        with patch(
            "quantum.evolution.evaluate_population",
            return_value=(scores, [vacuum_product_state(n, dtype, device)] * 4),
        ):
            new_pop, best_fitness, _, _, _, _, _ = run_generation(
                pop, n, 1, g, dt, device, dtype
            )
        self.assertEqual(best_fitness, 3.0)
        self.assertEqual(len(new_pop), 4)


class TestRunEvolution(unittest.TestCase):
    def test_zero_generations_returns_none(self):
        M_A, M_B = init_population(1, n, dtype, device)[0]
        result = run_evolution(M_A, M_B, n, 0, population_size=2)
        self.assertIsNone(result)

    def test_returns_complexity_dict(self):
        M_A, M_B = init_population(1, n, dtype, device)[0]
        result = run_evolution(
            M_A, M_B, n, 1, population_size=2, steps_per_eval=1, device=device, dtype=dtype
        )
        self.assertIn("entanglement", result)


class TestTrain(unittest.TestCase):
    @patch("quantum.evolution.print")
    def test_history_and_logging(self, mock_print):
        history, population = train(
            n=n,
            population_size=4,
            n_generations=3,
            steps_per_eval=1,
            device=device,
            dtype=dtype,
        )
        self.assertEqual(len(history), 3)
        self.assertEqual(len(population), 4)
        self.assertEqual(mock_print.call_count, 3)
        entry = history[0]
        self.assertEqual(entry["generation"], 0)
        self.assertIn("best_fitness", entry)
        self.assertIn("entanglement", entry)
        self.assertIn("participation_ratio", entry)

    @patch("quantum.evolution.print")
    @patch("quantum.evolution.LOG_INTERVAL", 5)
    def test_skips_non_log_generations(self, mock_print):
        train(
            n=n,
            population_size=2,
            n_generations=4,
            steps_per_eval=1,
            device=device,
            dtype=dtype,
        )
        self.assertEqual(mock_print.call_count, 1)

    def test_default_device_and_dtype(self):
        with patch("quantum.evolution.print"):
            history, population = train(
                n=n,
                population_size=2,
                n_generations=1,
                steps_per_eval=1,
            )
        self.assertEqual(len(history), 1)
        self.assertEqual(len(population), 2)


class TestCheckpoint(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        from quantum.checkpoint import load_latest_checkpoint, save_checkpoint

        population = init_population(3, n, dtype, device)
        M_A, M_B = population[0]
        history = [{"generation": 0, "best_fitness": 0.5, "entanglement": 0.1}]
        save_checkpoint(0, 0.5, population, history, M_A, M_B, n, 3, 5, g, dt)
        ckpt = load_latest_checkpoint(device, dtype)
        self.assertEqual(ckpt["generation"], 0)
        self.assertEqual(len(ckpt["population"]), 3)

    def test_load_missing_raises(self):
        import shutil

        import quantum.checkpoint as cp
        from quantum.checkpoint import load_latest_checkpoint

        shutil.rmtree(cp.DATA_DIR, ignore_errors=True)
        os.makedirs(cp.DATA_DIR, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            load_latest_checkpoint(device, dtype)

    def test_train_resume_increases_generation(self):
        from quantum.checkpoint import clear_saved_checkpoints, load_latest_checkpoint, save_checkpoint

        clear_saved_checkpoints()
        population = init_population(2, n, dtype, device)
        history = [{"generation": 0, "best_fitness": 0.1, "entanglement": 0.05}]
        M_A, M_B = population[0]
        save_checkpoint(0, 0.1, population, history, M_A, M_B, n, 2, 2, g, dt)
        with patch("quantum.evolution.print"):
            train(
                n=n,
                population_size=2,
                n_generations=2,
                steps_per_eval=2,
                load=True,
                save_checkpoints=True,
            )
        ckpt = load_latest_checkpoint(device, dtype)
        self.assertEqual(ckpt["generation"], 1)
        self.assertEqual(len(ckpt["history"]), 2)

    def test_train_resume_uses_checkpoint_population_size(self):
        from quantum.checkpoint import clear_saved_checkpoints, load_latest_checkpoint, save_checkpoint

        clear_saved_checkpoints()
        population = init_population(4, n, dtype, device)
        history = [{"generation": 0, "best_fitness": 0.1, "entanglement": 0.05}]
        M_A, M_B = population[0]
        save_checkpoint(0, 0.1, population, history, M_A, M_B, n, 4, 2, g, dt)
        with patch("quantum.evolution.print"):
            train(
                n=n,
                population_size=2,
                n_generations=2,
                steps_per_eval=2,
                load=True,
                save_checkpoints=True,
            )
        ckpt = load_latest_checkpoint(device, dtype)
        self.assertEqual(len(ckpt["population"]), 2)

    def test_train_resume_grows_population(self):
        from quantum.checkpoint import clear_saved_checkpoints, load_latest_checkpoint, save_checkpoint

        clear_saved_checkpoints()
        population = init_population(2, n, dtype, device)
        history = [{"generation": 0, "best_fitness": 0.1, "entanglement": 0.05}]
        M_A, M_B = population[0]
        save_checkpoint(0, 0.1, population, history, M_A, M_B, n, 2, 2, g, dt)
        with patch("quantum.evolution.print"):
            train(
                n=n,
                population_size=4,
                n_generations=2,
                steps_per_eval=2,
                load=True,
                save_checkpoints=True,
            )
        ckpt = load_latest_checkpoint(device, dtype)
        self.assertEqual(len(ckpt["population"]), 4)


class TestViz(unittest.TestCase):
    def test_ideal_s_and_basin_threshold(self):
        from quantum.viz import basin_threshold_s, ideal_s

        self.assertAlmostEqual(ideal_s(8), 0.5 * math.log(8), places=6)
        self.assertAlmostEqual(basin_threshold_s(8), 0.1 * math.log(8), places=6)

    def test_subsystem_push_grids_shapes(self):
        from quantum.viz import joint_amplitude_grid, subsystem_push_grids

        M_A, M_B = init_population(1, n, dtype, device)[0]
        psi = vacuum_product_state(n, dtype, device)
        psi_ij, push_A, push_B = subsystem_push_grids(M_A, M_B, psi, n)
        self.assertEqual(psi_ij.shape, (n, n))
        self.assertEqual(push_A.shape, (n, n))
        self.assertEqual(push_B.shape, (n, n))
        self.assertTrue(
            torch.allclose(psi_ij, joint_amplitude_grid(psi, n).cpu())
        )

    def test_entropy_trajectory_length(self):
        from quantum.viz import entropy_trajectory

        M_A, M_B = init_population(1, n, dtype, device)[0]
        psi = vacuum_product_state(n, dtype, device)
        s_a, s_b = entropy_trajectory(M_A, M_B, psi, n, 5, g, dt, device)
        self.assertEqual(len(s_a), 5)
        self.assertEqual(len(s_b), 5)

    def test_subsystem_entropies_equal_for_pure_state(self):
        from quantum.physics import partial_trace, partial_trace_A

        psi = random_product_state(n, dtype, device)
        rho_ab = torch.outer(psi, psi.conj())
        s_a = von_neumann_entropy(partial_trace_A(rho_ab, n)).item()
        s_b = von_neumann_entropy(partial_trace(rho_ab, n)).item()
        self.assertAlmostEqual(s_a, s_b, places=4)

    def test_position_probability_grid(self):
        from quantum.viz import probability_grid_numpy

        psi = vacuum_product_state(n, dtype, device)
        grid = probability_grid_numpy(psi, n)
        self.assertEqual(grid.shape, (n, n))
        self.assertGreater(grid.max(), 0.0)

    def test_energy_organization_rgb_shape(self):
        from quantum.viz import energy_organization_rgb_numpy

        M_A, M_B = init_population(1, n, dtype, device)[0]
        psi = vacuum_product_state(n, dtype, device)
        rgb = energy_organization_rgb_numpy(M_A, M_B, psi, n)
        self.assertEqual(rgb.shape, (n, n, 3))
        self.assertGreater(rgb[:, :, 0].max(), 0.0)


class TestBasin(unittest.TestCase):
    def test_threshold_grows_with_n(self):
        from quantum.basin import threshold

        self.assertLess(threshold(4), threshold(16))


class TestNoCheckpointDuringPytest(unittest.TestCase):
    def test_train_does_not_write_checkpoints(self):
        import quantum.checkpoint as cp

        before = set(glob.glob(os.path.join(cp.DATA_DIR, "quantum_gen*.pt")))
        with patch("quantum.evolution.print"):
            train(
                n=n,
                population_size=2,
                n_generations=1,
                steps_per_eval=1,
                device=device,
                dtype=dtype,
            )
        after = set(glob.glob(os.path.join(cp.DATA_DIR, "quantum_gen*.pt")))
        self.assertEqual(before, after)


class TestHermitian(unittest.TestCase):
    def test_hermitian_symmetry(self):
        M = torch.randn(n, n, dtype=dtype, device=device)
        H = hermitian(M)
        self.assertTrue(torch.allclose(H, H.conj().mT, atol=1e-5))


class TestParabolicEntropy(unittest.TestCase):
    def test_positive_at_mid_entropy(self):
        S = torch.tensor(0.5 * math.log(n), device=device)
        val = parabolic_entropy(S, n)
        self.assertGreater(val.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
