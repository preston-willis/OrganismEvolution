"""End-to-end physics verification across a full mini-simulation."""

import unittest
from unittest import mock

import torch

from quantum.device import get_device, get_dtype
from quantum.evolution import evaluate_organism, init_population, run_generation, train
from quantum.physics import (
    critical_edge_initial_state,
    evolution_step,
    matrix_exp_unitary,
    partial_trace,
    total_hamiltonian,
    track_criticality,
    von_neumann_entropy,
)

device = get_device()
dtype = get_dtype()
N_E2E = 4
G = 0.1
DT = 0.01
ATOL = 1e-4


def state_norm(psi):
    return torch.sqrt(torch.sum(torch.abs(psi) ** 2)).item()


def assert_hermitian(test_case, M, atol=ATOL):
    test_case.assertTrue(torch.allclose(M, M.conj().mT, atol=atol))


def assert_unitary(test_case, U, atol=1e-3):
    I = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)
    test_case.assertTrue(torch.allclose(U @ U.conj().mT, I, atol=atol))


class TestE2EPhysics(unittest.TestCase):
    def test_hamiltonian_and_unitary(self):
        M = init_population(1, N_E2E, dtype, device)[0]
        H_total, H = total_hamiltonian(M, N_E2E, G, dtype, device)
        assert_hermitian(self, H)
        assert_hermitian(self, H_total)
        assert_unitary(self, matrix_exp_unitary(H_total, DT))

    def test_evolution_preserves_norm_e2e(self):
        M = init_population(1, N_E2E, dtype, device)[0]
        psi = critical_edge_initial_state(N_E2E, dtype, device, 99)
        for _ in range(10):
            _, psi = evolution_step(M, psi, N_E2E, G, DT, device)
            self.assertAlmostEqual(state_norm(psi), 1.0, places=4)

    def test_train_full_simulation_physics(self):
        torch.manual_seed(42)
        with mock.patch("quantum.evolution.print"):
            history, population = train(
                n=N_E2E,
                population_size=4,
                n_generations=3,
                steps_per_eval=5,
                device=device,
                dtype=dtype,
            )

        self.assertEqual(len(history), 3)
        self.assertEqual(len(population), 4)

        for entry in history:
            self.assertIn("best_fitness", entry)
            self.assertIn("f_static", entry)
            self.assertIn("f_dynamic", entry)

        M = population[0]
        psi = critical_edge_initial_state(N_E2E, dtype, device, 1)
        _, evolved = evaluate_organism(M, psi, N_E2E, 5, G, DT, device)
        metrics = track_criticality(evolved, M, N_E2E, dtype, device)
        self.assertIn("r_mean", metrics)

    def test_run_generation_best_uses_evolved_state(self):
        pop = init_population(4, N_E2E, dtype, device)
        _, _, metrics, _, _, psi_final = run_generation(
            pop, N_E2E, 5, G, DT, device, dtype
        )
        M = pop[0]
        direct = track_criticality(psi_final, M, N_E2E, dtype, device)
        self.assertAlmostEqual(
            metrics["entanglement"], direct["entanglement"], places=4
        )

    def test_critical_edge_initial_zero_entanglement(self):
        psi = critical_edge_initial_state(N_E2E, dtype, device, 42)
        rho_B = partial_trace(torch.outer(psi, psi.conj()), N_E2E)
        self.assertAlmostEqual(von_neumann_entropy(rho_B).item(), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
