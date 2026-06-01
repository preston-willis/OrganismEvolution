"""End-to-end physics verification across a full mini-simulation."""

import unittest
from unittest import mock

import torch

from quantum.device import get_device, get_dtype
from quantum.evolution import evaluate_organism, init_population, run_generation, train
from quantum.physics import (
    evolution_step,
    matrix_exp_unitary,
    partial_trace,
    random_product_state,
    total_hamiltonian,
    track_complexity,
    vacuum_product_state,
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
        M_A, M_B = init_population(1, N_E2E, dtype, device)[0]
        H_total, H_A = total_hamiltonian(M_A, M_B, N_E2E, G, dtype, device)
        assert_hermitian(self, H_A)
        assert_hermitian(self, H_total)
        assert_unitary(self, matrix_exp_unitary(H_total, DT))

    def test_evolution_preserves_norm_e2e(self):
        M_A, M_B = init_population(1, N_E2E, dtype, device)[0]
        psi = random_product_state(N_E2E, dtype, device)
        for _ in range(10):
            _, psi = evolution_step(M_A, M_B, psi, N_E2E, G, DT, device)
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
            self.assertIn("entanglement", entry)
            self.assertGreaterEqual(entry["entanglement"], -1e-6)

        M_A, M_B = population[0]
        psi = vacuum_product_state(N_E2E, dtype, device)
        _, evolved = evaluate_organism(M_A, M_B, psi, N_E2E, 5, G, DT, device)
        metrics = track_complexity(evolved, M_A, N_E2E)
        self.assertIn("participation_ratio", metrics)

    def test_run_generation_best_uses_evolved_state(self):
        pop = init_population(4, N_E2E, dtype, device)
        _, _, complexity, _, _, _, psi_final = run_generation(
            pop, N_E2E, 5, G, DT, device, dtype
        )
        M_A, _ = pop[0]
        direct = track_complexity(psi_final, M_A, N_E2E)
        self.assertAlmostEqual(
            complexity["entanglement"], direct["entanglement"], places=4
        )

    def test_vacuum_partial_trace_zero_entropy(self):
        psi = vacuum_product_state(N_E2E, dtype, device)
        rho_A = partial_trace(torch.outer(psi, psi.conj()), N_E2E)
        self.assertAlmostEqual(von_neumann_entropy(rho_A).item(), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
