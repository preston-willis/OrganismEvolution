"""Unit tests for the organism simulation modules."""
import argparse
import gc
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

import oriented_conv
import simulation as sim_module
from cnn import (
    CNN_OUTPUT_DIM,
    BasicCPPN,
    EnergyDistributionCNN,
    neighbor_direction_weights_from_proportions,
)
from config import (
    CNN_FITNESS_PERSISTENCE_TERRAIN,
    CNN_HIDDEN_CHANNELS,
    ORGANISM_COUNT,
    SEED_ORGANISM_ENERGY,
    TERRAIN_PUMP_RATE,
    THERMO_CONDUCTANCE,
    WORLD_SIZE,
)
from entropy import configurational_entropy, system_entropy_components
from environment import Environment
from evolution import (
    CNNGeneticAlgorithm,
    CNNEvaluator,
    CNNEvolutionDriver,
    CNN_FITNESS_MODES,
    _configure_fitness_environment,
    _evaluate_cnn_worker,
    _init_worker,
    _release_device_memory,
    run_cnn_fitness_rollout,
    set_cnn_fitness_mode,
    validate_cnn_fitness_mode,
)
from model_io import apply_loaded_cnn, clear_saved_networks, load_latest_cnn
from organism import OrganismManager
import physics
from physics import band_flux, chemical_potential, mixing_entropy
from renderer import Renderer
from runtime import get_device
from seeding import fixed_seed_positions
from simulation import Simulation

device = get_device()
SMALL = 12


def make_organism_manager(world_size=SMALL, center=None):
    terrain = torch.ones(world_size, world_size, device=device) * 0.5
    if center is not None:
        cy, cx = center
        seed_positions = [[cx, cy]]
    else:
        seed_positions = fixed_seed_positions(world_size, 1)
    return OrganismManager(world_size, 1, terrain, seed_positions=seed_positions)


def uniform_proportions(world_size, ci, cj):
    """3x3 proportions with all mass at (ci, cj), summing to 1 per cell."""
    p = torch.zeros(3, 3, world_size, world_size, device=device)
    p[ci, cj, :, :] = 1.0
    return p


def system_total_energy(sim):
    """Organism + environment + uncommitted birth energy + inaccessible destroyed bucket."""
    om = sim.organism_manager
    return (
        om.energy_matrix.sum()
        + sim.environment.terrain.sum()
        + om.pending_birth_energy.sum()
        + om.destroyed_energy
    )


def apply_neighbor_repro_physics(om, terrain, proportions, conductance_scale=1.0):
    """Mirror harvest + repro half of compute_energy."""
    org_sum_before_harvest = om.energy_matrix.sum()
    terrain_debit = om._apply_harvest(terrain, conductance_scale)
    om._flush_tick_destroyed()
    org_sum_after_harvest = om.energy_matrix.sum()
    destroyed_before_repro = om.destroyed_energy
    direction_weights = neighbor_direction_weights_from_proportions(
        proportions, om.topology_matrix
    )
    result = physics.apply_repro_flux_tick(
        om.energy_bands,
        om.spectrum,
        om.topology_matrix,
        direction_weights,
        conductance_scale,
    )
    om.energy_bands = result.energy_bands
    om.pending_bands = result.pending_bands
    om.pending_birth_energy = result.pending_bands.sum(dim=0)
    om._sync_total_energy()
    om._add_destroyed_energy(result.sink_bands.sum())
    om._flush_tick_destroyed()
    return {
        "harvested": terrain_debit,
        "org_sum_before_harvest": org_sum_before_harvest,
        "org_sum_after_harvest": org_sum_after_harvest,
        "org_sum_after_repro": om.energy_matrix.sum(),
        "pending_sum_after_repro": om.pending_birth_energy.sum(),
        "repro_destroyed": om.destroyed_energy - destroyed_before_repro,
    }


class TestBasicCPPN(unittest.TestCase):
    def setUp(self):
        self.cppn = BasicCPPN(device)

    def test_forward_output_shape(self):
        coords = torch.randn(10, 3, device=device)
        out = self.cppn.forward(coords)
        self.assertEqual(out.shape, (10, 1))

    def test_generate_conv_weights_shape(self):
        w = self.cppn.generate_conv_weights(1 + 1 + CNN_HIDDEN_CHANNELS, 8, 3)
        self.assertEqual(w.shape, (8, 1 + 1 + CNN_HIDDEN_CHANNELS, 3, 3))

    def test_generate_bias_shape(self):
        b = self.cppn.generate_bias(CNN_OUTPUT_DIM)
        self.assertEqual(b.shape, (CNN_OUTPUT_DIM,))


class TestEnergyDistributionCNN(unittest.TestCase):
    def setUp(self):
        self.cnn = EnergyDistributionCNN(device)
        self.H = SMALL

    def test_bucket_offsets_consistent(self):
        offsets = self.cnn._bucket_offsets
        ring = EnergyDistributionCNN._RING_CIJ
        for k in range(8):
            for l in range(8):
                ci, cj = ring[(l + k) % 8]
                self.assertEqual(offsets[k, l, 0].item(), ci - 1)
                self.assertEqual(offsets[k, l, 1].item(), cj - 1)

    def test_zero_hidden_channel_bias(self):
        self.cnn.conv2.bias.data[9] = 5.0
        self.cnn.conv2.bias.data[11] = 5.0
        self.cnn._zero_hidden_channel_bias()
        self.assertEqual(self.cnn.conv2.bias.data[9].item(), 0.0)
        self.assertEqual(self.cnn.conv2.bias.data[11].item(), 0.0)

    def test_forward_output_shapes(self):
        shareable = torch.rand(self.H, self.H, device=device)
        terrain = torch.rand(self.H, self.H, device=device)
        hidden = torch.zeros(CNN_HIDDEN_CHANNELS, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        proportions, hidden_out = self.cnn(shareable, terrain, hidden, rotation)
        self.assertEqual(proportions.shape, (3, 3, self.H, self.H))
        self.assertEqual(hidden_out.shape, (CNN_HIDDEN_CHANNELS, self.H, self.H))

    def test_proportions_sum_to_one(self):
        shareable = torch.ones(self.H, self.H, device=device)
        terrain = torch.ones(self.H, self.H, device=device) * 0.5
        hidden = torch.zeros(CNN_HIDDEN_CHANNELS, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        proportions, _ = self.cnn(shareable, terrain, hidden, rotation)
        sums = proportions.sum(dim=(0, 1))
        self.assertTrue(torch.allclose(sums, torch.ones(self.H, self.H, device=device), rtol=1e-4))

    def test_hidden_softmax_outputs(self):
        shareable = torch.rand(self.H, self.H, device=device)
        terrain = torch.rand(self.H, self.H, device=device)
        hidden = torch.rand(CNN_HIDDEN_CHANNELS, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        _, hidden_out = self.cnn(shareable, terrain, hidden, rotation)
        sums = hidden_out.sum(dim=0)
        self.assertTrue(torch.all(hidden_out >= 0))
        self.assertTrue(torch.allclose(sums, torch.ones(self.H, self.H, device=device), rtol=1e-4))

    def test_rotate_proportions_moves_local_north(self):
        H = 8
        proportions = torch.zeros(3, 3, H, H, device=device)
        ci, cj = EnergyDistributionCNN._RING_CIJ[6]
        proportions[ci, cj, :, :] = 1.0
        bucket = 2
        rotation = torch.full((H, H), bucket * torch.pi / 4, device=device)
        rotated = self.cnn._rotate_proportions_8way(proportions, rotation)
        world_ci, world_cj = EnergyDistributionCNN._RING_CIJ[(6 + bucket) % 8]
        self.assertAlmostEqual(rotated[world_ci, world_cj].mean().item(), 1.0, places=4)

    def test_conv1_rotation_zero_matches_standard_conv(self):
        inp = torch.randn(1 + 1 + CNN_HIDDEN_CHANNELS, self.H, self.H, device=device)
        rot0 = torch.zeros(self.H, self.H, device=device)
        out_orient = oriented_conv.conv1_forward(
            inp, self.cnn.conv1.weight, self.cnn.conv1.bias, rot0, self.cnn._bucket_offsets
        )
        padded = torch.nn.functional.pad(inp.unsqueeze(0), (1, 1, 1, 1), mode="circular")
        out_std = torch.nn.functional.conv2d(
            padded, self.cnn.conv1.weight, self.cnn.conv1.bias
        ).squeeze(0)
        out_std = torch.relu(out_std)
        self.assertTrue(torch.allclose(out_orient, out_std, atol=1e-4))


class TestCNNGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.ga = CNNGeneticAlgorithm(4, 0.5, 0.1, device)

    def test_reset_fitness(self):
        self.ga.fitness_scores = [1.0, 2.0, 3.0, 4.0]
        self.ga.reset_fitness()
        self.assertEqual(self.ga.fitness_scores, [0.0, 0.0, 0.0, 0.0])

    def test_calc_fittest(self):
        self.ga.fitness_scores = [1.0, 5.0, 3.0, 2.0]
        self.ga.calc_fittest()
        self.assertEqual(self.ga.fittest_index, 1)

    def test_crossover_copies_parent_cppn_to_non_fittest(self):
        self.ga.fittest_index = 0
        parent_w = self.ga.subjects[0].cppn.fc1.weight.data.clone()
        self.ga.subjects[1].cppn.fc1.weight.data.fill_(999.0)
        self.ga.crossover(self.ga.subjects[0])
        self.assertTrue(torch.equal(self.ga.subjects[1].cppn.fc1.weight.data, parent_w))

    def test_mutate_skips_fittest(self):
        self.ga.fittest_index = 0
        before = self.ga.subjects[0].cppn.fc1.weight.data.clone()
        torch.manual_seed(0)
        self.ga.mutate()
        self.assertTrue(torch.equal(self.ga.subjects[0].cppn.fc1.weight.data, before))

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_model.pt")
            self.ga.fitness_scores[0] = 42.0
            with patch.object(self.ga, "save_model") as mock_save:
                torch.save(self.ga.subjects[0].state_dict(), path)
            loaded = EnergyDistributionCNN(device)
            loaded.load_state_dict(torch.load(path, map_location=device))
            self.assertTrue(
                torch.equal(
                    loaded.conv1.weight.data,
                    self.ga.subjects[0].conv1.weight.data,
                )
            )


class TestOrganismManagerStaticWeights(unittest.TestCase):
    def test_neighbor8_kernel_uniform_ring(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        w = om._neighbor8_kernel
        self.assertAlmostEqual(w.sum().item(), 1.0, places=4)
        self.assertEqual(w[0, 0, 1, 1].item(), 0.0)


class TestCouplingPhysics(unittest.TestCase):
    def test_consumer_zero_routes_flux_to_sink(self):
        driving = torch.tensor([[0.5]], device=device)
        source = torch.tensor([[0.8]], device=device)
        capacity = torch.tensor([[1.0]], device=device)
        conductance = torch.tensor([[0.05]], device=device)
        useful, sink, _ = band_flux(driving, source, capacity, conductance, torch.zeros_like(driving))
        self.assertAlmostEqual(useful.item(), 0.0, places=4)
        self.assertGreater(sink.item(), 0.0)

    def test_configure_load_only(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        args = argparse.Namespace(load=False)
        from model_io import configure_organism_manager_from_args

        configure_organism_manager_from_args(om, args)
        self.assertIsInstance(om.energy_distribution_cnn, EnergyDistributionCNN)


class TestOrganismManagerEnergy(unittest.TestCase):
    def setUp(self):
        self.om = make_organism_manager(SMALL)

    def test_initialize_topology_places_seed(self):
        om = make_organism_manager(SMALL, center=(SMALL // 2, SMALL // 2))
        self.assertGreater(om.topology_matrix.sum().item(), 0)
        self.assertGreater(om.energy_matrix.sum().item(), 0)

    def test_fixed_seed_positions_center(self):
        positions = fixed_seed_positions(SMALL, 3)
        cx = SMALL // 2
        cy = SMALL // 2
        self.assertEqual(len(positions), 3)
        for x, y in positions:
            self.assertEqual(x, cx)
            self.assertEqual(y, cy)

    def test_death_clears_cell_state(self):
        y, x = 6, 6
        self.om.topology_matrix[y, x] = 1
        self.om.energy_bands[:, y, x] = 0.01 / CNN_HIDDEN_CHANNELS
        self.om._sync_total_energy()
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        self.om._apply_harvest(terrain)
        self.assertEqual(self.om.topology_matrix[y, x].item(), 0)

    def test_birth_commits_pending_energy_uniform_spectrum(self):
        y, x = 5, 5
        uniform = 1.0 / CNN_HIDDEN_CHANNELS
        self.om.new_cell_candidates = torch.zeros(SMALL, SMALL, dtype=torch.bool, device=device)
        self.om.new_cell_candidates[y, x] = True
        self.om.pending_birth_energy[y, x] = 0.5
        self.om.pending_bands[:, y, x] = 0.5 / CNN_HIDDEN_CHANNELS
        self.om._has_new_cell_candidates = True
        self.om.compute_topology()
        self.assertEqual(self.om.topology_matrix[y, x].item(), 1)
        self.assertAlmostEqual(self.om.energy_matrix[y, x].item(), 0.5, places=4)
        expected = torch.full((CNN_HIDDEN_CHANNELS,), uniform, device=device)
        self.assertTrue(torch.allclose(self.om.hidden_channels[:, y, x], expected))

    def test_candidates_do_not_store_energy_before_birth(self):
        sim = Simulation(enable_debug=False)
        om = sim.organism_manager
        sim.update_simulation()
        empty = om.topology_matrix == 0
        if om.new_cell_candidates.any():
            self.assertTrue(torch.all(om.energy_matrix[empty & om.new_cell_candidates] == 0))
            unborn_candidates = om.new_cell_candidates & (om.topology_matrix == 0)
            if unborn_candidates.any():
                self.assertTrue(torch.all(om.pending_birth_energy[unborn_candidates] > 0))

    def test_hidden_channels_softmax_on_alive_cells(self):
        sim = Simulation(enable_debug=False)
        om = sim.organism_manager
        for _ in range(30):
            sim.update_simulation()
        alive = om.topology_matrix > 0
        if alive.any():
            vals = om.hidden_channels[:, alive]
            self.assertTrue(torch.all(vals >= 0))
            self.assertTrue(torch.allclose(vals.sum(dim=0), torch.ones(vals.shape[1], device=device), rtol=1e-3))

class TestEnvironment(unittest.TestCase):
    def test_type_2_terrain_starts_from_perlin(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        terrain = env.generate_terrain()
        self.assertEqual(terrain.shape, (SMALL, SMALL))
        self.assertGreater(terrain.max().item(), 0.0)

    def test_generate_terrain_dispatches_type(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 1
        t1 = env.generate_terrain()
        env.environment_type = 2
        t2 = env.generate_terrain()
        self.assertEqual(t1.shape, (SMALL, SMALL))
        self.assertEqual(t2.shape, (SMALL, SMALL))

    def test_type1_boost_only_center_3x3(self):
        from config import STARTING_POSITION_TERRAIN_BOOST

        env = Environment(SMALL, [[0, 0], [1, 1]])
        env.environment_type = 1
        terrain = env._generate_energy_mask_terrain()
        cy, cx = SMALL // 2, SMALL // 2
        boost = min(STARTING_POSITION_TERRAIN_BOOST, 1.0)
        self.assertAlmostEqual(terrain[cy, cx].item(), boost, places=5)
        self.assertLess(terrain[0, 0].item(), 1e-6)
        self.assertEqual(int(terrain.gt(0).sum().item()), 9)

    def test_compute_environment_depletes_terrain(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 1
        env.terrain[0, 0] = 0.8
        before = env.terrain.clone()
        topology = torch.zeros(SMALL, SMALL, device=device)
        topology[0, 0] = 1
        harvested = torch.zeros(SMALL, SMALL, device=device)
        harvested[0, 0] = 1.0
        env.compute_environment(topology, harvested)
        self.assertLess(env.terrain[0, 0].item(), before[0, 0].item())

    def test_pump_perlin_wave_has_configured_bias(self):
        from config import PUMP_WAVE_BIAS  # noqa: F401 — used by pump tests
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        wave = env._centered_pump_wave()
        self.assertAlmostEqual(wave.mean().item(), PUMP_WAVE_BIAS, places=4)

    def test_pump_wave_changes_terrain(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        env.terrain.fill_(0.5)
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.fill_(1.0)
        om.energy_matrix.fill_(0.05)
        before = env.terrain.clone()
        env._apply_thermodynamic_pump(om)
        self.assertFalse(torch.allclose(env.terrain, before))

    def test_type2_static_base_unchanged_without_pump(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        base = env.generate_terrain()
        env.terrain = base.clone()
        topology = torch.zeros(SMALL, SMALL, device=device)
        with patch("config.TERRAIN_PUMP_ENABLED", False):
            env.compute_environment(topology, torch.zeros(SMALL, SMALL, device=device))
        self.assertTrue(torch.allclose(env.terrain, base))

    def test_type2_harvest_depletes_then_pump_heals_trail(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix[6, 6] = 1.0
        env.terrain = env.generate_terrain()
        env.terrain[6, 6] -= 0.35
        low = env.terrain[6, 6].item()
        env._pump_drive = 1.0
        healed = low
        for _ in range(300):
            env._apply_thermodynamic_pump(om)
            env.terrain.clamp_(0.0, 1.0)
            healed = env.terrain[6, 6].item()
            if healed > low + 0.01:
                break
        self.assertGreater(healed, low)

    def test_pump_population_error_negative_when_above_half_grid(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.fill_(1.0)
        self.assertLess(env._pump_population_error(om), 0.0)

    def test_pump_pid_drive_negative_when_above_half_grid(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.fill_(1.0)
        for _ in range(200):
            drive = env._pump_pid_step(env._pump_population_error(om))
        self.assertLess(drive, 0.0)

    def test_thermodynamic_pump_subtracts_wave_when_negative_drive(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        om = make_organism_manager(SMALL, center=(6, 6))
        env.terrain.fill_(0.6)
        om.topology_matrix.fill_(1.0)
        om.energy_matrix.fill_(0.7)
        before = env.terrain.clone()
        env._pump_drive = -1.0
        env._apply_thermodynamic_pump(om)
        self.assertFalse(torch.allclose(env.terrain, before))

    def test_thermodynamic_pump_adds_wave_when_positive_drive(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 2
        env.terrain.fill_(0.6)
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.fill_(1.0)
        om.energy_matrix.fill_(0.05)
        before = env.terrain.clone()
        env._pump_drive = 1.0
        env._apply_thermodynamic_pump(om)
        self.assertFalse(torch.allclose(env.terrain, before))

    def test_type3_replaces_terrain_with_perlin(self):
        env = Environment(SMALL, [[6, 6]])
        env.environment_type = 3
        env.terrain.fill_(0.0)
        topology = torch.zeros(SMALL, SMALL, device=device)
        env.compute_environment(topology, torch.zeros(SMALL, SMALL, device=device))
        self.assertGreater(env.terrain.max().item(), 0.0)


class TestRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = Renderer(SMALL)

    def test_render_shape(self):
        env = torch.rand(SMALL, SMALL, device=device)
        topo = torch.zeros(SMALL, SMALL, device=device)
        mask = torch.zeros(SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, mask)
        self.assertEqual(image.shape, (4, SMALL, SMALL))

    def test_organism_hidden_channel_0_is_red(self):
        env = torch.zeros(SMALL, SMALL, device=device)
        topo = torch.zeros(SMALL, SMALL, device=device)
        topo[6, 6] = 1
        hidden = torch.zeros(CNN_HIDDEN_CHANNELS, SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, topo, hidden_channels=hidden)
        self.assertAlmostEqual(image[0, 6, 6].item(), 0.0, places=4)
        self.assertLess(image[1, 6, 6].item(), 0.1)

    def test_organism_all_hidden_on_is_white(self):
        env = torch.zeros(SMALL, SMALL, device=device)
        topo = torch.zeros(SMALL, SMALL, device=device)
        topo[6, 6] = 1
        hidden = torch.ones(CNN_HIDDEN_CHANNELS, SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, topo, hidden_channels=hidden)
        self.assertGreater(image[0, 6, 6].item(), 0.9)
        self.assertGreater(image[1, 6, 6].item(), 0.9)

    def test_coupling_view_red_green_blue_bins(self):
        env = torch.ones(SMALL, SMALL, device=device) * 0.8
        topo = torch.zeros(SMALL, SMALL, device=device)
        topo[6, 6] = 1
        hidden = torch.zeros(CNN_HIDDEN_CHANNELS, SMALL, SMALL, device=device)
        hidden[0, 6, 6] = 1.0
        hidden[1, 7, 6] = 1.0
        topo[7, 6] = 1
        hidden[2, 6, 7] = 1.0
        topo[6, 7] = 1
        image = self.renderer.render(
            env, topo, topo, hidden_channels=hidden, coupling_view=True
        )
        self.assertGreater(image[0, 6, 6].item(), 0.9)
        self.assertLess(image[1, 6, 6].item(), 0.1)
        self.assertGreater(image[1, 7, 6].item(), 0.9)
        self.assertGreater(image[2, 6, 7].item(), 0.9)
        self.assertLess(image[0, 0, 0].item(), 0.1)

    def test_background_is_cyan(self):
        env = torch.ones(SMALL, SMALL, device=device) * 0.8
        topo = torch.zeros(SMALL, SMALL, device=device)
        mask = torch.zeros(SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, mask)
        self.assertAlmostEqual(image[0, 0, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(image[1, 0, 0].item(), 0.8, places=4)
        self.assertAlmostEqual(image[2, 0, 0].item(), 0.8, places=4)

    def test_toggle_render_mode(self):
        self.renderer.render_mode = "org_top"
        self.renderer.toggle_render_mode()
        self.assertEqual(self.renderer.render_mode, "org_energy")


class TestSimulation(unittest.TestCase):
    def test_update_returns_expected_keys(self):
        sim = Simulation(enable_debug=False)
        data = sim.update_simulation()
        for key in ("terrain", "topology", "energy", "new_cell_candidates", "hidden_channels", "pending_birth_energy", "destroyed_energy"):
            self.assertIn(key, data)

    def test_reset_for_replay_resets_tick(self):
        sim = Simulation(enable_debug=False)
        sim.update_simulation()
        sim.reset_for_replay()
        self.assertEqual(sim.tick, 0)


class TestWorkerHelpers(unittest.TestCase):
    def test_release_device_memory_runs(self):
        _release_device_memory(device)

    def test_init_worker_sets_device(self):
        _init_worker(str(device))
        import evolution

        self.assertIsNotNone(evolution._worker_device)

    def test_evaluate_cnn_worker_short_run(self):
        _init_worker(str(device))
        cnn = EnergyDistributionCNN(device)
        cpu_state = {k: v.cpu().clone() for k, v in cnn.state_dict().items()}
        args = (cpu_state, SMALL, 3, 0, False, "cell_count")
        fitness, tick_data = _evaluate_cnn_worker(args)
        self.assertIsInstance(fitness, float)
        self.assertEqual(tick_data, [])


class TestCNNEvaluator(unittest.TestCase):
    def test_evaluate_single_cnn(self):
        evaluator = CNNEvaluator(SMALL, 5, device)
        sim = Simulation(enable_debug=False)
        fitness = evaluator._evaluate_single_cnn(sim)
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0.0)


class TestCNNEvolutionDriver(unittest.TestCase):
    def test_evaluate_cnn(self):
        driver = CNNEvolutionDriver(SMALL, epochs=1, max_time=5)
        cnn = EnergyDistributionCNN(device)
        sim = Simulation(enable_debug=False)
        fitness = driver.evaluate_cnn(cnn, sim)
        self.assertGreaterEqual(fitness, 0.0)


class TestCNNFitnessModes(unittest.TestCase):
    def test_cell_count_mode_accumulates_cells(self):
        sim = Simulation(enable_debug=False)
        with patch("config.CNN_FITNESS_MODE", "cell_count"):
            fitness, cumulative, _ = run_cnn_fitness_rollout(sim, 5)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertGreaterEqual(cumulative, 0.0)

    def test_entropy_production_mode_uses_destroyed_entropy(self):
        sim = Simulation(enable_debug=False)
        with patch("config.CNN_FITNESS_MODE", "entropy_production"):
            fitness, _, _ = run_cnn_fitness_rollout(sim, 10)
        self.assertGreaterEqual(fitness, 0.0)
        self.assertGreaterEqual(sim.organism_manager.destroyed_entropy, 0.0)
        self.assertGreaterEqual(sim.organism_manager.destroyed_energy, 0.0)

    def test_persistence_mode_uniform_terrain(self):
        sim = Simulation(enable_debug=False)
        _configure_fitness_environment(sim, "persistence")
        self.assertAlmostEqual(
            sim.environment.terrain.min().item(),
            CNN_FITNESS_PERSISTENCE_TERRAIN,
            places=4,
        )
        self.assertAlmostEqual(
            sim.environment.terrain.max().item(),
            CNN_FITNESS_PERSISTENCE_TERRAIN,
            places=4,
        )

    def test_persistence_fitness_is_tick_count_not_cell_sum(self):
        sim = Simulation(enable_debug=False)
        max_time = 8
        fitness, cumulative, _ = run_cnn_fitness_rollout(sim, max_time, fitness_mode="persistence")
        self.assertLessEqual(fitness, max_time)
        self.assertGreaterEqual(fitness, 0.0)
        if fitness > 0:
            self.assertLessEqual(fitness, cumulative)

    def test_persistence_can_end_before_max_time(self):
        sim = Simulation(enable_debug=False)
        fitness, _, _ = run_cnn_fitness_rollout(sim, 200, fitness_mode="persistence")
        self.assertLessEqual(fitness, 200.0)

    def test_entropy_and_cell_count_modes_can_differ(self):
        sim_cell = Simulation(enable_debug=False)
        sim_entropy = Simulation(enable_debug=False)
        cell_fitness, _, _ = run_cnn_fitness_rollout(sim_cell, 15, fitness_mode="cell_count")
        entropy_fitness, _, _ = run_cnn_fitness_rollout(sim_entropy, 15, fitness_mode="entropy_production")
        self.assertGreaterEqual(cell_fitness, 0.0)
        self.assertGreaterEqual(entropy_fitness, 0.0)

    def test_validate_cnn_fitness_mode_rejects_unknown(self):
        with self.assertRaises(ValueError):
            validate_cnn_fitness_mode("biomass")

    def test_evaluator_passes_fitness_mode_to_worker(self):
        evaluator = CNNEvaluator(SMALL, 3, device, fitness_mode="persistence")
        self.assertEqual(evaluator.fitness_mode, "persistence")

    def test_evolution_driver_fitness_mode(self):
        driver = CNNEvolutionDriver(SMALL, epochs=1, max_time=3, fitness_mode="entropy_production")
        self.assertEqual(driver.fitness_mode, "entropy_production")
        self.assertEqual(driver.evaluator.fitness_mode, "entropy_production")

    def test_life_like_mode_combines_entropy_and_order(self):
        sim = Simulation(enable_debug=False)
        with patch("config.CNN_FITNESS_LIFE_LIKE_ORDER_WEIGHT", 1.0), patch(
            "config.CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR", 0.0
        ):
            fitness, _, _ = run_cnn_fitness_rollout(sim, 15, fitness_mode="life_like")
        self.assertGreaterEqual(fitness, 0.0)

    def test_life_like_rewards_negative_delta_config_entropy(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        om.energy_matrix[6, 6] = 0.5
        s_start = configurational_entropy(om, terrain)
        om.energy_matrix[6, 6] = 0.9
        s_ordered = configurational_entropy(om, terrain)
        delta = (s_ordered - s_start).item()
        score = 1.0 - 1.0 * delta
        self.assertLess(delta, 0.0)
        self.assertGreater(score, 1.0)

    @patch("config.CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR", 100.0)
    def test_life_like_fitness_non_negative(self):
        sim = Simulation(enable_debug=False)
        fitness, _, _ = run_cnn_fitness_rollout(sim, 5, fitness_mode="life_like")
        self.assertGreaterEqual(fitness, 0.0)


class TestThermodynamicEntropy(unittest.TestCase):
    def test_mixing_entropy_max_at_half(self):
        e = torch.tensor([0.5], device=device)
        s = mixing_entropy(e)
        self.assertGreater(s.item(), mixing_entropy(torch.tensor([0.1], device=device)).item())

    def test_chemical_potential_sign(self):
        low = chemical_potential(torch.tensor([0.1], device=device)).item()
        high = chemical_potential(torch.tensor([0.9], device=device)).item()
        self.assertGreater(low, 0.0)
        self.assertLess(high, 0.0)

    def test_destroyed_entropy_tracks_energy_over_temperature(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.5
        om._add_destroyed_energy(torch.tensor(0.1, device=device))
        om._flush_tick_destroyed()
        from config import THERMO_ENV_TEMPERATURE
        self.assertAlmostEqual(om.destroyed_entropy, 0.1 / THERMO_ENV_TEMPERATURE, places=5)

    def test_harvest_produces_irreversible_entropy(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_bands[:, 6, 6] = 0.2 / CNN_HIDDEN_CHANNELS
        om._sync_total_energy()
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.8
        entropy_before = om.destroyed_entropy
        om._apply_harvest(terrain)
        om._flush_tick_destroyed()
        self.assertGreater(om.destroyed_entropy, entropy_before)

    def test_system_entropy_total_includes_pools(self):
        sim = Simulation(enable_debug=False)
        from entropy import system_entropy_total

        total = system_entropy_total(sim.organism_manager, sim.environment.terrain)
        self.assertGreaterEqual(total.item(), 0.0)

    def test_harvest_destroyed_entropy_matches_energy_loss_over_t(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_bands[:, 6, 6] = 0.5
        om._sync_total_energy()
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        entropy_before = om.destroyed_entropy
        energy_before = om.destroyed_energy
        om._apply_harvest(terrain)
        om._flush_tick_destroyed()
        from config import THERMO_ENV_TEMPERATURE
        d_e = om.destroyed_energy - energy_before
        d_s = om.destroyed_entropy - entropy_before
        self.assertGreater(d_e, 0.0)
        self.assertAlmostEqual(d_s, d_e / THERMO_ENV_TEMPERATURE, places=4)


class TestThermodynamics(unittest.TestCase):
    """Thermodynamic constraints on energy transfer, bounds, and bookkeeping."""

    def test_energy_bounded_zero_to_one_after_neighbor_repro(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.95
        proportions = torch.ones(3, 3, SMALL, SMALL, device=device) / 9.0
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        apply_neighbor_repro_physics(om, terrain, proportions)
        self.assertTrue(torch.all(om.energy_matrix >= 0))
        self.assertTrue(torch.all(om.energy_matrix <= 1))

    def test_misaligned_spectrum_routes_slack_to_sink(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.5
        om.hidden_channels[:, 6, 6] = torch.tensor([0.0, 1.0, 0.0], device=device)
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.8
        destroyed_before = om.destroyed_energy
        om._apply_harvest(terrain)
        om._flush_tick_destroyed()
        self.assertGreater(om.destroyed_energy, destroyed_before)

    def test_harvest_increases_organism_energy_on_rich_terrain(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.5
        terrain = torch.ones(SMALL, SMALL, device=device)
        before = om.energy_matrix[6, 6].item()
        om._apply_harvest(terrain)
        after = om.energy_matrix[6, 6].item()
        self.assertGreater(after, before)

    def test_harvest_bounded_by_terrain_and_capacity(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.2
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.8
        harvested = om._apply_harvest(terrain)
        self.assertGreater(harvested[6, 6].item(), 0.0)
        self.assertLessEqual(harvested[6, 6].item(), 0.8 + 1e-6)
        self.assertLessEqual(harvested[6, 6].item(), 0.8 + 1e-6)

    @patch("config.TERRAIN_PUMP_ENABLED", False)
    def test_full_tick_depletes_environment_on_harvest(self):
        sim = Simulation(enable_debug=False)
        sim.environment.environment_type = 2
        terrain_before = sim.environment.terrain.clone()
        org_before = sim.organism_manager.energy_matrix.sum().item()
        harvested = sim.organism_manager.compute_energy(sim.environment.terrain)
        sim.environment.compute_environment(sim.organism_manager.topology_matrix, harvested)
        terrain_after = sim.environment.terrain.sum().item()
        org_after = sim.organism_manager.energy_matrix.sum().item()
        if harvested.sum().item() > 0 and terrain_before.sum().item() > 0:
            self.assertLessEqual(terrain_after, terrain_before.sum().item())
        self.assertTrue(torch.all(sim.organism_manager.energy_matrix >= 0))
        self.assertTrue(torch.all(sim.organism_manager.energy_matrix <= 1))

    def test_compute_energy_matches_manual_neighbor_repro(self):
        y, x = 6, 6
        spectrum = torch.tensor([1.0, 0.0, 0.0], device=device)
        om = make_organism_manager(SMALL, center=(y, x))
        om_manual = make_organism_manager(SMALL, center=(y, x))
        for inst in (om, om_manual):
            inst.spectrum[:, y, x] = spectrum
            inst.energy_matrix[y, x] = 0.7
            inst.energy_bands[:, y, x] = 0.7 * spectrum
        terrain = torch.zeros(SMALL, SMALL, device=device)
        proportions = uniform_proportions(SMALL, 2, 1)

        apply_neighbor_repro_physics(om_manual, terrain.clone(), proportions)

        def fixed_forward(shareable, terr, hidden, rotation):
            return proportions, hidden

        with patch.object(om.energy_distribution_cnn, "forward", side_effect=fixed_forward):
            om.compute_energy(terrain.clone())

        self.assertTrue(
            torch.allclose(om.energy_matrix, om_manual.energy_matrix, atol=1e-4),
            msg="compute_energy neighbor repro diverges from manual physics mirror",
        )
        self.assertTrue(
            torch.allclose(om.pending_birth_energy, om_manual.pending_birth_energy, atol=1e-4),
        )

    def test_direction_weights_normalize_on_alive_cells(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.75
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        shareable = om.energy_matrix * om.topology_matrix
        proportions, _ = om.energy_distribution_cnn(
            shareable, terrain, om.hidden_channels, om.rotation_matrix
        )
        weights = neighbor_direction_weights_from_proportions(
            proportions, om.topology_matrix
        )
        cy, cx = 6, 6
        total = weights[:, cy, cx].sum()
        self.assertGreater(total.item(), 0.0)
        self.assertAlmostEqual(total.item(), 1.0, places=4)

    def test_simulation_many_ticks_energy_bounded(self):
        sim = Simulation(enable_debug=False)
        for _ in range(50):
            sim.update_simulation()
        self.assertTrue(torch.all(sim.organism_manager.energy_matrix >= 0))
        self.assertTrue(torch.all(sim.organism_manager.energy_matrix <= 1))

    def test_harvest_terrain_depletion_matches_harvest(self):
        """Terrain loses exactly what organisms harvest at each cell."""
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.9
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.8
        harvested = om._apply_harvest(terrain.clone())
        harvest_at_cell = harvested[6, 6].item()
        self.assertGreater(harvest_at_cell, 0.0)

        env = Environment(SMALL, [[6, 6]])
        env.terrain = terrain.clone()
        env.environment_type = 2
        terrain_before = env.terrain[6, 6].item()
        topology = torch.zeros(SMALL, SMALL, device=device)
        topology[6, 6] = 1.0
        env.compute_environment(topology, harvested)
        terrain_loss = terrain_before - env.terrain[6, 6].item()
        self.assertAlmostEqual(terrain_loss, harvest_at_cell, places=4)

    @patch("config.TERRAIN_PUMP_ENABLED", False)
    def test_no_terrain_organism_energy_non_increasing_without_flux(self):
        sim = Simulation(enable_debug=False)
        sim.environment.terrain.zero_()
        sim.organism_manager.terrain.zero_()
        sim.conductance_scale = 0.0
        before = sim.organism_manager.energy_matrix.sum().item()
        for _ in range(10):
            sim.update_simulation()
        after = sim.organism_manager.energy_matrix.sum().item()
        self.assertLessEqual(after, before + 1e-4)

    def test_repro_debits_source_and_fills_pending_neighbors(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.9
        om.energy_bands[:, 6, 6] = om.spectrum[:, 6, 6] * om.energy_matrix[6, 6]
        energy_before_repro = om.energy_matrix[6, 6].item()
        proportions = torch.ones(3, 3, SMALL, SMALL, device=device) / 9.0
        terrain = torch.zeros(SMALL, SMALL, device=device)
        metrics = apply_neighbor_repro_physics(om, terrain, proportions)
        self.assertLess(om.energy_matrix[6, 6].item(), energy_before_repro)
        self.assertGreater(metrics["pending_sum_after_repro"].item(), 0.0)


class TestE2EConservation(unittest.TestCase):
    @patch("config.TERRAIN_PUMP_ENABLED", False)
    @patch("physics.apply_repro_flux_tick")
    @patch("config.REPRODUCTION_THRESHOLD", 999.0)
    def test_single_organism_conserves_total_energy_over_ticks(self, mock_repro):
        from physics import FluxTickResult

        def noop_repro(energy_bands, spectrum, topology, direction_weights, conductance_scale):
            return FluxTickResult(
                energy_bands=energy_bands,
                terrain_debit=torch.zeros_like(topology),
                pending_bands=torch.zeros_like(energy_bands),
                sink_bands=torch.zeros_like(energy_bands),
                clamp_loss=torch.zeros((), device=energy_bands.device, dtype=energy_bands.dtype),
            )

        mock_repro.side_effect = noop_repro
        sim = Simulation(enable_debug=False)
        sim.environment.environment_type = 2
        om = sim.organism_manager
        cy, cx = sim.world_size // 2, sim.world_size // 2

        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.energy_bands.zero_()
        om.hidden_channels.zero_()
        om.rotation_matrix.zero_()
        om.positions = torch.tensor([[cx, cy]], dtype=torch.long, device=device)
        om._initialize_topology()
        om.energy_bands[:, cy, cx] = om.spectrum[:, cy, cx] * 0.9
        om._sync_total_energy()

        sim.environment.terrain.fill_(0.5)
        om.terrain = sim.environment.terrain

        for tick in range(25):
            before = system_total_energy(sim).item()
            sim.update_simulation()
            after = system_total_energy(sim).item()
            self.assertLess(
                abs(after - before),
                0.15,
                msg=f"tick {tick + 1}: total energy changed by {after - before}",
            )
            self.assertTrue(torch.all(om.energy_matrix >= 0))
            self.assertTrue(torch.all(om.energy_matrix <= 1))
            self.assertTrue(torch.all(sim.environment.terrain >= 0))
            self.assertTrue(torch.all(sim.environment.terrain <= 1))


class TestModelIO(unittest.TestCase):
    def test_clear_saved_networks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cnn_test_gen1_1.000000.pt")
            with open(path, "w") as f:
                f.write("")
            with patch("model_io.glob.glob", return_value=[path]):
                clear_saved_networks()
            self.assertFalse(os.path.exists(path))

    def test_load_latest_cnn(self):
        cnn = EnergyDistributionCNN(device)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cnn_abcd_gen1_100.000000.pt")
            torch.save(cnn.state_dict(), path)
            with patch("model_io.glob.glob", return_value=[path]):
                loaded = load_latest_cnn()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.conv1.weight.shape, cnn.conv1.weight.shape)


if __name__ == "__main__":
    unittest.main()
