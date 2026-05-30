"""Unit tests for main.py — see specs/main.md for behavior specifications."""
import gc
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

import main as main_module
from main import (
    BasicCPPN,
    CNNGeneticAlgorithm,
    CNNEvaluator,
    CNNEvolutionDriver,
    EnergyDistributionCNN,
    Environment,
    OrganismManager,
    Renderer,
    Simulation,
    _evaluate_cnn_worker,
    _init_worker,
    _release_device_memory,
    clear_saved_networks,
    load_latest_cnn,
)
import oriented_conv
from config import ENERGY_DECAY, ENERGY_HARVEST_RATE

device = main_module.device
SMALL = 12


def make_organism_manager(world_size=SMALL, center=None):
    terrain = torch.ones(world_size, world_size, device=device) * 0.5
    om = OrganismManager(world_size, 1, terrain)
    if center is not None:
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.hidden_channels.zero_()
        om.rotation_matrix.zero_()
        om.parent_giver_dir.fill_(-1)
        cy, cx = center
        om.positions = torch.tensor([[cx, cy]], dtype=torch.long, device=device)
        om._initialize_topology()
    elif om.topology_matrix.sum().item() == 0:
        cy, cx = world_size // 2, world_size // 2
        om.positions = torch.tensor([[cx, cy]], dtype=torch.long, device=device)
        om._initialize_topology()
    return om


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


def apply_sharing_physics(om, terrain, proportions):
    """
    Mirror OrganismManager.compute_energy sharing steps (after harvest/decay).
    Returns metrics for thermodynamic assertions.
    """
    org_sum_before_harvest = om.energy_matrix.sum()
    harvested = om._apply_harvest_and_decay(terrain)
    org_sum_after_harvest = om.energy_matrix.sum()
    destroyed_before_sharing = om.destroyed_energy

    source_energy = om.energy_matrix.clone()
    shareable_energy = source_energy * om.topology_matrix * om.sharing_rate_matrix
    contributions = om._compute_energy_contributions(shareable_energy, proportions)
    total_outflow = contributions.sum(dim=(0, 1))

    full_distributed = om._shift_sum_contributions(contributions)
    parent_incoming = om._compute_parent_incoming(contributions)
    has_parent = (om.parent_giver_dir >= 0) & (om.topology_matrix > 0)
    is_seed = (om.topology_matrix > 0) & (om.parent_giver_dir < 0)
    distributed_total = full_distributed
    distributed_total = torch.where(has_parent, shareable_energy + parent_incoming, distributed_total)
    distributed_total = torch.where(is_seed, shareable_energy, distributed_total)

    om.new_cell_candidates = (distributed_total > om.reproduction_threshold) & (om.topology_matrix == 0)
    receiving_mask = (om.topology_matrix.bool() | om.new_cell_candidates).float()

    energy_incoming = distributed_total - shareable_energy
    capacity = (1.0 - om.energy_matrix) * receiving_mask
    actual_received = torch.min(energy_incoming, capacity)
    om.pending_birth_energy = actual_received * om.new_cell_candidates.float()
    dest_efficiency = torch.where(
        energy_incoming > 0,
        actual_received / energy_incoming,
        torch.zeros_like(energy_incoming),
    ) * receiving_mask
    source_removed = om._compute_source_removed(contributions, dest_efficiency, receiving_mask)
    total_received = actual_received.sum()
    total_removed = source_removed.sum()
    if total_removed > total_received:
        source_removed = source_removed * (total_received / total_removed)

    incoming = distributed_total - shareable_energy
    valid_mask = om.topology_matrix
    unclamped = (om.energy_matrix + incoming * dest_efficiency - source_removed) * valid_mask
    om.energy_matrix = torch.clamp(unclamped, 0, 1)
    sharing_clamp_loss = (unclamped - om.energy_matrix).sum()
    om.destroyed_energy += sharing_clamp_loss.item()

    return {
        "harvested": harvested,
        "shareable_energy": shareable_energy,
        "contributions": contributions,
        "total_outflow": total_outflow,
        "source_removed": source_removed,
        "actual_received": actual_received,
        "energy_incoming": energy_incoming,
        "parent_incoming": parent_incoming,
        "full_distributed": full_distributed,
        "distributed_total": distributed_total,
        "org_sum_before_harvest": org_sum_before_harvest,
        "org_sum_after_harvest": org_sum_after_harvest,
        "org_sum_after_sharing": om.energy_matrix.sum(),
        "pending_sum_after_sharing": om.pending_birth_energy.sum(),
        "sharing_destroyed": om.destroyed_energy - destroyed_before_sharing,
    }


class TestBasicCPPN(unittest.TestCase):
    def setUp(self):
        self.cppn = BasicCPPN(device)

    def test_forward_output_shape(self):
        coords = torch.randn(10, 3, device=device)
        out = self.cppn.forward(coords)
        self.assertEqual(out.shape, (10, 1))

    def test_generate_conv_weights_shape(self):
        w = self.cppn.generate_conv_weights(4, 8, 3)
        self.assertEqual(w.shape, (8, 4, 3, 3))

    def test_generate_bias_shape(self):
        b = self.cppn.generate_bias(11)
        self.assertEqual(b.shape, (11,))


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
        self.cnn.conv2.bias.data[10] = 5.0
        self.cnn._zero_hidden_channel_bias()
        self.assertEqual(self.cnn.conv2.bias.data[10].item(), 0.0)

    def test_forward_output_shapes(self):
        shareable = torch.rand(self.H, self.H, device=device)
        terrain = torch.rand(self.H, self.H, device=device)
        sharing = torch.ones(self.H, self.H, device=device)
        hidden = torch.zeros(1, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        proportions, sharing_out, hidden_out = self.cnn(
            shareable, terrain, sharing, hidden, rotation
        )
        self.assertEqual(proportions.shape, (3, 3, self.H, self.H))
        self.assertEqual(sharing_out.shape, (self.H, self.H))
        self.assertEqual(hidden_out.shape, (1, self.H, self.H))

    def test_proportions_sum_to_one(self):
        shareable = torch.ones(self.H, self.H, device=device)
        terrain = torch.ones(self.H, self.H, device=device) * 0.5
        sharing = torch.ones(self.H, self.H, device=device)
        hidden = torch.zeros(1, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        proportions, _, _ = self.cnn(shareable, terrain, sharing, hidden, rotation)
        sums = proportions.sum(dim=(0, 1))
        self.assertTrue(torch.allclose(sums, torch.ones(self.H, self.H, device=device), rtol=1e-4))

    def test_binary_outputs(self):
        shareable = torch.rand(self.H, self.H, device=device)
        terrain = torch.rand(self.H, self.H, device=device)
        sharing = torch.rand(self.H, self.H, device=device)
        hidden = torch.rand(1, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        _, sharing_out, hidden_out = self.cnn(shareable, terrain, sharing, hidden, rotation)
        self.assertTrue(torch.all((sharing_out == 0) | (sharing_out == 1)))
        self.assertTrue(torch.all((hidden_out == 0) | (hidden_out == 1)))

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
        inp = torch.randn(4, self.H, self.H, device=device)
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
    def test_contrib_accum_weight_maps_center(self):
        w = OrganismManager._make_contrib_accum_weight(device)
        self.assertEqual(w[0, 4, 1, 1].item(), 1.0)

    def test_dest_eff_gather_weight_maps_center(self):
        w = OrganismManager._make_dest_eff_gather_weight(device)
        self.assertEqual(w[4, 0, 1, 1].item(), 1.0)


class TestOrganismManagerEnergy(unittest.TestCase):
    def setUp(self):
        self.om = make_organism_manager(SMALL)

    def test_initialize_topology_places_seed(self):
        om = make_organism_manager(SMALL, center=(SMALL // 2, SMALL // 2))
        self.assertGreater(om.topology_matrix.sum().item(), 0)
        self.assertGreater(om.energy_matrix.sum().item(), 0)

    def test_compute_energy_contributions_shape(self):
        shareable = torch.ones(SMALL, SMALL, device=device)
        proportions = torch.ones(3, 3, SMALL, SMALL, device=device) / 9.0
        c = self.om._compute_energy_contributions(shareable, proportions)
        self.assertEqual(c.shape, (3, 3, SMALL, SMALL))

    def test_shift_sum_accumulates_neighbor_contribution(self):
        contributions = torch.zeros(3, 3, SMALL, SMALL, device=device)
        cy, cx = 6, 6
        contributions[2, 1, cy - 1, cx] = 2.0
        total = self.om._shift_sum_contributions(contributions)
        self.assertAlmostEqual(total[cy, cx].item(), 2.0, places=4)

    def test_apply_capacity_constraints(self):
        energy = torch.tensor([[0.9]], device=device)
        receiving = torch.tensor([[1.0]], device=device)
        new_energy = torch.tensor([[1.5]], device=device)
        self.om.energy_matrix = energy
        result, received, incoming = self.om._apply_capacity_constraints(new_energy, receiving)
        self.assertAlmostEqual(result[0, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(received[0, 0].item(), 0.1, places=4)

    def test_death_clears_cell_state(self):
        y, x = 6, 6
        self.om.topology_matrix[y, x] = 1
        self.om.energy_matrix[y, x] = 0.01
        self.om.parent_giver_dir[y, x] = 3
        self.om.sharing_rate_matrix[y, x] = 1.0
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        self.om._apply_harvest_and_decay(terrain)
        self.assertEqual(self.om.topology_matrix[y, x].item(), 0)
        self.assertEqual(self.om.parent_giver_dir[y, x].item(), -1)

    def test_birth_sets_parent_and_sharing_from_parent_hidden(self):
        y, x = 6, 6
        py, px = 5, 6
        self.om.topology_matrix[py, px] = 1
        self.om.hidden_channels[0, py, px] = 1.0
        self.om.new_cell_candidates = torch.zeros(SMALL, SMALL, dtype=torch.bool, device=device)
        self.om.new_cell_candidates[y, x] = True
        self.om.pending_birth_energy[y, x] = 0.5
        ring = EnergyDistributionCNN._RING_CIJ
        contrib = torch.zeros(3, 3, SMALL, SMALL, device=device)
        ci, cj = ring[(6 + 4) % 8]
        contrib[ci, cj, py, px] = 5.0
        self.om.new_cell_contributions = contrib
        self.om.compute_topology()
        self.assertEqual(self.om.topology_matrix[y, x].item(), 1)
        self.assertEqual(self.om.parent_giver_dir[y, x].item(), 6)
        self.assertEqual(self.om.sharing_rate_matrix[y, x].item(), 1.0)

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

    def test_sharing_rate_stable_after_birth(self):
        sim = Simulation(enable_debug=False)
        om = sim.organism_manager
        for _ in range(30):
            sim.update_simulation()
        alive = om.topology_matrix > 0
        if alive.any():
            snap = om.sharing_rate_matrix[alive].clone()
            for _ in range(10):
                sim.update_simulation()
            self.assertTrue(torch.equal(om.sharing_rate_matrix[alive], snap))

    def test_parent_incoming_reads_from_parent_not_candidate(self):
        ring = EnergyDistributionCNN._RING_CIJ
        y, x = 6, 6
        py, px = 5, 6
        self.om.parent_giver_dir[y, x] = 6
        contrib = torch.zeros(3, 3, SMALL, SMALL, device=device)
        ci, cj = ring[(6 + 4) % 8]
        contrib[ci, cj, py, px] = 3.0
        contrib[ci, cj, y, x] = 99.0
        incoming = self.om._compute_parent_incoming(contrib)
        self.assertAlmostEqual(incoming[y, x].item(), 3.0, places=4)


class TestEnvironment(unittest.TestCase):
    def test_generate_terrain_sine_in_range(self):
        env = Environment(SMALL, 0.01, 0.01)
        env.environment_type = 2
        terrain = env._generate_sine_terrain()
        self.assertEqual(terrain.shape, (SMALL, SMALL))
        self.assertTrue(torch.all(terrain >= 0))
        self.assertTrue(torch.all(terrain <= 1))

    def test_generate_terrain_dispatches_type(self):
        env = Environment(SMALL, 0.01, 0.01)
        env.environment_type = 1
        t1 = env.generate_terrain()
        env.environment_type = 2
        t2 = env.generate_terrain()
        self.assertEqual(t1.shape, (SMALL, SMALL))
        self.assertEqual(t2.shape, (SMALL, SMALL))

    def test_compute_environment_depletes_terrain(self):
        env = Environment(SMALL, 0.01, 0.01)
        env.environment_type = 2
        before = env.terrain.clone()
        topology = torch.zeros(SMALL, SMALL, device=device)
        topology[6, 6] = 1
        harvested = torch.zeros(SMALL, SMALL, device=device)
        harvested[6, 6] = 1.0
        env.compute_environment(topology, harvested)
        self.assertLess(env.terrain[6, 6].item(), before[6, 6].item())


class TestRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = Renderer(SMALL)

    def test_render_shape(self):
        env = torch.rand(SMALL, SMALL, device=device)
        topo = torch.zeros(SMALL, SMALL, device=device)
        mask = torch.zeros(SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, mask)
        self.assertEqual(image.shape, (4, SMALL, SMALL))

    def test_organism_sharing_off_is_red(self):
        env = torch.zeros(SMALL, SMALL, device=device)
        topo = torch.zeros(SMALL, SMALL, device=device)
        topo[6, 6] = 1
        sharing = torch.zeros(SMALL, SMALL, device=device)
        hidden = torch.zeros(1, SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, topo, sharing_rate=sharing, hidden_channels=hidden)
        self.assertGreater(image[0, 6, 6].item(), 0.9)
        self.assertLess(image[1, 6, 6].item(), 0.1)

    def test_organism_sharing_on_hidden_off_is_white(self):
        env = torch.zeros(SMALL, SMALL, device=device)
        topo = torch.zeros(SMALL, SMALL, device=device)
        topo[6, 6] = 1
        sharing = torch.ones(SMALL, SMALL, device=device)
        hidden = torch.zeros(1, SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, topo, sharing_rate=sharing, hidden_channels=hidden)
        self.assertGreater(image[0, 6, 6].item(), 0.9)
        self.assertGreater(image[1, 6, 6].item(), 0.9)

    def test_background_is_cyan(self):
        env = torch.ones(SMALL, SMALL, device=device) * 0.8
        topo = torch.zeros(SMALL, SMALL, device=device)
        mask = torch.zeros(SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, mask, sharing_rate=torch.zeros(SMALL, SMALL, device=device))
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
        for key in ("terrain", "topology", "energy", "new_cell_candidates", "sharing_rate", "hidden_channels", "pending_birth_energy", "destroyed_energy"):
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
        self.assertIsNotNone(main_module._worker_device)

    def test_evaluate_cnn_worker_short_run(self):
        _init_worker(str(device))
        cnn = EnergyDistributionCNN(device)
        cpu_state = {k: v.cpu().clone() for k, v in cnn.state_dict().items()}
        args = (cpu_state, SMALL, 3, 0, False)
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


class TestThermodynamics(unittest.TestCase):
    """Thermodynamic constraints on energy transfer, bounds, and bookkeeping."""

    def test_outflow_per_cell_equals_shareable(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.8
        om.sharing_rate_matrix[6, 6] = 1.0
        proportions = uniform_proportions(SMALL, 1, 2)
        shareable = om.energy_matrix[6, 6] * om.sharing_rate_matrix[6, 6]
        contributions = om._compute_energy_contributions(
            om.energy_matrix * om.topology_matrix * om.sharing_rate_matrix,
            proportions,
        )
        self.assertAlmostEqual(contributions[:, :, 6, 6].sum().item(), shareable.item(), places=4)

    def test_source_removed_equals_actual_received_globally(self):
        om = make_organism_manager(SMALL)
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        py, px, cy, cx = 6, 6, 6, 7
        for y, x in [(py, px), (cy, cx)]:
            om.topology_matrix[y, x] = 1
            om.energy_matrix[y, x] = 0.6
            om.sharing_rate_matrix[y, x] = 1.0
        om.parent_giver_dir[cy, cx] = 6
        proportions = uniform_proportions(SMALL, 1, 2)
        terrain = torch.zeros(SMALL, SMALL, device=device)
        metrics = apply_sharing_physics(om, terrain, proportions)
        removed = metrics["source_removed"].sum().item()
        received = metrics["actual_received"].sum().item()
        self.assertAlmostEqual(removed, received, places=4)

    def test_energy_bounded_zero_to_one_after_sharing(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.95
        om.sharing_rate_matrix[6, 6] = 1.0
        proportions = torch.ones(3, 3, SMALL, SMALL, device=device) / 9.0
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        apply_sharing_physics(om, terrain, proportions)
        self.assertTrue(torch.all(om.energy_matrix >= 0))
        self.assertTrue(torch.all(om.energy_matrix <= 1))

    def test_parent_only_blocks_non_parent_inflow(self):
        om = make_organism_manager(SMALL)
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        py, px, cy, cx = 6, 6, 6, 7
        om.topology_matrix[py, px] = 1
        om.topology_matrix[cy, cx] = 1
        om.energy_matrix[py, px] = 0.7
        om.energy_matrix[cy, cx] = 0.2
        om.sharing_rate_matrix[py, px] = 1.0
        om.sharing_rate_matrix[cy, cx] = 1.0
        om.parent_giver_dir[cy, cx] = 6
        proportions = uniform_proportions(SMALL, 1, 2)
        shareable = om.energy_matrix * om.topology_matrix * om.sharing_rate_matrix
        contributions = om._compute_energy_contributions(shareable, proportions)
        full_in = om._shift_sum_contributions(contributions)
        parent_in = om._compute_parent_incoming(contributions)
        self.assertGreater(full_in[cy, cx].item(), parent_in[cy, cx].item())

    def test_seed_receives_no_sharing_income(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.5
        om.sharing_rate_matrix[6, 6] = 1.0
        om.parent_giver_dir[6, 6] = -1
        neighbor_y, neighbor_x = 6, 7
        om.topology_matrix[neighbor_y, neighbor_x] = 1
        om.energy_matrix[neighbor_y, neighbor_x] = 0.8
        om.sharing_rate_matrix[neighbor_y, neighbor_x] = 1.0
        proportions = uniform_proportions(SMALL, 1, 0)
        terrain = torch.zeros(SMALL, SMALL, device=device)
        metrics = apply_sharing_physics(om, terrain, proportions)
        self.assertAlmostEqual(metrics["energy_incoming"][6, 6].item(), 0.0, places=4)

    def test_decay_destroys_energy_without_harvest(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.5
        om.sharing_rate_matrix[6, 6] = 1.0
        terrain = torch.zeros(SMALL, SMALL, device=device)
        before = om.energy_matrix.sum().item()
        with patch("main.ENERGY_DENSITY_DECAY_MODIFIER", 1.0):
            om._apply_harvest_and_decay(terrain)
        after = om.energy_matrix.sum().item()
        self.assertLess(after, before)

    @patch("main.ENERGY_DECAY", 0.0)
    def test_harvest_increases_organism_energy_when_decay_is_small(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.5
        om.sharing_rate_matrix[6, 6] = 0.05
        terrain = torch.ones(SMALL, SMALL, device=device)
        before = om.energy_matrix[6, 6].item()
        om._apply_harvest_and_decay(terrain)
        after = om.energy_matrix[6, 6].item()
        self.assertGreater(after, before)

    def test_harvest_bounded_by_rate_and_terrain(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.sharing_rate_matrix[6, 6] = 1.0
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.8
        harvested = om._apply_harvest_and_decay(terrain)
        self.assertLessEqual(harvested[6, 6].item(), ENERGY_HARVEST_RATE + 1e-6)
        self.assertLessEqual(harvested[6, 6].item(), 0.8 + 1e-6)

    def test_full_tick_depletes_environment_on_harvest(self):
        sim = Simulation(enable_debug=False)
        sim.environment.environment_type = 2
        terrain_before = sim.environment.terrain.clone()
        org_before = sim.organism_manager.energy_matrix.sum().item()
        harvested = sim.organism_manager.compute_energy(sim.environment.terrain)
        sim.environment.compute_environment(sim.organism_manager.topology_matrix, harvested)
        terrain_after = sim.environment.terrain.sum().item()
        org_after = sim.organism_manager.energy_matrix.sum().item()
        if harvested.sum().item() > 0:
            self.assertLessEqual(terrain_after, terrain_before.sum().item())
        self.assertTrue(torch.all(sim.organism_manager.energy_matrix >= 0))
        self.assertTrue(torch.all(sim.organism_manager.energy_matrix <= 1))

    def test_compute_energy_matches_manual_sharing_physics(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.7
        om.sharing_rate_matrix[6, 6] = 1.0
        terrain = torch.zeros(SMALL, SMALL, device=device)
        proportions = uniform_proportions(SMALL, 2, 1)

        om_manual = make_organism_manager(SMALL, center=(6, 6))
        om_manual.energy_matrix[6, 6] = 0.7
        om_manual.sharing_rate_matrix[6, 6] = 1.0
        apply_sharing_physics(om_manual, terrain.clone(), proportions)

        def fixed_forward(shareable, terr, sharing, hidden, rotation):
            return proportions, torch.zeros_like(sharing), hidden

        om.energy_matrix[6, 6] = 0.7
        om.sharing_rate_matrix[6, 6] = 1.0
        with patch.object(om.energy_distribution_cnn, "forward", side_effect=fixed_forward):
            om.compute_energy(terrain.clone())

        self.assertTrue(
            torch.allclose(om.energy_matrix, om_manual.energy_matrix, atol=1e-4),
            msg="compute_energy sharing result diverges from manual physics mirror",
        )

    def test_cnn_proportions_outflow_equals_shareable(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.75
        om.sharing_rate_matrix[6, 6] = 1.0
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        shareable = om.energy_matrix * om.topology_matrix * om.sharing_rate_matrix
        proportions, _, _ = om.energy_distribution_cnn(
            shareable, terrain, om.sharing_rate_matrix, om.hidden_channels, om.rotation_matrix
        )
        contributions = om._compute_energy_contributions(shareable, proportions)
        alive = om.topology_matrix > 0
        outflow = contributions.sum(dim=(0, 1))
        self.assertTrue(torch.allclose(outflow[alive], shareable[alive], rtol=1e-4))

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
        om.sharing_rate_matrix[6, 6] = 1.0
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.8
        harvested = om._apply_harvest_and_decay(terrain.clone())
        harvest_at_cell = harvested[6, 6].item()
        self.assertGreater(harvest_at_cell, 0.0)

        env = Environment(SMALL, 0.01, 0.01)
        env.terrain = terrain.clone()
        env.environment_type = 2
        terrain_before = env.terrain[6, 6].item()
        topology = torch.zeros(SMALL, SMALL, device=device)
        topology[6, 6] = 1.0
        env.compute_environment(topology, harvested)
        terrain_loss = terrain_before - env.terrain[6, 6].item()
        self.assertAlmostEqual(terrain_loss, harvest_at_cell, places=4)

    def test_no_terrain_organism_energy_non_increasing_from_decay(self):
        sim = Simulation(enable_debug=False)
        sim.environment.terrain.zero_()
        sim.organism_manager.terrain.zero_()
        before = sim.organism_manager.energy_matrix.sum().item()
        for _ in range(10):
            sim.update_simulation()
        after = sim.organism_manager.energy_matrix.sum().item()
        self.assertLessEqual(after, before + 1e-4)

    def test_source_removed_never_exceeds_outflow(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.9
        om.sharing_rate_matrix[6, 6] = 1.0
        proportions = torch.ones(3, 3, SMALL, SMALL, device=device) / 9.0
        terrain = torch.zeros(SMALL, SMALL, device=device)
        metrics = apply_sharing_physics(om, terrain, proportions)
        alive = om.topology_matrix > 0
        self.assertTrue(
            torch.all(metrics["source_removed"][alive] <= metrics["total_outflow"][alive] + 1e-5)
        )

    def test_sharing_net_change_equals_received_minus_removed(self):
        om = make_organism_manager(SMALL)
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        for y, x in [(6, 6), (6, 7), (7, 6)]:
            om.topology_matrix[y, x] = 1
            om.energy_matrix[y, x] = 0.5
            om.sharing_rate_matrix[y, x] = 1.0
        om.parent_giver_dir[6, 7] = 6
        om.parent_giver_dir[7, 6] = 4
        proportions = torch.ones(3, 3, SMALL, SMALL, device=device) / 9.0
        terrain = torch.zeros(SMALL, SMALL, device=device)
        metrics = apply_sharing_physics(om, terrain, proportions)
        net = (
            metrics["org_sum_after_sharing"]
            + metrics["pending_sum_after_sharing"]
            - metrics["org_sum_after_harvest"]
        )
        removed = metrics["source_removed"].sum().item()
        received = metrics["actual_received"].sum().item()
        sharing_destroyed = metrics["sharing_destroyed"]
        self.assertAlmostEqual(
            net.item() + sharing_destroyed,
            received - removed,
            places=3,
        )


class TestE2EConservation(unittest.TestCase):
    def test_single_organism_conserves_total_energy_over_ticks(self):
        sim = Simulation(enable_debug=False)
        sim.environment.environment_type = 2
        om = sim.organism_manager
        cy, cx = sim.world_size // 2, sim.world_size // 2

        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.hidden_channels.zero_()
        om.rotation_matrix.zero_()
        om.parent_giver_dir.fill_(-1)
        om.positions = torch.tensor([[cx, cy]], dtype=torch.long, device=device)
        om._initialize_topology()
        om.energy_matrix[cy, cx] = 0.9

        sim.environment.terrain.fill_(0.5)
        om.terrain = sim.environment.terrain

        for tick in range(25):
            before = system_total_energy(sim).item()
            sim.update_simulation()
            after = system_total_energy(sim).item()
            self.assertAlmostEqual(
                before,
                after,
                places=3,
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
            with patch("main.glob.glob", return_value=[path]):
                clear_saved_networks()
            self.assertFalse(os.path.exists(path))

    def test_load_latest_cnn(self):
        cnn = EnergyDistributionCNN(device)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cnn_abcd_gen1_100.000000.pt")
            torch.save(cnn.state_dict(), path)
            with patch("main.glob.glob", return_value=[path]):
                loaded = load_latest_cnn()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.conv1.weight.shape, cnn.conv1.weight.shape)


if __name__ == "__main__":
    unittest.main()
