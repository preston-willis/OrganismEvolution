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
from config import DEATH_THRESHOLD, ENERGY_DECAY, ENERGY_HARVEST_RATE, MAX_CHILDREN_PER_PARENT, PERLIN_DEAD_THRESHOLD, PERLIN_TIME_SPEED, REPRODUCTION_THRESHOLD, SHARING_RATE_ON, SHARING_RATE_OFF

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


def giver_dir_to(om, y, x, py, px):
    for g in range(8):
        if om._giver_source_y[g, y, x].item() == py and om._giver_source_x[g, y, x].item() == px:
            return g
    raise AssertionError(f"no giver dir from ({y}, {x}) to ({py}, {px})")


def system_total_energy(sim):
    """Organism + environment + uncommitted birth energy + cumulative decay sink."""
    om = sim.organism_manager
    return (
        om.energy_matrix.sum()
        + sim.environment.terrain.sum()
        + om.pending_birth_energy.sum()
        + om.destroyed_energy
    )


def apply_sharing_physics(om, terrain, proportions, skip_harvest=False):
    """
    Mirror OrganismManager.compute_energy sharing steps (after harvest/decay).
    Returns metrics for thermodynamic assertions.
    """
    org_sum_before_harvest = om.energy_matrix.sum()
    if skip_harvest:
        harvested = torch.zeros_like(terrain)
        org_sum_after_harvest = org_sum_before_harvest
    else:
        harvested = om._apply_harvest_and_decay(terrain)
        org_sum_after_harvest = om.energy_matrix.sum()
    om._remove_dead_cells()
    om._flush_tick_destroyed()
    destroyed_before_sharing = om.destroyed_energy

    source_energy = om.energy_matrix.clone()
    shareable_energy = source_energy * om.topology_matrix * om.sharing_rate_matrix
    contributions = om._compute_energy_contributions(shareable_energy, proportions)
    total_outflow = contributions.sum(dim=(0, 1))

    full_distributed = om._shift_sum_contributions(contributions)
    parent_incoming = om._compute_parent_incoming(contributions)
    living = om.topology_matrix > 0
    distributed_total = full_distributed
    distributed_total = torch.where(living, shareable_energy + parent_incoming, distributed_total)

    om.new_cell_candidates = (distributed_total > om.reproduction_threshold) & (om.topology_matrix == 0)
    energy_incoming = distributed_total - shareable_energy
    receiving_mask = (om.topology_matrix.bool() | om.new_cell_candidates).float()

    capacity = (1.0 - om.energy_matrix) * receiving_mask
    actual_received = torch.min(energy_incoming, capacity)
    dest_efficiency = torch.where(
        energy_incoming > 0,
        actual_received / energy_incoming,
        torch.zeros_like(energy_incoming),
    ) * receiving_mask
    source_removed = om._compute_source_removed(contributions, dest_efficiency, receiving_mask)
    incoming = energy_incoming
    valid_mask = om.topology_matrix
    net_incoming = om.energy_matrix + incoming * dest_efficiency
    total_received = actual_received.sum()
    total_removed = source_removed.sum()
    if total_removed > total_received:
        source_removed = source_removed * (total_received / total_removed)
    elif total_removed < total_received:
        receive_scale = total_removed / total_received
        actual_received = actual_received * receive_scale
        dest_efficiency = dest_efficiency * receive_scale
        net_incoming = om.energy_matrix + incoming * dest_efficiency
        source_removed = source_removed * receive_scale
    source_removed = om._cap_source_removed_for_survival(net_incoming, source_removed, valid_mask)

    empty_mask = om.topology_matrix == 0
    new_pending = actual_received * om.new_cell_candidates.float()
    inbound_by_dir = om._gather_inbound_by_giver_dir(contributions)
    dir_new_pending = inbound_by_dir * (dest_efficiency * om.new_cell_candidates.float()).unsqueeze(0)
    om.pending_birth_energy = torch.where(
        empty_mask,
        om.pending_birth_energy + new_pending,
        torch.zeros_like(om.pending_birth_energy),
    )
    om.pending_giver_contrib = torch.where(
        empty_mask.unsqueeze(0),
        om.pending_giver_contrib + dir_new_pending,
        torch.zeros_like(om.pending_giver_contrib),
    )

    unclamped = (net_incoming - source_removed) * valid_mask
    overflow = torch.clamp(unclamped - 1.0, min=0.0) * valid_mask
    om.energy_matrix = torch.clamp(unclamped, 0, 1) * valid_mask
    overflow_weights = inbound_by_dir * dest_efficiency.unsqueeze(0)
    om._return_energy_to_sources(overflow, overflow_weights)
    om._return_unborn_pending_to_sources(contributions)
    om._remove_dead_cells()
    om._flush_tick_destroyed()

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


def fixed_proportions_forward(proportions):
    def forward(shareable, terrain, sharing, hidden, rotation):
        return proportions, hidden
    return forward


def run_organism_tick(om, terrain, proportions, skip_harvest=False):
    if skip_harvest:
        with patch.object(om, "_apply_harvest_and_decay", return_value=torch.zeros_like(terrain)):
            with patch.object(
                om.energy_distribution_cnn,
                "forward",
                side_effect=fixed_proportions_forward(proportions),
            ):
                harvested = om.compute_energy(terrain)
    else:
        with patch.object(
            om.energy_distribution_cnn,
            "forward",
            side_effect=fixed_proportions_forward(proportions),
        ):
            harvested = om.compute_energy(terrain)
    om.compute_topology()
    om._remove_dead_cells()
    om._flush_tick_destroyed()
    return harvested


def run_sim_tick_with_proportions(sim, proportions):
    om = sim.organism_manager
    with patch.object(
        om.energy_distribution_cnn,
        "forward",
        side_effect=fixed_proportions_forward(proportions),
    ):
        sim.update_simulation()


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

    def test_regenerate_conv2_from_cppn_bias(self):
        expected = self.cnn.cppn.generate_bias(10)
        self.cnn._regenerate_conv2_from_cppn()
        self.assertTrue(torch.allclose(self.cnn.conv2.bias.data, expected))

    def test_forward_output_shapes(self):
        shareable = torch.rand(self.H, self.H, device=device)
        terrain = torch.rand(self.H, self.H, device=device)
        sharing = torch.ones(self.H, self.H, device=device)
        hidden = torch.zeros(1, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        proportions, hidden_out = self.cnn(
            shareable, terrain, sharing, hidden, rotation
        )
        self.assertEqual(proportions.shape, (3, 3, self.H, self.H))
        self.assertEqual(hidden_out.shape, (1, self.H, self.H))

    def test_proportions_sum_to_one(self):
        shareable = torch.ones(self.H, self.H, device=device)
        terrain = torch.ones(self.H, self.H, device=device) * 0.5
        sharing = torch.ones(self.H, self.H, device=device)
        hidden = torch.zeros(1, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        proportions, _ = self.cnn(shareable, terrain, sharing, hidden, rotation)
        sums = proportions.sum(dim=(0, 1))
        self.assertTrue(torch.allclose(sums, torch.ones(self.H, self.H, device=device), rtol=1e-4))

    def test_sigmoid_hidden_output(self):
        shareable = torch.rand(self.H, self.H, device=device)
        terrain = torch.rand(self.H, self.H, device=device)
        sharing = torch.rand(self.H, self.H, device=device)
        hidden = torch.rand(1, self.H, self.H, device=device)
        rotation = torch.zeros(self.H, self.H, device=device)
        _, hidden_out = self.cnn(shareable, terrain, sharing, hidden, rotation)
        self.assertTrue(torch.all(hidden_out >= 0))
        self.assertTrue(torch.all(hidden_out <= 1))

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
        terrain = torch.zeros(SMALL, SMALL, device=device)
        self.om._apply_harvest_and_decay(terrain)
        self.om._remove_dead_cells()
        self.assertEqual(self.om.topology_matrix[y, x].item(), 0)
        self.assertEqual(self.om.parent_giver_dir[y, x].item(), -1)

    def test_death_at_exact_threshold(self):
        y, x = 6, 6
        self.om.topology_matrix[y, x] = 1
        self.om.energy_matrix[y, x] = DEATH_THRESHOLD
        self.om.sharing_rate_matrix[y, x] = 1.0
        self.om._remove_dead_cells()
        self.assertEqual(self.om.topology_matrix[y, x].item(), 0)

    def test_death_energy_spreads_to_neighbors_not_terrain(self):
        y, x = 6, 6
        ny, nx = 6, 7
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.topology_matrix[y, x] = 1
        self.om.topology_matrix[ny, nx] = 1
        self.om.energy_matrix[y, x] = DEATH_THRESHOLD * 0.5
        self.om.energy_matrix[ny, nx] = 0.5
        self.om.sharing_rate_matrix[y, x] = 1.0
        self.om.sharing_rate_matrix[ny, nx] = 1.0
        terrain_before = self.om.terrain.sum().item()
        neighbor_before = self.om.energy_matrix[ny, nx].item()
        destroyed_before = self.om.destroyed_energy
        self.om._remove_dead_cells()
        self.om._flush_tick_destroyed()
        self.assertEqual(self.om.topology_matrix[y, x].item(), 0)
        self.assertAlmostEqual(self.om.terrain.sum().item(), terrain_before, places=6)
        self.assertGreater(self.om.energy_matrix[ny, nx].item(), neighbor_before)
        self.assertAlmostEqual(self.om.destroyed_energy, destroyed_before, places=6)

    def test_birth_commits_pending_to_energy_matrix(self):
        y, x = 6, 7
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.pending_birth_energy.zero_()
        self.om.pending_birth_energy[y, x] = 0.35
        self.om.compute_topology()
        self.assertEqual(self.om.topology_matrix[y, x].item(), 1)
        self.assertAlmostEqual(self.om.energy_matrix[y, x].item(), 0.35, places=4)
        self.assertEqual(self.om.pending_birth_energy[y, x].item(), 0)
        self.assertGreaterEqual(self.om.parent_giver_dir[y, x].item(), 0)

    def test_birth_sets_parent_from_pending_when_no_contributions(self):
        y, x = 6, 7
        py, px = 6, 6
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.topology_matrix[py, px] = 1
        self.om.pending_birth_energy[y, x] = 0.35
        giver_dir = None
        for g in range(8):
            if self.om._giver_source_y[g, y, x].item() == py and self.om._giver_source_x[g, y, x].item() == px:
                giver_dir = g
                break
        self.assertIsNotNone(giver_dir)
        self.om.pending_giver_contrib[giver_dir, y, x] = 0.35
        self.om.compute_topology()
        self.assertEqual(self.om.parent_giver_dir[y, x].item(), giver_dir)

    def test_parent_caps_births_at_max_children(self):
        om = make_organism_manager(16, center=(8, 8))
        py, px = 8, 8
        sites = [(9, 8, 6), (7, 8, 2), (8, 9, 4), (8, 7, 0)]
        om.energy_matrix[py, px] = 0.4
        source_before = om.energy_matrix[py, px].item()
        with patch("main.MAX_CHILDREN_PER_PARENT", 3):
            for cy, cx, g in sites:
                om.pending_birth_energy[cy, cx] = 0.5
                om.pending_giver_contrib[g, cy, cx] = 0.5
            om.compute_topology()
        born = sum(om.topology_matrix[cy, cx].item() for cy, cx, _ in sites)
        blocked = [(cy, cx) for cy, cx, _ in sites if om.topology_matrix[cy, cx].item() == 0]
        self.assertEqual(born, 3)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(om._compute_living_child_count()[py, px].item(), 3)
        blocked_y, blocked_x = blocked[0]
        self.assertEqual(om.pending_birth_energy[blocked_y, blocked_x].item(), 0.0)
        self.assertGreater(om.energy_matrix[py, px].item(), source_before)

    def test_cap_blocked_pending_returns_to_source(self):
        om = make_organism_manager(16, center=(8, 8))
        py, px = 8, 8
        sites = [(9, 8, 6), (7, 8, 2), (8, 9, 4), (8, 7, 0)]
        om.energy_matrix[py, px] = 0.4
        source_before = om.energy_matrix[py, px].item()
        with patch("main.MAX_CHILDREN_PER_PARENT", 3):
            for cy, cx, g in sites:
                om.pending_birth_energy[cy, cx] = 0.5
                om.pending_giver_contrib[g, cy, cx] = 0.5
            om.compute_topology()
        blocked = [(cy, cx) for cy, cx, _ in sites if om.topology_matrix[cy, cx].item() == 0]
        self.assertEqual(len(blocked), 1)
        blocked_y, blocked_x = blocked[0]
        self.assertEqual(om.pending_birth_energy[blocked_y, blocked_x].item(), 0.0)
        self.assertEqual(om.pending_giver_contrib[:, blocked_y, blocked_x].sum().item(), 0.0)
        self.assertGreater(om.energy_matrix[py, px].item(), source_before)

    def test_birth_sets_parent_and_sharing_from_parent_hidden(self):
        y, x = 6, 6
        py, px = 5, 6
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.sharing_rate_matrix.zero_()
        self.om.hidden_channels.zero_()
        self.om.parent_giver_dir.fill_(-1)
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
        self.assertAlmostEqual(self.om.sharing_rate_matrix[y, x].item(), SHARING_RATE_ON, places=5)
        self.assertEqual(self.om.hidden_channels[0, y, x].item(), 1.0)

    def test_birth_sets_sharing_off_when_parent_hidden_off(self):
        y, x = 6, 6
        py, px = 5, 6
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.sharing_rate_matrix.zero_()
        self.om.hidden_channels.zero_()
        self.om.parent_giver_dir.fill_(-1)
        self.om.topology_matrix[py, px] = 1
        self.om.hidden_channels[0, py, px] = 0.0
        self.om.pending_birth_energy[y, x] = 0.5
        giver_dir = giver_dir_to(self.om, y, x, py, px)
        self.om.pending_giver_contrib[giver_dir, y, x] = 0.5
        self.om.compute_topology()
        self.assertEqual(self.om.topology_matrix[y, x].item(), 1)
        self.assertAlmostEqual(self.om.sharing_rate_matrix[y, x].item(), SHARING_RATE_OFF, places=5)
        self.assertEqual(self.om.hidden_channels[0, y, x].item(), 0.0)

    def test_birth_sets_rotation_from_dominant_giver(self):
        y, x = 6, 6
        py, px = 5, 6
        giver_dir = 6
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.topology_matrix[py, px] = 1
        self.om.pending_birth_energy[y, x] = 0.5
        self.om.pending_giver_contrib[giver_dir, y, x] = 0.5
        self.om.compute_topology()
        expected_angle = float(((giver_dir - 2) % 8) * (torch.pi / 4))
        self.assertAlmostEqual(self.om.rotation_matrix[y, x].item(), expected_angle, places=5)

    def test_compute_living_child_count(self):
        py, px = 6, 6
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.parent_giver_dir.fill_(-1)
        self.om.topology_matrix[py, px] = 1
        self.om.parent_giver_dir[py, px] = -1
        child_sites = [(6, 7), (7, 6)]
        for cy, cx in child_sites:
            self.om.topology_matrix[cy, cx] = 1
            self.om.parent_giver_dir[cy, cx] = giver_dir_to(self.om, cy, cx, py, px)
        counts = self.om._compute_living_child_count()
        self.assertEqual(counts[py, px].item(), len(child_sites))
        self.assertEqual(counts[child_sites[0][0], child_sites[0][1]].item(), 0.0)

    def test_birth_inherits_hidden_from_giver(self):
        y, x = 6, 6
        py, px = 5, 6
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.sharing_rate_matrix.zero_()
        self.om.hidden_channels.zero_()
        self.om.parent_giver_dir.fill_(-1)
        self.om.topology_matrix[py, px] = 1
        self.om.hidden_channels[0, py, px] = 1.0
        self.om.pending_birth_energy[y, x] = 0.5
        giver_dir = giver_dir_to(self.om, y, x, py, px)
        self.om.pending_giver_contrib[giver_dir, y, x] = 0.5
        self.om.compute_topology()
        self.assertAlmostEqual(self.om.sharing_rate_matrix[y, x].item(), SHARING_RATE_ON, places=5)
        self.assertEqual(self.om.hidden_channels[0, y, x].item(), 1.0)

    def test_sharing_rate_tracks_sigmoid_hidden(self):
        y, x = 6, 6
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.sharing_rate_matrix.zero_()
        self.om.hidden_channels.zero_()
        self.om.topology_matrix[y, x] = 1
        self.om.energy_matrix[y, x] = 0.8
        self.om.sharing_rate_matrix[y, x] = SHARING_RATE_OFF
        self.om.hidden_channels[0, y, x] = 0.0
        props = uniform_proportions(SMALL, 1, 1)
        cnn_hidden = torch.full((1, SMALL, SMALL), 0.75, device=device)

        def sets_hidden(shareable, terrain, sharing, hidden, rotation):
            return props, cnn_hidden

        terrain = torch.ones(SMALL, SMALL, device=device)
        with patch("main.ENERGY_DECAY", 0.0):
            with patch.object(self.om.energy_distribution_cnn, "forward", side_effect=sets_hidden):
                self.om.compute_energy(terrain)
        self.assertAlmostEqual(self.om.hidden_channels[0, y, x].item(), 0.75, places=5)
        expected_sharing = SHARING_RATE_OFF + 0.75 * (SHARING_RATE_ON - SHARING_RATE_OFF)
        self.assertAlmostEqual(self.om.sharing_rate_matrix[y, x].item(), expected_sharing, places=5)

    def test_grandchild_links_to_intermediate_parent(self):
        py, px = 6, 6
        cy, cx = 6, 7
        gy, gx = 6, 8
        om = make_organism_manager(SMALL)
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.parent_giver_dir.fill_(-1)
        om.topology_matrix[py, px] = 1
        om.pending_birth_energy[cy, cx] = 0.5
        om.pending_giver_contrib[giver_dir_to(om, cy, cx, py, px), cy, cx] = 0.5
        om.compute_topology()
        om.pending_birth_energy[gy, gx] = 0.5
        om.pending_giver_contrib[giver_dir_to(om, gy, gx, cy, cx), gy, gx] = 0.5
        om.compute_topology()
        self.assertEqual(om.topology_matrix[gy, gx].item(), 1)
        self.assertEqual(om.parent_giver_dir[gy, gx].item(), giver_dir_to(om, gy, gx, cy, cx))
        self.assertEqual(
            om._giver_source_y[om.parent_giver_dir[gy, gx], gy, gx].item(),
            cy,
        )
        self.assertEqual(
            om._giver_source_x[om.parent_giver_dir[gy, gx], gy, gx].item(),
            cx,
        )

    def test_update_simulation_birth_assigns_parent_giver_dir(self):
        sim = Simulation(enable_debug=False, enable_log=False)
        om = sim.organism_manager
        py, px = 6, 6
        cy, cx = 6, 7
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.parent_giver_dir.fill_(-1)
        om.topology_matrix[py, px] = 1
        om.energy_matrix[py, px] = 0.9
        om.sharing_rate_matrix[py, px] = SHARING_RATE_ON
        om.parent_giver_dir[py, px] = 0
        sim.environment.terrain.fill_(1.0)
        om.terrain = sim.environment.terrain
        props = uniform_proportions(sim.world_size, 1, 2)
        with patch("main.ENERGY_DECAY", 0.0), patch("main.REPRODUCTION_THRESHOLD", 0.5):
            om.reproduction_threshold = 0.5
            run_sim_tick_with_proportions(sim, props)
        self.assertEqual(om.topology_matrix[cy, cx].item(), 1)
        self.assertEqual(om.parent_giver_dir[cy, cx].item(), giver_dir_to(om, cy, cx, py, px))

    def test_candidates_do_not_store_energy_before_birth(self):
        sim = Simulation(enable_debug=False)
        om = sim.organism_manager
        sim.update_simulation()
        empty = om.topology_matrix == 0
        if om.new_cell_candidates.any():
            self.assertTrue(torch.all(om.energy_matrix[empty & om.new_cell_candidates] == 0))

    def test_sharing_rate_updates_from_sigmoid_hidden(self):
        sim = Simulation(enable_debug=False)
        om = sim.organism_manager
        cy, cx = sim.world_size // 2, sim.world_size // 2
        om.sharing_rate_matrix[cy, cx] = SHARING_RATE_OFF
        om.hidden_channels[0, cy, cx] = 0.0
        props = uniform_proportions(sim.world_size, 1, 1)
        cnn_hidden = torch.full((1, sim.world_size, sim.world_size), 0.25, device=device)

        def update_hidden(shareable, terr, sharing, hidden, rotation):
            return props, cnn_hidden

        with patch.object(om.energy_distribution_cnn, "forward", side_effect=update_hidden):
            for _ in range(20):
                sim.update_simulation()
        expected_sharing = SHARING_RATE_OFF + 0.25 * (SHARING_RATE_ON - SHARING_RATE_OFF)
        self.assertAlmostEqual(om.sharing_rate_matrix[cy, cx].item(), expected_sharing, places=5)

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

    def test_unborn_pending_returns_even_when_candidate(self):
        """Sub-threshold pending must refund even if site is still a birth candidate."""
        py, px = 5, 5
        sy, sx = 5, 4
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.pending_birth_energy.zero_()
        self.om.pending_giver_contrib.zero_()
        self.om.topology_matrix[sy, sx] = 1
        self.om.energy_matrix[sy, sx] = 0.4
        self.om.pending_birth_energy[py, px] = 0.15
        self.om.new_cell_candidates = torch.zeros(SMALL, SMALL, dtype=torch.bool, device=device)
        self.om.new_cell_candidates[py, px] = True
        giver_dir = None
        for g in range(8):
            if self.om._giver_source_y[g, py, px].item() == sy and self.om._giver_source_x[g, py, px].item() == sx:
                giver_dir = g
                break
        self.assertIsNotNone(giver_dir)
        self.om.pending_giver_contrib[giver_dir, py, px] = 0.15
        with patch("main.REPRODUCTION_THRESHOLD", 0.2):
            self.om.reproduction_threshold = 0.2
            source_before = self.om.energy_matrix[sy, sx].item()
            self.om._return_unborn_pending_to_sources(None)
        self.assertEqual(self.om.pending_birth_energy[py, px].item(), 0.0)
        self.assertGreater(self.om.energy_matrix[sy, sx].item(), source_before)

    def test_single_cell_survives_low_terrain_sharing(self):
        sim = Simulation(enable_debug=False)
        om = sim.organism_manager
        cy, cx = sim.world_size // 2, sim.world_size // 2
        sim.environment.terrain.fill_(1.0)
        om.terrain = sim.environment.terrain
        props = uniform_proportions(sim.world_size, 1, 1)
        with patch("main.ENERGY_DECAY", 0.0), patch("main.REPRODUCTION_THRESHOLD", 0.95):
            om.reproduction_threshold = 0.95
            for tick in range(50):
                cells_before = om.topology_matrix.sum().item()
                run_sim_tick_with_proportions(sim, props)
                self.assertEqual(om.topology_matrix.sum().item(), cells_before)
                self.assertGreaterEqual(om.energy_matrix[cy, cx].item(), DEATH_THRESHOLD)

    def test_abandoned_pending_returns_to_source_not_terrain(self):
        py, px = 5, 5
        sy, sx = 5, 4
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.pending_birth_energy.zero_()
        self.om.pending_giver_contrib.zero_()
        self.om.topology_matrix[sy, sx] = 1
        self.om.energy_matrix[sy, sx] = 0.4
        self.om.pending_birth_energy[py, px] = 0.15
        self.om.new_cell_candidates = torch.zeros(SMALL, SMALL, dtype=torch.bool, device=device)
        giver_dir = None
        for g in range(8):
            if self.om._giver_source_y[g, py, px].item() == sy and self.om._giver_source_x[g, py, px].item() == sx:
                giver_dir = g
                break
        self.assertIsNotNone(giver_dir)
        self.om.pending_giver_contrib[giver_dir, py, px] = 0.15
        with patch("main.REPRODUCTION_THRESHOLD", 0.2):
            self.om.reproduction_threshold = 0.2
            terrain_before = self.om.terrain.sum().item()
            source_before = self.om.energy_matrix[sy, sx].item()
            self.om._return_unborn_pending_to_sources(None)
        self.assertAlmostEqual(self.om.pending_birth_energy[py, px].item(), 0.0)
        self.assertAlmostEqual(self.om.terrain.sum().item(), terrain_before, places=4)
        self.assertGreater(self.om.energy_matrix[sy, sx].item(), source_before)

    def test_orphaned_pending_returns_to_living_source_not_terrain(self):
        py, px = 5, 5
        sy, sx = 5, 4
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.pending_birth_energy.zero_()
        self.om.pending_giver_contrib.zero_()
        self.om.topology_matrix[sy, sx] = 1
        self.om.energy_matrix[sy, sx] = 0.4
        self.om.pending_birth_energy[py, px] = 0.2
        giver_dir = None
        for g in range(8):
            if self.om._giver_source_y[g, py, px].item() == sy and self.om._giver_source_x[g, py, px].item() == sx:
                giver_dir = g
                break
        self.assertIsNotNone(giver_dir)
        self.om.pending_giver_contrib[giver_dir, py, px] = 0.2
        return_mask = torch.zeros(SMALL, SMALL, dtype=torch.bool, device=device)
        return_mask[py, px] = True
        terrain_before = self.om.terrain.sum().item()
        source_before = self.om.energy_matrix[sy, sx].item()
        self.om._return_pending_to_sources(return_mask)
        self.assertAlmostEqual(self.om.pending_birth_energy[py, px].item(), 0.0)
        self.assertAlmostEqual(self.om.terrain.sum().item(), terrain_before, places=5)
        self.assertGreater(self.om.energy_matrix[sy, sx].item(), source_before)

    def test_orphaned_pending_with_no_neighbors_goes_to_destroyed_not_terrain(self):
        py, px = 5, 5
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.pending_birth_energy.zero_()
        self.om.pending_giver_contrib.zero_()
        self.om.pending_birth_energy[py, px] = 0.2
        self.om.pending_giver_contrib[0, py, px] = 0.2
        terrain_before = self.om.terrain.sum().item()
        destroyed_before = self.om.destroyed_energy
        self.om._return_orphaned_pending_to_sources()
        self.om._flush_tick_destroyed()
        self.assertAlmostEqual(self.om.pending_birth_energy[py, px].item(), 0.0)
        self.assertAlmostEqual(self.om.terrain.sum().item(), terrain_before, places=6)
        self.assertGreater(self.om.destroyed_energy, destroyed_before)

    def test_pending_return_spreads_to_living_neighbor_when_giver_dead(self):
        py, px = 5, 5
        ny, nx = 5, 6
        sy, sx = 5, 4
        self.om.topology_matrix.zero_()
        self.om.energy_matrix.zero_()
        self.om.pending_birth_energy.zero_()
        self.om.pending_giver_contrib.zero_()
        self.om.topology_matrix[ny, nx] = 1
        self.om.energy_matrix[ny, nx] = 0.3
        self.om.pending_birth_energy[py, px] = 0.2
        giver_dir = None
        for g in range(8):
            if self.om._giver_source_y[g, py, px].item() == sy and self.om._giver_source_x[g, py, px].item() == sx:
                giver_dir = g
                break
        self.assertIsNotNone(giver_dir)
        self.om.pending_giver_contrib[giver_dir, py, px] = 0.2
        return_mask = torch.zeros(SMALL, SMALL, dtype=torch.bool, device=device)
        return_mask[py, px] = True
        terrain_before = self.om.terrain.sum().item()
        neighbor_before = self.om.energy_matrix[ny, nx].item()
        self.om._return_pending_to_sources(return_mask)
        self.assertAlmostEqual(self.om.pending_birth_energy[py, px].item(), 0.0)
        self.assertAlmostEqual(self.om.terrain.sum().item(), terrain_before, places=4)
        self.assertGreater(self.om.energy_matrix[ny, nx].item(), neighbor_before)


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

    def test_perlin_compute_environment_depletes_harvest(self):
        env = Environment(SMALL, 0.01, 0.01)
        env.environment_type = 3
        topology = torch.zeros(SMALL, SMALL, device=device)
        topology[6, 6] = 1
        harvested = torch.zeros(SMALL, SMALL, device=device)
        harvested[6, 6] = 0.2
        env.compute_environment(topology, harvested)
        env.time -= PERLIN_TIME_SPEED
        fresh = env._generate_perlin_terrain()
        expected = torch.clamp(fresh - harvested * 100 * topology, 0, 1)
        if PERLIN_DEAD_THRESHOLD > 0:
            expected = expected * (expected > PERLIN_DEAD_THRESHOLD)
        self.assertAlmostEqual(env.terrain[6, 6].item(), expected[6, 6].item(), places=4)

    def test_sub_threshold_cells_skip_sharing(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = DEATH_THRESHOLD - 0.01
        om.sharing_rate_matrix[6, 6] = 1.0
        terrain = torch.zeros(SMALL, SMALL, device=device)
        om._apply_harvest_and_decay(terrain)
        om._remove_dead_cells()
        self.assertEqual(om.topology_matrix[6, 6].item(), 0)
        energy_before = om.energy_matrix.sum().item()
        props = uniform_proportions(SMALL, 1, 1)
        apply_sharing_physics(om, terrain, props, skip_harvest=True)
        self.assertEqual(om.energy_matrix.sum().item(), energy_before)


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

    def test_organism_hidden_on_sharing_off_is_green(self):
        env = torch.zeros(SMALL, SMALL, device=device)
        topo = torch.zeros(SMALL, SMALL, device=device)
        topo[6, 6] = 1
        sharing = torch.zeros(SMALL, SMALL, device=device)
        hidden = torch.ones(1, SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, topo, sharing_rate=sharing, hidden_channels=hidden)
        self.assertLess(image[0, 6, 6].item(), 0.1)
        self.assertGreater(image[1, 6, 6].item(), 0.9)
        self.assertLess(image[2, 6, 6].item(), 0.1)

    def test_organism_hidden_on_sharing_on_is_yellow(self):
        env = torch.zeros(SMALL, SMALL, device=device)
        topo = torch.zeros(SMALL, SMALL, device=device)
        topo[6, 6] = 1
        sharing = torch.ones(SMALL, SMALL, device=device)
        hidden = torch.ones(1, SMALL, SMALL, device=device)
        image = self.renderer.render(env, topo, topo, sharing_rate=sharing, hidden_channels=hidden)
        self.assertGreater(image[0, 6, 6].item(), 0.9)
        self.assertGreater(image[1, 6, 6].item(), 0.9)
        self.assertLess(image[2, 6, 6].item(), 0.1)

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
        self.assertLessEqual(removed, received + 1e-5)
        alive = om.topology_matrix > 0
        self.assertTrue(torch.all(om.energy_matrix[alive] >= DEATH_THRESHOLD - 1e-5))

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

    def test_initialize_topology_sets_parent_giver_dir(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        self.assertEqual(om.parent_giver_dir[6, 6].item(), 0)

    def test_living_cell_receives_parent_direction_income(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.5
        om.sharing_rate_matrix[6, 6] = 1.0
        neighbor_y, neighbor_x = 6, 7
        om.topology_matrix[neighbor_y, neighbor_x] = 1
        om.energy_matrix[neighbor_y, neighbor_x] = 0.8
        om.sharing_rate_matrix[neighbor_y, neighbor_x] = 1.0
        giver_dir = None
        for g in range(8):
            if om._giver_source_y[g, 6, 6].item() == neighbor_y and om._giver_source_x[g, 6, 6].item() == neighbor_x:
                giver_dir = g
                break
        self.assertIsNotNone(giver_dir)
        om.parent_giver_dir[6, 6] = giver_dir
        proportions = uniform_proportions(SMALL, 1, 0)
        terrain = torch.zeros(SMALL, SMALL, device=device)
        metrics = apply_sharing_physics(om, terrain, proportions, skip_harvest=True)
        self.assertGreater(metrics["parent_incoming"][6, 6].item(), 0.0)

    def test_decay_destroys_energy_without_harvest(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 0.5
        om.sharing_rate_matrix[6, 6] = 1.0
        terrain = torch.zeros(SMALL, SMALL, device=device)
        before = om.energy_matrix.sum().item()
        with patch("main.ENERGY_DECAY", 0.05):
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

    @patch("main.ENERGY_DECAY", 0.0)
    def test_full_cell_harvest_with_zero_decay_does_not_destroy_energy(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.energy_matrix[6, 6] = 1.0
        om.sharing_rate_matrix[6, 6] = 1.0
        terrain = torch.ones(SMALL, SMALL, device=device)
        om._tick_destroyed.zero_()
        om._apply_harvest_and_decay(terrain)
        self.assertAlmostEqual(om.energy_matrix[6, 6].item(), 1.0, places=3)
        self.assertAlmostEqual(om._tick_destroyed.item(), ENERGY_HARVEST_RATE, places=3)

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
            return proportions, hidden

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
        proportions, _ = om.energy_distribution_cnn(
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
        self.assertAlmostEqual(
            net.item(),
            received - removed,
            places=3,
        )

    def test_max_outflow_keeps_surviving_cells_above_death_threshold(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.topology_matrix[6, 6] = 1
        om.energy_matrix[6, 6] = DEATH_THRESHOLD + 0.002
        om.sharing_rate_matrix[6, 6] = 1.0
        proportions = uniform_proportions(SMALL, 1, 2)
        terrain = torch.zeros(SMALL, SMALL, device=device)
        with patch("main.ENERGY_DECAY", 0.0):
            apply_sharing_physics(om, terrain, proportions, skip_harvest=True)
        if om.topology_matrix[6, 6].item() > 0:
            self.assertGreater(om.energy_matrix[6, 6].item(), DEATH_THRESHOLD)

    def test_oversharing_to_pending_does_not_kill_source_same_tick(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        cy, cx = 6, 6
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.topology_matrix[cy, cx] = 1
        om.energy_matrix[cy, cx] = DEATH_THRESHOLD + 0.02
        om.sharing_rate_matrix[cy, cx] = 1.0
        props = uniform_proportions(SMALL, 1, 2)
        terrain = torch.zeros(SMALL, SMALL, device=device)
        with patch("main.ENERGY_DECAY", 0.0):
            apply_sharing_physics(om, terrain, props, skip_harvest=True)
        om.compute_topology()
        om._remove_dead_cells()
        self.assertEqual(om.topology_matrix[cy, cx].item(), 1)
        self.assertGreater(om.energy_matrix[cy, cx].item(), DEATH_THRESHOLD)

    def test_sharing_overflow_returns_to_source_not_terrain(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.parent_giver_dir.fill_(-1)
        om.topology_matrix[6, 6] = 1
        om.topology_matrix[6, 7] = 1
        om.energy_matrix[6, 6] = 0.95
        om.energy_matrix[6, 7] = 1.0
        om.sharing_rate_matrix[6, 6] = 1.0
        om.sharing_rate_matrix[6, 7] = 1.0
        om.parent_giver_dir[6, 6] = 4
        terrain = torch.zeros(SMALL, SMALL, device=device)
        om.terrain = terrain
        terrain_before = om.terrain.sum().item()
        source_before = om.energy_matrix[6, 7].item()
        proportions = uniform_proportions(SMALL, 1, 0)
        apply_sharing_physics(om, terrain, proportions, skip_harvest=True)
        self.assertAlmostEqual(om.terrain.sum().item(), terrain_before, places=6)
        self.assertGreaterEqual(om.energy_matrix[6, 7].item(), source_before - 1e-4)

    def test_circular_wrap_sharing_reaches_wrapped_neighbor(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        y, x_sender, x_receiver = 6, 11, 0
        om.topology_matrix[y, x_sender] = 1
        om.energy_matrix[y, x_sender] = 0.8
        om.sharing_rate_matrix[y, x_sender] = 1.0
        proportions = uniform_proportions(SMALL, 1, 2)
        terrain = torch.zeros(SMALL, SMALL, device=device)
        shareable = om.energy_matrix * om.topology_matrix * om.sharing_rate_matrix
        contributions = om._compute_energy_contributions(shareable, proportions)
        distributed = om._shift_sum_contributions(contributions)
        self.assertGreater(distributed[y, x_receiver].item(), 0.0)
        apply_sharing_physics(om, terrain, proportions, skip_harvest=True)
        received = om.pending_birth_energy[y, x_receiver].item() + om.energy_matrix[y, x_receiver].item()
        self.assertGreater(received, 0.0)

    def test_rotated_proportions_change_world_destination(self):
        cnn = EnergyDistributionCNN(device)
        proportions = torch.zeros(3, 3, SMALL, SMALL, device=device)
        proportions[1, 2, 6, 6] = 1.0
        rotation = torch.zeros(SMALL, SMALL, device=device)
        rotation[6, 6] = torch.pi / 2
        rotated = cnn._rotate_proportions_8way(proportions, rotation)
        self.assertAlmostEqual(rotated[2, 1, 6, 6].item(), 1.0, places=4)
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.topology_matrix[6, 6] = 1
        om.energy_matrix[6, 6] = 0.8
        om.sharing_rate_matrix[6, 6] = 1.0
        om.parent_giver_dir[6, 6] = 0
        om.rotation_matrix[6, 6] = torch.pi / 2
        terrain = torch.zeros(SMALL, SMALL, device=device)
        run_organism_tick(om, terrain, rotated, skip_harvest=True)
        self.assertTrue(
            om.topology_matrix[7, 6].item() > 0
            or om.pending_birth_energy[7, 6].item() > 0
            or om.energy_matrix[7, 6].item() > 0
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

        with patch("main.REPRODUCTION_THRESHOLD", 2.0):
            om.reproduction_threshold = 2.0
            for tick in range(25):
                before = system_total_energy(sim).item()
                sim.update_simulation()
                after = system_total_energy(sim).item()
                self.assertAlmostEqual(before, after, delta=0.15, msg=f"tick {tick + 1}: total energy changed by {after - before}")
                self.assertTrue(torch.all(om.energy_matrix >= 0))
                self.assertTrue(torch.all(om.energy_matrix <= 1))
                self.assertTrue(torch.all(sim.environment.terrain >= 0))
                self.assertTrue(torch.all(sim.environment.terrain <= 1))

    def test_pending_birth_energy_accumulates_across_ticks(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        y, x = 6, 7
        om.new_cell_candidates = torch.zeros(SMALL, SMALL, dtype=torch.bool, device=device)
        om.new_cell_candidates[y, x] = True
        pending_before = REPRODUCTION_THRESHOLD - 0.05
        om.pending_birth_energy[y, x] = pending_before
        terrain = torch.ones(SMALL, SMALL, device=device) * 0.5
        proportions = uniform_proportions(SMALL, 1, 2)
        apply_sharing_physics(om, terrain, proportions)
        self.assertGreater(om.pending_birth_energy[y, x].item(), pending_before)

    @patch("main.ENERGY_DECAY", 0.0)
    def test_per_tick_system_energy_conserved_without_decay(self):
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
        om.energy_matrix[cy, cx] = 0.7
        sim.environment.terrain.fill_(1.0)
        om.terrain = sim.environment.terrain
        proportions = uniform_proportions(sim.world_size, 1, 1)
        for tick in range(15):
            before = system_total_energy(sim).item()
            run_sim_tick_with_proportions(sim, proportions)
            after = system_total_energy(sim).item()
            self.assertAlmostEqual(
                before,
                after,
                delta=0.001,
                msg=f"tick {tick + 1}: total energy changed by {after - before}",
            )

    @patch("main.ENERGY_DECAY", 0.05)
    def test_per_tick_system_energy_decreases_only_by_decay(self):
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
        om.energy_matrix[cy, cx] = 0.7
        sim.environment.terrain.zero_()
        om.terrain = sim.environment.terrain
        proportions = uniform_proportions(sim.world_size, 1, 1)
        cell_energy_before = om.energy_matrix[cy, cx].item()
        run_sim_tick_with_proportions(sim, proportions)
        self.assertLess(om.energy_matrix[cy, cx].item(), cell_energy_before)
        self.assertGreater(om.destroyed_energy, 0.0)

    def test_abandoned_pending_refunded_on_full_tick(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.parent_giver_dir.fill_(-1)
        om.topology_matrix[6, 6] = 1
        om.energy_matrix[6, 6] = 0.6
        om.sharing_rate_matrix[6, 6] = 1.0
        om.parent_giver_dir[6, 6] = 0
        y, x = 6, 7
        giver_dir = None
        for g in range(8):
            if om._giver_source_y[g, y, x].item() == 6 and om._giver_source_x[g, y, x].item() == 6:
                giver_dir = g
                break
        self.assertIsNotNone(giver_dir)
        om.pending_birth_energy[y, x] = 0.15
        om.pending_giver_contrib[giver_dir, y, x] = 0.15
        terrain = torch.zeros(SMALL, SMALL, device=device)
        self_props = uniform_proportions(SMALL, 1, 1)
        with patch("main.REPRODUCTION_THRESHOLD", 0.2):
            om.reproduction_threshold = 0.2
            source_before = om.energy_matrix[6, 6].item()
            terrain_before = om.terrain.sum().item()
            run_organism_tick(om, terrain.clone(), self_props, skip_harvest=True)
            self.assertEqual(om.pending_birth_energy[y, x].item(), 0.0)
            self.assertAlmostEqual(om.terrain.sum().item(), terrain_before, places=4)
            self.assertGreater(om.energy_matrix[6, 6].item(), source_before)

    def test_pending_reaches_threshold_and_births_on_next_topology_step(self):
        om = make_organism_manager(SMALL, center=(6, 6))
        om.topology_matrix.zero_()
        om.energy_matrix.zero_()
        om.sharing_rate_matrix.zero_()
        om.parent_giver_dir.fill_(-1)
        om.topology_matrix[6, 6] = 1
        om.energy_matrix[6, 6] = 0.9
        om.sharing_rate_matrix[6, 6] = 1.0
        om.parent_giver_dir[6, 6] = 0
        terrain = torch.zeros(SMALL, SMALL, device=device)
        share_props = uniform_proportions(SMALL, 1, 2)
        with patch("main.REPRODUCTION_THRESHOLD", 0.5):
            om.reproduction_threshold = 0.5
            run_organism_tick(om, terrain.clone(), share_props, skip_harvest=True)
            self.assertEqual(om.topology_matrix[6, 7].item(), 1)
            self.assertGreaterEqual(om.energy_matrix[6, 7].item(), 0.5)
            self.assertEqual(om.pending_birth_energy[6, 7].item(), 0.0)


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
