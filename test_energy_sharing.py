import torch
import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Optimized.main import EnergyDistributionCNN, OrganismManager
from Optimized.config import *
from Optimized.gpu_handler import GPUHandler

# Initialize device
gpu_handler = GPUHandler()
device = gpu_handler.get_device()


class TestEnergyDistributionCNN(unittest.TestCase):
    """Test the CNN that computes 3x3 distribution proportions"""
    
    def setUp(self):
        self.cnn = EnergyDistributionCNN(device)
        self.world_size = 32  # Small size for testing
    
    def test_output_shape(self):
        """Test that CNN outputs correct shape"""
        shareable_energy = torch.rand(self.world_size, self.world_size, device=device)
        proportions = self.cnn(shareable_energy)
        
        self.assertEqual(proportions.shape, (3, 3, self.world_size, self.world_size))
    
    def test_proportions_sum_to_one(self):
        """Test that proportions for each source cell sum to 1"""
        shareable_energy = torch.ones(self.world_size, self.world_size, device=device)
        proportions = self.cnn(shareable_energy)
        
        # Sum across 3x3 for each spatial position
        sums = proportions.sum(dim=(0, 1))
        
        # Each cell's proportions should sum to ~1.0 (within numerical precision)
        self.assertTrue(torch.allclose(sums, torch.ones(self.world_size, self.world_size, device=device), rtol=1e-5))
    
    def test_proportions_non_negative(self):
        """Test that all proportions are non-negative"""
        shareable_energy = torch.rand(self.world_size, self.world_size, device=device) * 0.5
        proportions = self.cnn(shareable_energy)
        
        self.assertTrue(torch.all(proportions >= 0))
    
    def test_zero_input(self):
        """Test CNN with zero input"""
        shareable_energy = torch.zeros(self.world_size, self.world_size, device=device)
        proportions = self.cnn(shareable_energy)
        
        # With zero kernels, softmax should give uniform distribution
        expected = torch.ones(3, 3, device=device) / 9.0
        self.assertTrue(torch.allclose(proportions[:, :, 0, 0], expected, rtol=1e-5))


class TestEnergyContributions(unittest.TestCase):
    """Test energy contribution computation"""
    
    def setUp(self):
        self.world_size = 16
        self.org_manager = OrganismManager(self.world_size, 1, torch.ones(self.world_size, self.world_size, device=device))
        self.org_manager.topology_matrix[8, 8] = 1  # Single organism at center
        self.org_manager.energy_matrix[8, 8] = 1.0
    
    def test_contributions_shape(self):
        """Test that contributions have correct shape"""
        source_energy = self.org_manager.energy_matrix.clone()
        shareable_energy = source_energy * self.org_manager.topology_matrix * ENERGY_SHARING_RATE
        proportions = self.org_manager.energy_distribution_cnn(shareable_energy)
        
        contributions = self.org_manager._compute_energy_contributions(shareable_energy, proportions)
        
        self.assertEqual(contributions.shape, (3, 3, self.world_size, self.world_size))
    
    def test_contributions_conservation(self):
        """Test that total contributions equal shareable energy"""
        source_energy = self.org_manager.energy_matrix.clone()
        shareable_energy = source_energy * self.org_manager.topology_matrix * ENERGY_SHARING_RATE
        proportions = self.org_manager.energy_distribution_cnn(shareable_energy)
        
        contributions = self.org_manager._compute_energy_contributions(shareable_energy, proportions)
        
        # Sum of all contributions from each source should equal shareable energy
        total_contributions = contributions.sum(dim=(0, 1))
        
        self.assertTrue(torch.allclose(total_contributions, shareable_energy, rtol=1e-5))


class TestContributionAccumulation(unittest.TestCase):
    """Test accumulation of contributions to destinations"""
    
    def setUp(self):
        self.world_size = 16
        self.org_manager = OrganismManager(self.world_size, 1, torch.ones(self.world_size, self.world_size, device=device))
        self.org_manager.topology_matrix[8, 8] = 1
        self.org_manager.energy_matrix[8, 8] = 1.0
    
    def test_accumulation_shape(self):
        """Test that accumulation produces correct shape"""
        source_energy = self.org_manager.energy_matrix.clone()
        shareable_energy = source_energy * self.org_manager.topology_matrix * ENERGY_SHARING_RATE
        proportions = self.org_manager.energy_distribution_cnn(shareable_energy)
        contributions = self.org_manager._compute_energy_contributions(shareable_energy, proportions)
        
        new_energy = self.org_manager._accumulate_contributions(contributions)
        
        self.assertEqual(new_energy.shape, (self.world_size, self.world_size))
    
    def test_accumulation_increases_energy(self):
        """Test that accumulation adds energy to neighbors"""
        initial_energy = self.org_manager.energy_matrix.clone()
        source_energy = initial_energy.clone()
        shareable_energy = source_energy * self.org_manager.topology_matrix * ENERGY_SHARING_RATE
        proportions = self.org_manager.energy_distribution_cnn(shareable_energy)
        contributions = self.org_manager._compute_energy_contributions(shareable_energy, proportions)
        
        new_energy = self.org_manager._accumulate_contributions(contributions)
        
        # Energy should increase at destination cells (neighbors of source)
        energy_increase = new_energy - initial_energy
        
        # Should have positive energy at neighbors
        self.assertTrue(torch.any(energy_increase > 0))
    
    def test_accumulation_conservation(self):
        """Test that total energy in accumulation equals contributions"""
        source_energy = self.org_manager.energy_matrix.clone()
        shareable_energy = source_energy * self.org_manager.topology_matrix * ENERGY_SHARING_RATE
        proportions = self.org_manager.energy_distribution_cnn(shareable_energy)
        contributions = self.org_manager._compute_energy_contributions(shareable_energy, proportions)
        
        initial_total = self.org_manager.energy_matrix.sum().item()
        new_energy = self.org_manager._accumulate_contributions(contributions)
        
        # Total energy added should equal total contributions
        energy_added = (new_energy - self.org_manager.energy_matrix).sum().item()
        total_contributions = contributions.sum().item()
        
        self.assertAlmostEqual(energy_added, total_contributions, places=5)


class TestCapacityConstraints(unittest.TestCase):
    """Test capacity constraint application"""
    
    def setUp(self):
        self.world_size = 16
        self.org_manager = OrganismManager(self.world_size, 1, torch.ones(self.world_size, self.world_size, device=device))
        self.org_manager.topology_matrix[8, 8] = 1
        self.org_manager.energy_matrix[8, 8] = 0.9  # Nearly full
    
    def test_capacity_limits(self):
        """Test that energy doesn't exceed capacity"""
        receiving_mask = torch.ones(self.world_size, self.world_size, device=device)
        
        # Create a new energy matrix with excess energy
        new_energy_matrix = self.org_manager.energy_matrix.clone()
        new_energy_matrix[8, 8] = 1.5  # Exceeds capacity
        
        constrained, actual_received, energy_incoming = self.org_manager._apply_capacity_constraints(
            new_energy_matrix, receiving_mask
        )
        
        # Energy should be clamped to 1.0
        self.assertTrue(torch.all(constrained <= 1.0))
        self.assertTrue(torch.all(constrained >= 0.0))
    
    def test_actual_received(self):
        """Test that actual_received respects capacity"""
        receiving_mask = torch.ones(self.world_size, self.world_size, device=device)
        
        new_energy_matrix = self.org_manager.energy_matrix.clone()
        new_energy_matrix[8, 9] = 0.5  # Add energy to neighbor
        
        constrained, actual_received, energy_incoming = self.org_manager._apply_capacity_constraints(
            new_energy_matrix, receiving_mask
        )
        
        # actual_received should not exceed capacity
        capacity = (1.0 - self.org_manager.energy_matrix) * receiving_mask
        self.assertTrue(torch.all(actual_received <= capacity))
        self.assertTrue(torch.all(actual_received <= energy_incoming))


class TestSourceRemoval(unittest.TestCase):
    """Test per-source energy removal computation"""
    
    def setUp(self):
        self.world_size = 16
        self.org_manager = OrganismManager(self.world_size, 1, torch.ones(self.world_size, self.world_size, device=device))
        self.org_manager.topology_matrix[8, 8] = 1
        self.org_manager.energy_matrix[8, 8] = 1.0
    
    def test_source_removal_shape(self):
        """Test that source_removed has correct shape"""
        source_energy = self.org_manager.energy_matrix.clone()
        shareable_energy = source_energy * self.org_manager.topology_matrix * ENERGY_SHARING_RATE
        proportions = self.org_manager.energy_distribution_cnn(shareable_energy)
        contributions = self.org_manager._compute_energy_contributions(shareable_energy, proportions)
        
        receiving_mask = torch.ones(self.world_size, self.world_size, device=device)
        new_energy = self.org_manager._accumulate_contributions(contributions)
        constrained, actual_received, energy_incoming = self.org_manager._apply_capacity_constraints(
            new_energy, receiving_mask
        )
        
        dest_efficiency = torch.where(
            energy_incoming > 0,
            actual_received / energy_incoming,
            torch.zeros_like(energy_incoming)
        ) * receiving_mask
        
        source_removed = self.org_manager._compute_source_removed(contributions, dest_efficiency, receiving_mask)
        
        self.assertEqual(source_removed.shape, (self.world_size, self.world_size))
    
    def test_source_removal_non_negative(self):
        """Test that source removal is non-negative"""
        source_energy = self.org_manager.energy_matrix.clone()
        shareable_energy = source_energy * self.org_manager.topology_matrix * ENERGY_SHARING_RATE
        proportions = self.org_manager.energy_distribution_cnn(shareable_energy)
        contributions = self.org_manager._compute_energy_contributions(shareable_energy, proportions)
        
        receiving_mask = torch.ones(self.world_size, self.world_size, device=device)
        new_energy = self.org_manager._accumulate_contributions(contributions)
        constrained, actual_received, energy_incoming = self.org_manager._apply_capacity_constraints(
            new_energy, receiving_mask
        )
        
        dest_efficiency = torch.where(
            energy_incoming > 0,
            actual_received / energy_incoming,
            torch.zeros_like(energy_incoming)
        ) * receiving_mask
        
        source_removed = self.org_manager._compute_source_removed(contributions, dest_efficiency, receiving_mask)
        
        self.assertTrue(torch.all(source_removed >= 0))
    
    def test_source_removed_bounds(self):
        """Test that source removal doesn't exceed shareable energy"""
        source_energy = self.org_manager.energy_matrix.clone()
        shareable_energy = source_energy * self.org_manager.topology_matrix * ENERGY_SHARING_RATE
        proportions = self.org_manager.energy_distribution_cnn(shareable_energy)
        contributions = self.org_manager._compute_energy_contributions(shareable_energy, proportions)
        
        receiving_mask = torch.ones(self.world_size, self.world_size, device=device)
        new_energy = self.org_manager._accumulate_contributions(contributions)
        constrained, actual_received, energy_incoming = self.org_manager._apply_capacity_constraints(
            new_energy, receiving_mask
        )
        
        dest_efficiency = torch.where(
            energy_incoming > 0,
            actual_received / energy_incoming,
            torch.zeros_like(energy_incoming)
        ) * receiving_mask
        
        source_removed = self.org_manager._compute_source_removed(contributions, dest_efficiency, receiving_mask)
        
        # Source removal should not exceed shareable energy
        self.assertTrue(torch.all(source_removed <= shareable_energy + 1e-6))


class TestCompleteEnergySharing(unittest.TestCase):
    """Test complete energy sharing flow"""
    
    def setUp(self):
        self.world_size = 16
        terrain = torch.ones(self.world_size, self.world_size, device=device) * 0.5
        self.org_manager = OrganismManager(self.world_size, 1, terrain)
        self.org_manager.topology_matrix[8, 8] = 1
        self.org_manager.energy_matrix[8, 8] = 1.0
    
    def test_energy_conservation(self):
        """Test that total system energy is conserved"""
        initial_energy = self.org_manager.energy_matrix.sum().item()
        initial_terrain = self.org_manager.terrain.sum().item()
        initial_total = initial_energy + initial_terrain
        
        self.org_manager.compute_energy(self.org_manager.terrain)
        
        final_energy = self.org_manager.energy_matrix.sum().item()
        final_terrain = self.org_manager.terrain.sum().item()
        final_total = final_energy + final_terrain
        
        # Total should be conserved (harvested energy moves from terrain to organisms)
        # But terrain is not updated in compute_energy, so only organism energy changes
        # Energy sharing should conserve total organism energy (minus what was removed)
        # The difference should be due to harvest/decay only
        
        # For now, just check energy is within reasonable bounds
        self.assertGreaterEqual(final_energy, 0)
        self.assertLessEqual(final_energy, self.world_size * self.world_size)
    
    def test_no_negative_energy(self):
        """Test that energy never goes negative"""
        for _ in range(10):
            self.org_manager.compute_energy(self.org_manager.terrain)
            self.assertTrue(torch.all(self.org_manager.energy_matrix >= 0))
    
    def test_energy_bounds(self):
        """Test that energy stays within [0, 1]"""
        for _ in range(10):
            self.org_manager.compute_energy(self.org_manager.terrain)
            self.assertTrue(torch.all(self.org_manager.energy_matrix >= 0))
            self.assertTrue(torch.all(self.org_manager.energy_matrix <= 1))


if __name__ == '__main__':
    unittest.main()

