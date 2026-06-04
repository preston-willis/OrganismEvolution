import torch

from config import (
    CNN_HIDDEN_CHANNELS,
    DEATH_THRESHOLD,
    REPRODUCTION_THRESHOLD,
    SEED_ORGANISM_ENERGY,
    THERMO_ENV_TEMPERATURE,
)
from cnn import EnergyDistributionCNN, neighbor_direction_weights_from_proportions
import physics
from physics import total_energy
from seeding import fixed_seed_positions
from runtime import get_device


class OrganismManager:
    def __init__(self, world_size, organism_count, terrain, seed_positions=None):
        self.world_size = world_size
        self.terrain = terrain
        device = get_device()

        if seed_positions is None:
            seed_positions = fixed_seed_positions(world_size, organism_count)
        self.positions = torch.tensor(seed_positions, dtype=torch.long, device=device)
        self.topology_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.energy_bands = torch.zeros(
            (CNN_HIDDEN_CHANNELS, world_size, world_size), dtype=torch.float32, device=device
        )
        self.energy_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.spectrum = torch.zeros(
            (CNN_HIDDEN_CHANNELS, world_size, world_size), dtype=torch.float32, device=device
        )
        self.rotation_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.new_cell_candidates = torch.zeros((world_size, world_size), dtype=torch.bool, device=device)
        self.pending_birth_energy = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.pending_bands = torch.zeros(
            (CNN_HIDDEN_CHANNELS, world_size, world_size), dtype=torch.float32, device=device
        )
        self.destroyed_energy = 0.0
        self.destroyed_entropy = 0.0
        self.last_tick_destroyed_energy = 0.0
        self._tick_destroyed = torch.zeros((), device=device, dtype=torch.float32)
        self._tick_entropy = torch.zeros((), device=device, dtype=torch.float32)
        self._has_new_cell_candidates = False
        self.reproduction_threshold = REPRODUCTION_THRESHOLD
        self.energy_distribution_cnn = EnergyDistributionCNN(device)
        self._init_neighbor_kernel()

        self._initialize_topology()

    @property
    def hidden_channels(self):
        return self.spectrum

    @hidden_channels.setter
    def hidden_channels(self, value):
        self.spectrum = value

    def _init_neighbor_kernel(self):
        device = get_device()
        neighbor8 = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
        for ci in range(3):
            for cj in range(3):
                if ci == 1 and cj == 1:
                    continue
                neighbor8[0, 0, ci, cj] = 1.0 / 8.0
        self._neighbor8_kernel = neighbor8

    def _sync_total_energy(self):
        self.energy_matrix = torch.clamp(total_energy(self.energy_bands), 0.0, 1.0)

    def _add_destroyed_energy(self, energy_delta):
        if isinstance(energy_delta, torch.Tensor):
            energy_delta = energy_delta.sum()
        delta = float(energy_delta)
        if delta <= 0.0:
            return
        self._tick_destroyed += delta
        self._tick_entropy += delta / THERMO_ENV_TEMPERATURE

    def _flush_tick_destroyed(self):
        self.last_tick_destroyed_energy = self._tick_destroyed.item()
        self.destroyed_energy += self.last_tick_destroyed_energy
        self.destroyed_entropy += self._tick_entropy.item()
        self._tick_destroyed.zero_()
        self._tick_entropy.zero_()

    def _place_seed_cell(self, y, x):
        self.topology_matrix[y, x] = 1
        uniform = 1.0 / CNN_HIDDEN_CHANNELS
        self.spectrum[:, y, x] = uniform
        self.energy_bands[:, y, x] = SEED_ORGANISM_ENERGY * uniform
        self.rotation_matrix[y, x] = 0
        self._sync_total_energy()

    def _initialize_topology(self):
        if self.positions.numel() > 0:
            y_coords, x_coords = self.positions[:, 1], self.positions[:, 0]
            for idx in range(y_coords.shape[0]):
                self._place_seed_cell(y_coords[idx].item(), x_coords[idx].item())

    def reseed_organism_positions_if_extinct(self):
        if self.topology_matrix.sum().item() != 0:
            return
        self.energy_bands.zero_()
        self.spectrum.zero_()
        self.rotation_matrix.zero_()
        self.pending_birth_energy.zero_()
        self.pending_bands.zero_()
        self.new_cell_candidates.zero_()
        self._has_new_cell_candidates = False
        self._initialize_topology()

    def compute_topology(self):
        if not self._has_new_cell_candidates:
            if not self.new_cell_candidates.any().item():
                return
        self._has_new_cell_candidates = False
        energy_mask = self.new_cell_candidates & (
            self.pending_birth_energy >= self.reproduction_threshold
        )
        uniform = 1.0 / CNN_HIDDEN_CHANNELS
        self.topology_matrix[energy_mask] = 1
        born_total = self.pending_birth_energy[energy_mask]
        self.energy_bands[:, energy_mask] = born_total * uniform
        self.spectrum[:, energy_mask] = uniform
        self.rotation_matrix[energy_mask] = 0
        self.pending_bands[:, energy_mask] = 0
        self.pending_birth_energy[energy_mask] = 0
        self._sync_total_energy()

    def _apply_death(self):
        self._sync_total_energy()
        low_energy_mask = self.energy_matrix < DEATH_THRESHOLD
        self._add_destroyed_energy(self.energy_matrix * low_energy_mask.float())
        self.energy_bands[:, low_energy_mask] = 0
        self.topology_matrix[low_energy_mask] = 0
        self.spectrum[:, low_energy_mask] = 0
        self.rotation_matrix[low_energy_mask] = 0
        self.energy_matrix[low_energy_mask] = 0

    def _apply_harvest(self, terrain, conductance_scale=1.0):
        if terrain is not None:
            self.terrain = terrain
        self._apply_death()
        result = physics.apply_harvest_flux_tick(
            self.energy_bands,
            self.spectrum,
            self.terrain,
            self.topology_matrix,
            self._neighbor8_kernel,
            conductance_scale,
        )
        self.energy_bands = result.energy_bands
        self._add_destroyed_energy(result.sink_bands.sum() + result.clamp_loss)
        self._sync_total_energy()
        return result.terrain_debit

    def compute_energy(self, terrain, conductance_scale=1.0):
        self._tick_destroyed.zero_()
        self._tick_entropy.zero_()
        terrain_debit = self._apply_harvest(terrain, conductance_scale)

        shareable_energy = self.energy_matrix * self.topology_matrix
        proportions, spectrum_out = self.energy_distribution_cnn(
            shareable_energy,
            terrain,
            self.spectrum,
            self.rotation_matrix,
        )
        topology_mask = self.topology_matrix.unsqueeze(0)
        self.spectrum = self.spectrum * (1 - topology_mask) + spectrum_out * topology_mask

        direction_weights = neighbor_direction_weights_from_proportions(
            proportions, self.topology_matrix
        )
        result = physics.apply_repro_flux_tick(
            self.energy_bands,
            self.spectrum,
            self.topology_matrix,
            direction_weights,
            conductance_scale,
        )
        self.energy_bands = result.energy_bands
        self.pending_bands = result.pending_bands
        self.pending_birth_energy = result.pending_bands.sum(dim=0)
        self.new_cell_candidates = (
            (self.pending_birth_energy >= self.reproduction_threshold)
            & (self.topology_matrix == 0)
        )
        self._has_new_cell_candidates = self.new_cell_candidates.any().item()
        self._add_destroyed_energy(result.sink_bands.sum() + result.clamp_loss)
        self._sync_total_energy()
        self._flush_tick_destroyed()
        return terrain_debit
