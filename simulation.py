import gc

import torch

from config import (
    DEBUG_PRINT_INTERVAL,
    GPU_CACHE_CLEAR_INTERVAL,
    ORGANISM_COUNT,
    WORLD_SIZE,
)
from entropy import system_entropy_components, system_entropy_total
from environment import Environment
from gpu_handler import GPUHandler
from logger import Logger
from organism import OrganismManager
from runtime import get_device, set_device
from seeding import fixed_seed_positions

_gpu_handler = GPUHandler()
set_device(_gpu_handler.get_device())


class Simulation:
    def __init__(self, enable_debug: bool = True, seed_positions=None):
        self.world_size = WORLD_SIZE
        if seed_positions is None:
            seed_positions = fixed_seed_positions(self.world_size, ORGANISM_COUNT)
        self.seed_positions = seed_positions
        self.environment = Environment(self.world_size, self.seed_positions)
        self.organism_manager = OrganismManager(
            self.world_size,
            ORGANISM_COUNT,
            self.environment.terrain,
            self.seed_positions,
        )
        self.logger = Logger()
        self.tick = 0
        self.enable_debug = enable_debug
        self.conductance_scale = 1.0

    def update_simulation(self):
        terrain_debit = self.organism_manager.compute_energy(
            self.environment.terrain,
            self.conductance_scale,
        )
        self.organism_manager.compute_topology()
        self.organism_manager.reseed_organism_positions_if_extinct()
        self.environment.compute_environment(
            self.organism_manager.topology_matrix,
            terrain_debit,
            self.organism_manager,
        )
        self.logger.update_fps()

        log_debug_metrics = self.enable_debug and self.tick % DEBUG_PRINT_INTERVAL == 0
        if log_debug_metrics:
            debug_info = self.logger.get_debug_info()
            om = self.organism_manager
            total_energy = torch.sum(om.energy_matrix).item()
            total_terrain = torch.sum(self.environment.terrain).item()
            total_pending = torch.sum(om.pending_birth_energy).item()
            system_energy = total_energy + total_terrain + total_pending + om.destroyed_energy
            self.logger.log_tick(self.tick, 0, debug_info, None, None, total_energy, total_terrain, system_energy)

        if self.enable_debug and self.tick > 0 and self.tick % GPU_CACHE_CLEAR_INTERVAL == 0:
            _gpu_handler.clear_cache()
            if self.environment.environment_type == 3:
                gc.collect()
                device = get_device()
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

        self.tick += 1
        om = self.organism_manager
        sim_data = {
            "terrain": self.environment.terrain,
            "topology": om.topology_matrix,
            "energy": om.energy_matrix,
            "new_cell_candidates": om.new_cell_candidates,
            "hidden_channels": om.spectrum,
            "pending_birth_energy": om.pending_birth_energy,
            "destroyed_energy": om.destroyed_energy,
            "destroyed_entropy": om.destroyed_entropy,
            "entropy_produced": om.destroyed_entropy,
        }
        if log_debug_metrics:
            organism_s, terrain_s, pending_s = system_entropy_components(
                om.energy_matrix,
                self.environment.terrain,
                om.topology_matrix,
                om.pending_birth_energy,
                om.new_cell_candidates.float(),
            )
            sim_data["organism_entropy"] = organism_s
            sim_data["terrain_entropy"] = terrain_s
            sim_data["pending_entropy"] = pending_s
            sim_data["system_entropy"] = system_entropy_total(om, self.environment.terrain)
        return sim_data

    def reset_for_replay(self):
        self.organism_manager = OrganismManager(
            self.world_size,
            ORGANISM_COUNT,
            self.environment.terrain,
            self.seed_positions,
        )
        self.tick = 0
