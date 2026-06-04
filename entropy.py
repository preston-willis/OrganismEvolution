from config import THERMO_ENV_TEMPERATURE
from physics import mixing_entropy


def system_entropy_components(energy_matrix, terrain, topology, pending_energy, pending_mask):
    organism_entropy = (mixing_entropy(energy_matrix) * topology).sum()
    terrain_entropy = mixing_entropy(terrain).sum()
    pending_entropy = (mixing_entropy(pending_energy) * pending_mask).sum()
    return organism_entropy, terrain_entropy, pending_entropy


def configurational_entropy(organism_manager, terrain):
    organism_entropy, terrain_entropy, pending_entropy = system_entropy_components(
        organism_manager.energy_matrix,
        terrain,
        organism_manager.topology_matrix,
        organism_manager.pending_birth_energy,
        organism_manager.new_cell_candidates.float(),
    )
    return organism_entropy + terrain_entropy + pending_entropy


def system_entropy_total(organism_manager, terrain):
    organism_entropy, terrain_entropy, pending_entropy = system_entropy_components(
        organism_manager.energy_matrix,
        terrain,
        organism_manager.topology_matrix,
        organism_manager.pending_birth_energy,
        organism_manager.new_cell_candidates.float(),
    )
    heat_bath_entropy = organism_manager.destroyed_energy / THERMO_ENV_TEMPERATURE
    return organism_entropy + terrain_entropy + pending_entropy + heat_bath_entropy
