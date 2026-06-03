import numpy as np
import torch
import time
from noise import pnoise2, pnoise3
from scipy import ndimage
import torchvision
import torchvision.transforms as transforms
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from OpenGL.GL import glRasterPos2f, glCallLists
import ctypes
import uuid
import random
import os
import glob
import multiprocessing
from functools import partial
import gc

# Import our modules
from config import *
from gpu_handler import GPUHandler
from logger import Logger
from input_handler import InputHandler
from Grapher import Grapher
import argparse
import oriented_conv

# Initialize GPU handler
gpu_handler = GPUHandler()
device = gpu_handler.get_device()

# Global harvest rate (can be modified at runtime)
current_harvest_rate = ENERGY_HARVEST_RATE


def mixing_entropy(energy):
    """Binary mixing entropy S(e) on [0, 1] (unitless)."""
    e = torch.clamp(energy, ENTROPY_EPSILON, 1.0 - ENTROPY_EPSILON)
    return -(e * torch.log(e) + (1.0 - e) * torch.log(1.0 - e)).nan_to_num(0.0)


def chemical_potential(energy):
    """μ(e) = ∂S/∂e for binary mixing entropy."""
    e = torch.clamp(energy, ENTROPY_EPSILON, 1.0 - ENTROPY_EPSILON)
    return -torch.log((e + ENTROPY_EPSILON) / (1.0 - e + ENTROPY_EPSILON))


def decay_conductance(sharing_rate, org_avg_neighbor_energy, topology):
    """Onsager coefficient L_i for maintenance/decay (1/temperature-scaled)."""
    return torch.clamp(
        (1.0 / THERMO_ENV_TEMPERATURE)
        * ENERGY_DECAY
        * sharing_rate
        * (1.0 - org_avg_neighbor_energy)
        * topology 
    , 0.001, 0.01)


def thermodynamic_decay_step(energy, sharing_rate, org_avg_neighbor_energy, topology):
    """
    Linear irreversible decay: J = L*μ, σ_dot = J*μ = L*μ², ΔE = T*σ_dot (dt=1).
    Returns (energy_loss, entropy_production_rate) per cell.
    """
    mu = chemical_potential(energy)
    conductance = decay_conductance(sharing_rate, org_avg_neighbor_energy, topology)
    entropy_production_rate = conductance * mu * mu
    energy_loss = torch.clamp(
        torch.minimum(energy, THERMO_ENV_TEMPERATURE * entropy_production_rate),
        0.0,
        1.0,
    )
    return energy_loss, entropy_production_rate


def harvest_conductance(sharing_rate, topology):
    """Onsager coefficient for environment → organism harvest."""
    return (1.0 / THERMO_ENV_TEMPERATURE) * ENERGY_HARVEST_RATE * sharing_rate * topology


def thermodynamic_harvest_step(terrain_energy, organism_energy, sharing_rate, topology):
    """
    Irreversible influx: X = μ_org - μ_terrain, σ_dot = L*X², ΔE = min(terrain, 1-e, T*σ_dot).
    Returns (energy_transfer, entropy_production_rate) per cell.
    """
    mu_terrain = chemical_potential(terrain_energy)
    mu_organism = chemical_potential(organism_energy)
    driving_force = mu_organism - mu_terrain
    conductance = harvest_conductance(sharing_rate, topology)
    entropy_production_rate = conductance * torch.clamp(driving_force, min=0.0) ** 2
    capacity = (1.0 - organism_energy) * topology
    energy_transfer = torch.clamp(
        torch.minimum(
            terrain_energy,
            torch.minimum(capacity, THERMO_ENV_TEMPERATURE * entropy_production_rate),
        ),
        0.0,
        1.0,
    )
    return energy_transfer, entropy_production_rate


def system_entropy_components(energy_matrix, terrain, topology, pending_energy, pending_mask):
    """Configurational entropy in organism, terrain, and pending birth sites."""
    organism_entropy = (mixing_entropy(energy_matrix) * topology).sum()
    terrain_entropy = mixing_entropy(terrain).sum()
    pending_entropy = (mixing_entropy(pending_energy) * pending_mask).sum()
    return organism_entropy, terrain_entropy, pending_entropy


def configurational_entropy(organism_manager, terrain):
    """S_org + S_terrain + S_pending (active pools only, no heat bath)."""
    organism_entropy, terrain_entropy, pending_entropy = system_entropy_components(
        organism_manager.energy_matrix,
        terrain,
        organism_manager.topology_matrix,
        organism_manager.pending_birth_energy,
        organism_manager.new_cell_candidates.float(),
    )
    return organism_entropy + terrain_entropy + pending_entropy


def system_entropy_total(organism_manager, terrain):
    """
    Configurational entropy in active pools plus heat-bath term U_destroyed/T_env.
    Cumulative destroyed_entropy tracks integrated irreversible production (σ̇) separately.
    """
    organism_entropy, terrain_entropy, pending_entropy = system_entropy_components(
        organism_manager.energy_matrix,
        terrain,
        organism_manager.topology_matrix,
        organism_manager.pending_birth_energy,
        organism_manager.new_cell_candidates.float(),
    )
    heat_bath_entropy = organism_manager.destroyed_energy / THERMO_ENV_TEMPERATURE
    return organism_entropy + terrain_entropy + pending_entropy + heat_bath_entropy

# Global best CNN for replay
current_best_cnn = None
replay_mode = False

class BasicCPPN(torch.nn.Module):
    """Basic Compositional Pattern-Producing Network for generating CNN kernel weights"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Input: 3D coordinates (radial_distance, input_channel, output_channel) normalized
        # Hidden layers
        self.fc1 = torch.nn.Linear(3, 16, device=device)
        self.fc2 = torch.nn.Linear(16, 16, device=device)
        self.fc3 = torch.nn.Linear(16, 1, device=device)
        self.to(device)
    
    def forward(self, coords):
        """
        Input: coords of shape (N, 3) where columns are [radial_distance, in_ch, out_ch]
        Output: weights of shape (N, 1)
        """
        x = torch.tanh(self.fc1(coords))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def generate_conv_weights(self, in_channels, out_channels, kernel_size):
        """Generate weights for a convolutional layer"""
        # Create coordinate grid
        coords_list = []
        for out_ch in range(out_channels):
            for in_ch in range(in_channels):
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        # Normalize coordinates to [-1, 1]
                        x_norm = (kx / max(kernel_size - 1, 1)) * 2 - 1 if kernel_size > 1 else 0
                        y_norm = (ky / max(kernel_size - 1, 1)) * 2 - 1 if kernel_size > 1 else 0
                        # Use radial distance from center for symmetry
                        r = (x_norm**2 + y_norm**2) ** 0.5
                        in_ch_norm = (in_ch / max(in_channels - 1, 1)) * 2 - 1 if in_channels > 1 else 0
                        out_ch_norm = (out_ch / max(out_channels - 1, 1)) * 2 - 1 if out_channels > 1 else 0
                        coords_list.append([r, in_ch_norm, out_ch_norm])
        
        coords = torch.tensor(coords_list, dtype=torch.float32, device=self.device)
        weights_flat = self.forward(coords).squeeze(-1)
        
        # Reshape to (out_channels, in_channels, kernel_size, kernel_size)
        weights = weights_flat.view(out_channels, in_channels, kernel_size, kernel_size)
        return weights
    
    def generate_bias(self, out_channels):
        """Generate bias values"""
        coords_list = []
        for out_ch in range(out_channels):
            out_ch_norm = (out_ch / max(out_channels - 1, 1)) * 2 - 1 if out_channels > 1 else 0
            coords_list.append([0.0, 0.0, out_ch_norm])
        
        coords = torch.tensor(coords_list, dtype=torch.float32, device=self.device)
        bias = self.forward(coords).squeeze(-1)
        return bias


CNN_OUTPUT_DIM = 11
_NEIGHBOR_OFFSETS = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _hsv_to_rgb(h, s, v):
    """Hue in [0, 1); returns (r, g, b) in [0, 1]."""
    h = h % 1.0
    c = v * s
    x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
    m = v - c
    sector = int(h * 6.0) % 6
    if sector == 0:
        r, g, b = c, x, 0.0
    elif sector == 1:
        r, g, b = x, c, 0.0
    elif sector == 2:
        r, g, b = 0.0, c, x
    elif sector == 3:
        r, g, b = 0.0, x, c
    elif sector == 4:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return r + m, g + m, b + m


_LINEAGE_BMM_CHUNK = 32


def apply_lineage_mutation_logits(logits, transform_field, apply_mask, lineage_cell_count):
    """Apply per-cell transforms on the grid via batched bmm; write back only where apply_mask."""
    if lineage_cell_count == 0:
        return logits
    c, h, w = logits.shape
    flat = h * w
    transforms = transform_field.permute(2, 3, 0, 1).reshape(flat, c, c)
    vectors = logits.permute(1, 2, 0).reshape(flat, c, 1)
    updated_flat = vectors.new_empty(flat, c, 1)
    for start in range(0, flat, _LINEAGE_BMM_CHUNK):
        end = min(start + _LINEAGE_BMM_CHUNK, flat)
        updated_flat[start:end] = torch.bmm(transforms[start:end], vectors[start:end])
    updated = updated_flat.reshape(h, w, c).permute(2, 0, 1)
    logits[:, apply_mask] = updated[:, apply_mask]
    return logits


class ColonyMutationGraph:
    """
    Lineage adjacency of differential transforms on CNN logits (11 channels).
    Each parent->child edge stores delta (11x11); composed transform T_child = (I+delta) @ T_parent.
    Crossover + mutation only when an alive neighbor has a different genome_id.
    """

    def __init__(self, world_size, device, mut_rate, mut_mag):
        self.world_size = world_size
        self.device = device
        self.mut_rate = mut_rate
        self.mut_mag = mut_mag
        self.cell_id = torch.full((world_size, world_size), -1, dtype=torch.long, device=device)
        self.genome_id_field = torch.full((world_size, world_size), -1, dtype=torch.long, device=device)
        self.next_cell_id = 0
        self.next_genome_id = 0
        self.genomes = {}
        self._genome_tensor = torch.zeros(0, CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=device)
        self._genome_key_to_id = {}
        self.adjacency = []
        self.transform_field = self._identity_field()
        self.non_identity_mask = torch.zeros(
            (world_size, world_size), dtype=torch.bool, device=device
        )
        self.lineage_mutation_cell_count = 0
        self._genome_rgb = torch.zeros(0, 3, device=device)

    def _identity(self):
        return torch.eye(CNN_OUTPUT_DIM, device=self.device)

    def _identity_field(self):
        eye = self._identity()
        return eye.view(CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, 1, 1).expand(
            CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, self.world_size, self.world_size
        ).clone()

    def _genome_key(self, transform):
        return tuple(torch.round(transform.flatten() * 100).to(torch.int16).cpu().tolist())

    def _grow_genome_tensor(self, size):
        if size <= self._genome_tensor.shape[0]:
            return
        grown = torch.zeros(size, CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=self.device)
        if self._genome_tensor.shape[0] > 0:
            grown[: self._genome_tensor.shape[0]] = self._genome_tensor
        self._genome_tensor = grown

    def _grow_genome_rgb(self, size):
        if size <= self._genome_rgb.shape[0]:
            return
        grown = torch.zeros(size, 3, device=self.device)
        if self._genome_rgb.shape[0] > 0:
            grown[: self._genome_rgb.shape[0]] = self._genome_rgb
        self._genome_rgb = grown

    def _sample_genome_rgb(self):
        hue = torch.rand((), device=self.device).item()
        r, g, b = _hsv_to_rgb(hue, 0.85, 0.95)
        return torch.tensor((r, g, b), device=self.device, dtype=torch.float32)

    def _assign_genome_rgb(self, genome_id):
        genome_id = int(genome_id)
        self._grow_genome_rgb(genome_id + 1)
        self._genome_rgb[genome_id] = self._sample_genome_rgb()

    def _store_genome_tensor(self, genome_id, transform):
        genome_id = int(genome_id)
        is_new = genome_id >= self._genome_rgb.shape[0]
        self._grow_genome_tensor(genome_id + 1)
        self._genome_tensor[genome_id] = transform.clone()
        self.genomes[genome_id] = self._genome_tensor[genome_id]
        if is_new:
            self._assign_genome_rgb(genome_id)

    def _register_genome(self, transform):
        key = self._genome_key(transform)
        existing = self._genome_key_to_id.get(key)
        if existing is not None:
            return existing
        genome_id = self.next_genome_id
        self.next_genome_id += 1
        self._genome_key_to_id[key] = genome_id
        self._store_genome_tensor(genome_id, transform)
        return genome_id

    def genome_color_field(self, topology):
        """(3, H, W) RGB from per-genome hue assigned at registration."""
        alive = topology > 0
        gids = self.genome_id_field.clamp(min=0)
        colors = self._genome_rgb[gids].permute(2, 0, 1)
        return colors * alive.unsqueeze(0).float()

    def genome_panel_lines(self, topology):
        """Alive genome count and top 10 by cell count: (text, (r, g, b)) per line."""
        alive = topology > 0
        if not alive.any():
            return 0, [("(no alive cells)", (1.0, 1.0, 1.0))]
        total = int(alive.sum().item())
        gids = self.genome_id_field[alive]
        unique_gids = torch.unique(gids)
        ranked = []
        for genome_id in unique_gids.cpu().tolist():
            genome_id = int(genome_id)
            if genome_id < 0:
                continue
            count = int((self.genome_id_field == genome_id).sum().item())
            ranked.append((genome_id, count))
        ranked.sort(key=lambda row: row[1], reverse=True)
        lines = []
        for genome_id, count in ranked[:10]:
            pct = 100.0 * count / total
            rgb = self._genome_rgb[genome_id]
            color = (rgb[0].item(), rgb[1].item(), rgb[2].item())
            lines.append((f"#{genome_id}  {count} cells  {pct:.1f}%", color))
        return len(ranked), lines

    def sample_edge_delta(self):
        delta = torch.zeros(CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=self.device)
        mask = torch.rand(CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=self.device) < self.mut_rate
        noise = (torch.rand(CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=self.device) - 0.5) * 2 * self.mut_mag
        delta[mask] = noise[mask]
        return delta

    def sample_seed_genome_transform(self):
        """Random initial lineage genome for --lineage seeds (distinct per call)."""
        delta = (torch.rand(CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=self.device) - 0.5) * 2 * self.mut_mag
        return self._identity() + delta

    def _set_cell_genome(self, y, x, transform):
        genome_id = self._register_genome(transform)
        self.genome_id_field[y, x] = genome_id
        return genome_id

    def _assign_genome_id_fast(self, y, x, transform):
        """Assign a new genome id without CPU hashing (hot path)."""
        genome_id = self.next_genome_id
        self.next_genome_id += 1
        self._store_genome_tensor(genome_id, transform)
        self.genome_id_field[y, x] = genome_id
        return genome_id

    def sample_edge_delta_batch(self, count):
        delta = torch.zeros(count, CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=self.device)
        mask = torch.rand(count, CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=self.device) < self.mut_rate
        noise = (torch.rand(count, CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, device=self.device) - 0.5) * 2 * self.mut_mag
        delta[mask] = noise[mask]
        return delta

    def _mark_non_identity_cells(self, y, x, is_non_identity):
        was = self.non_identity_mask[y, x]
        self.non_identity_mask[y, x] = is_non_identity
        self.lineage_mutation_cell_count += int((is_non_identity & ~was).sum().item())

    def compute_heterogeneous_neighborhood_mask(self, topology):
        """Alive cells touching a different genome_id on the 8-neighbor ring."""
        alive = topology > 0
        center_gid = self.genome_id_field
        heterogeneous = torch.zeros_like(alive)
        y_grid = torch.arange(self.world_size, device=self.device).view(self.world_size, 1)
        x_grid = torch.arange(self.world_size, device=self.device).view(1, self.world_size)
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny = (y_grid + dy) % self.world_size
            nx = (x_grid + dx) % self.world_size
            neighbor_alive = topology[ny, nx] > 0
            neighbor_gid = self.genome_id_field[ny, nx]
            heterogeneous = heterogeneous | (
                alive
                & neighbor_alive
                & (neighbor_gid != center_gid)
                & (neighbor_gid >= 0)
            )
        return heterogeneous

    def _birth_recombines_mask(self, child_y, child_x, parent_genome_id, topology):
        """Per-birth: True when some alive neighbor of the child has genome_id != parent."""
        recombines = torch.zeros(child_y.shape[0], dtype=torch.bool, device=self.device)
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny = (child_y + dy) % self.world_size
            nx = (child_x + dx) % self.world_size
            neighbor_alive = topology[ny, nx] > 0
            neighbor_gid = self.genome_id_field[ny, nx]
            recombines = recombines | (
                neighbor_alive & (neighbor_gid != parent_genome_id) & (neighbor_gid >= 0)
            )
        return recombines

    def _pick_crossover_genomes_batch(self, child_y, child_x, parent_genome_id, topology):
        """First valid neighbor genome != parent_genome_id for each birth (-1 if none)."""
        picks = torch.full(child_y.shape, -1, dtype=torch.long, device=self.device)
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny = (child_y + dy) % self.world_size
            nx = (child_x + dx) % self.world_size
            neighbor_alive = topology[ny, nx] > 0
            neighbor_gid = self.genome_id_field[ny, nx]
            valid = neighbor_alive & (neighbor_gid != parent_genome_id) & (neighbor_gid >= 0)
            picks = torch.where((picks < 0) & valid, neighbor_gid, picks)
        return picks

    def register_seed(self, y, x):
        cell_id = self.next_cell_id
        self.next_cell_id += 1
        self.cell_id[y, x] = cell_id
        composed = self.sample_seed_genome_transform()
        self.transform_field[:, :, y, x] = composed
        is_non_identity = (composed - self._identity()).abs().max() > 1e-5
        self._mark_non_identity_cells(
            torch.tensor([y], device=self.device, dtype=torch.long),
            torch.tensor([x], device=self.device, dtype=torch.long),
            torch.tensor([is_non_identity], device=self.device, dtype=torch.bool),
        )
        self._set_cell_genome(y, x, composed)

    def register_birth(self, child_y, child_x, parent_y, parent_x, topology):
        cy = torch.tensor([child_y], device=self.device, dtype=torch.long)
        cx = torch.tensor([child_x], device=self.device, dtype=torch.long)
        py = torch.tensor([parent_y], device=self.device, dtype=torch.long)
        px = torch.tensor([parent_x], device=self.device, dtype=torch.long)
        has_parent = torch.tensor([True], device=self.device, dtype=torch.bool)
        self.register_births(cy, cx, py, px, has_parent, topology)

    def _apply_recombine_births_batch(
        self,
        child_y,
        child_x,
        parent_y,
        parent_x,
        parent_ids,
        crossover_genome_id,
        child_ids,
    ):
        identity = self._identity()
        n = child_y.shape[0]
        has_parent = parent_ids >= 0
        base = torch.where(
            has_parent,
            self.transform_field[:, :, parent_y, parent_x],
            identity.view(CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, 1).expand(-1, -1, n),
        )
        self._grow_genome_tensor(int(crossover_genome_id.max().item()) + 1)
        other = self._genome_tensor[crossover_genome_id.long()].permute(1, 2, 0)
        blended = 0.5 * base + 0.5 * other
        mutate = torch.rand(n, device=self.device) < self.mut_rate
        child_transforms = blended
        if mutate.any():
            identity_n = identity.unsqueeze(0).expand(n, -1, -1)
            deltas = self.sample_edge_delta_batch(n)
            blended_n = blended.permute(2, 0, 1)
            child_transforms = torch.bmm(identity_n + deltas, blended_n).permute(1, 2, 0)
            child_transforms = torch.where(
                mutate.view(1, 1, -1),
                child_transforms,
                blended,
            )
            edge_mutate = mutate & has_parent
            if edge_mutate.any():
                edge_idx = edge_mutate.nonzero(as_tuple=True)[0]
                for idx, parent_id, child_id in zip(
                    edge_idx.tolist(),
                    parent_ids[edge_idx].tolist(),
                    child_ids[edge_idx].tolist(),
                ):
                    self.adjacency.append((parent_id, child_id, deltas[idx]))
        self.transform_field[:, :, child_y, child_x] = child_transforms
        identity_view = identity.view(CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, 1)
        return (child_transforms - identity_view).abs().amax(dim=(0, 1)) > 1e-5

    def register_births(
        self,
        child_y,
        child_x,
        parent_y,
        parent_x,
        has_parent,
        topology,
    ):
        """Batch-register births; crossover/mutation only in heterogeneous neighborhoods."""
        n = child_y.shape[0]
        if n == 0:
            return
        child_ids = torch.arange(
            self.next_cell_id,
            self.next_cell_id + n,
            device=self.device,
            dtype=torch.long,
        )
        self.next_cell_id += n
        self.cell_id[child_y, child_x] = child_ids

        parent_ids = self.cell_id[parent_y, parent_x]
        parent_genome_id = torch.where(
            has_parent,
            self.genome_id_field[parent_y, parent_x],
            torch.full((n,), -1, dtype=torch.long, device=self.device),
        )
        crossover_genome_id = self._pick_crossover_genomes_batch(
            child_y, child_x, parent_genome_id, topology
        )
        heterogeneous = crossover_genome_id >= 0
        recombines = heterogeneous & (
            torch.rand(n, device=self.device) < INTERACTION_RATE
        )

        self.transform_field[:, :, child_y, child_x] = self.transform_field[
            :, :, parent_y, parent_x
        ]
        self.genome_id_field[child_y, child_x] = self.genome_id_field[parent_y, parent_x]
        new_non_identity = self.non_identity_mask[parent_y, parent_x].clone()

        if recombines.any():
            recomb_child_y = child_y[recombines]
            recomb_child_x = child_x[recombines]
            recomb_parent_y = parent_y[recombines]
            recomb_parent_x = parent_x[recombines]
            recomb_parent_ids = parent_ids[recombines]
            recomb_child_ids = child_ids[recombines]
            recomb_crossover = crossover_genome_id[recombines]
            recomb_non_identity = self._apply_recombine_births_batch(
                recomb_child_y,
                recomb_child_x,
                recomb_parent_y,
                recomb_parent_x,
                recomb_parent_ids,
                recomb_crossover,
                recomb_child_ids,
            )
            new_gids = torch.arange(
                self.next_genome_id,
                self.next_genome_id + recomb_child_y.shape[0],
                device=self.device,
                dtype=torch.long,
            )
            self.next_genome_id += recomb_child_y.shape[0]
            self.genome_id_field[recomb_child_y, recomb_child_x] = new_gids
            self._grow_genome_tensor(self.next_genome_id)
            recomb_transforms = self.transform_field[:, :, recomb_child_y, recomb_child_x].permute(2, 0, 1)
            self._genome_tensor[new_gids] = recomb_transforms
            for genome_id in new_gids.tolist():
                self.genomes[genome_id] = self._genome_tensor[genome_id]
                self._assign_genome_rgb(genome_id)
            new_non_identity[recombines] = recomb_non_identity

        was_non_identity = self.non_identity_mask[child_y, child_x]
        self.non_identity_mask[child_y, child_x] = new_non_identity
        self.lineage_mutation_cell_count += int((new_non_identity & ~was_non_identity).sum().item())

    def mutation_apply_mask(self, topology):
        """Alive cells whose lineage transform is not identity."""
        return (topology > 0) & self.non_identity_mask

    def clear_cells_mask(self, mask):
        if not mask.any():
            return
        dead_non_identity = mask & self.non_identity_mask
        self.lineage_mutation_cell_count -= int(dead_non_identity.sum().item())
        if self.lineage_mutation_cell_count < 0:
            self.lineage_mutation_cell_count = 0
        self.cell_id[mask] = -1
        self.genome_id_field[mask] = -1
        self.non_identity_mask[mask] = False
        identity = self._identity().view(CNN_OUTPUT_DIM, CNN_OUTPUT_DIM, 1)
        self.transform_field[:, :, mask] = identity


class EnergyDistributionCNN(torch.nn.Module):
    """CNN that outputs 3x3 distribution proportions for each source cell"""
    # 8 compass directions clockwise from East: E, SE, S, SW, W, NW, N, NE
    _RING_CIJ = [(1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2)]

    @staticmethod
    def _make_bucket_offsets(device):
        """Per-bucket (8) local-ring (8) dy/dx offsets into the world grid."""
        offsets = torch.zeros(8, 8, 2, dtype=torch.long, device=device)
        for k in range(8):
            for l in range(8):
                ci, cj = EnergyDistributionCNN._RING_CIJ[(l + k) % 8]
                offsets[k, l, 0] = ci - 1
                offsets[k, l, 1] = cj - 1
        return offsets

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.register_buffer(
            '_bucket_offsets',
            self._make_bucket_offsets(device),
            persistent=False,
        )
        ring_ci = []
        ring_cj = []
        for ci, cj in self._RING_CIJ:
            ring_ci.append(ci)
            ring_cj.append(cj)
        self.register_buffer(
            '_ring_ci',
            torch.tensor(ring_ci, device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            '_ring_cj',
            torch.tensor(ring_cj, device=device, dtype=torch.long),
            persistent=False,
        )
        # CPPN for generating kernel weights
        self.cppn = BasicCPPN(device)
        
        # Conv layers for processing input channels directly
        # Input: 4 channels (shareable_energy + terrain + sharing_rate + 1 hidden channel)
        # Output: 11 channels (9 for 3x3 distribution matrix + 1 for sharing_rate + 1 for hidden channel)
        # conv1 processes 3x3 patches with stride=1
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=0, device=device)
        self.conv2 = torch.nn.Conv2d(32, 11, kernel_size=1, device=device)
        
        # Generate weights and biases from CPPN
        self.conv1.weight.data = self.cppn.generate_conv_weights(4, 32, 3)
        self.conv1.bias.data = self.cppn.generate_bias(32)
        self._regenerate_conv2_from_cppn()
        # Move model to device
        self.to(device)

    def _regenerate_conv2_from_cppn(self):
        self.conv2.weight.data = self.cppn.generate_conv_weights(32, 11, 1)
        self.conv2.bias.data = self.cppn.generate_bias(11)
        self._zero_hidden_channel_bias()

    def _zero_hidden_channel_bias(self):
        self.conv2.bias.data[10] = 0

    def _rotate_proportions_8way(self, proportions, rotation_matrix):
        """Rotate cell-local 3x3 proportions to world frame using 8-way cell orientation."""
        bucket = (torch.round(rotation_matrix / (torch.pi / 4)) % 8).long()
        ring = torch.stack([proportions[ci, cj] for ci, cj in self._RING_CIJ])
        d_indices = torch.arange(8, device=proportions.device).view(8, 1, 1)
        source_idx = (d_indices - bucket.unsqueeze(0)) % 8
        rotated_ring = torch.gather(ring, 0, source_idx)
        rotated = proportions.clone()
        rotated[self._ring_ci, self._ring_cj] = rotated_ring
        return rotated

    def forward(
        self,
        shareable_energy,
        terrain,
        sharing_rate,
        hidden_channels,
        rotation_matrix,
        mutation_transform=None,
        mutation_apply_mask=None,
        lineage_mutation_cell_count=0,
    ):
        world_size = shareable_energy.shape[0]
        
        # Stack input channels: (4, H, W)
        input_channels = torch.cat([
            shareable_energy.unsqueeze(0),
            terrain.unsqueeze(0),
            sharing_rate.unsqueeze(0),
            hidden_channels
        ], dim=0)  # (4, H, W)
        
        x = oriented_conv.conv1_forward(
            input_channels,
            self.conv1.weight,
            self.conv1.bias,
            rotation_matrix,
            self._bucket_offsets,
        )
        x = self.conv2(x.unsqueeze(0)).squeeze(0)
        if mutation_transform is not None and mutation_apply_mask is not None:
            x = apply_lineage_mutation_logits(
                x,
                mutation_transform,
                mutation_apply_mask,
                lineage_mutation_cell_count,
            )

        # (11, H, W)
        proportions_flat = x[:9]  # (9, H, W)
        sharing_rate_output = x[9:10].squeeze(0)  # (H, W)
        hidden_channels_output = x[10:11]  # (1, H, W)
        
        proportions_flat = torch.nn.functional.softmax(proportions_flat, dim=0)  # (9, H, W)
        
        # Reshape to (3, 3, H, W), then rotate from cell-local frame to world frame
        proportions = proportions_flat.view(3, 3, world_size, world_size)
        proportions = self._rotate_proportions_8way(proportions, rotation_matrix)
        
        # Binary sharing rate: 1 if logit > 0, else 0
        sharing_rate_output = (sharing_rate_output > 0).float()  # (H, W)
        
        # Binary hidden state: 1 if logit > 0, else 0
        hidden_channels_output = (hidden_channels_output > 0).float()  # (1, H, W)
        
        return proportions, sharing_rate_output, hidden_channels_output


class CNNGeneticAlgorithm:
    """Genetic algorithm for training CNN weights with GPU evaluation"""
    def __init__(self, pop_size, mut_rate, mut_mag, device):
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.mut_mag = mut_mag
        self.device = device
        self.fittest_index = 0
        self.run_id = str(uuid.uuid1())[:4]
        
        # Initialize population of CNNs
        self.subjects = [EnergyDistributionCNN(device) for _ in range(pop_size)]
        self.fitness_scores = [0.0] * pop_size
        
    def reset_fitness(self):
        """Reset all fitness scores"""
        self.fitness_scores = [0.0] * self.pop_size
        
    def compute_generation(self):
        """Run one generation of evolution"""
        self.calc_fittest()
        
        # Save best model if fitness > 0
        # if self.fitness_scores[self.fittest_index] > 0:
        #     self.save_model(self.fittest_index)
            
        # Crossover and mutation
        self.crossover(self.subjects[self.fittest_index])
        self.mutate()
        
    def calc_fittest(self):
        """Find the fittest individual"""
        best_fitness = 0
        best_index = 0
        for i, fitness in enumerate(self.fitness_scores):
            if fitness > best_fitness:
                best_fitness = fitness
                best_index = i
        self.fittest_index = best_index
        
    def crossover(self, parent):
        """Copy parent CPPN weights to all subjects and regenerate CNN kernels"""
        for i in range(self.pop_size):
            if i != self.fittest_index:  # Don't overwrite the parent
                # Copy CPPN parameters
                self.subjects[i].cppn.fc1.weight.data = parent.cppn.fc1.weight.data.clone()
                self.subjects[i].cppn.fc1.bias.data = parent.cppn.fc1.bias.data.clone()
                self.subjects[i].cppn.fc2.weight.data = parent.cppn.fc2.weight.data.clone()
                self.subjects[i].cppn.fc2.bias.data = parent.cppn.fc2.bias.data.clone()
                self.subjects[i].cppn.fc3.weight.data = parent.cppn.fc3.weight.data.clone()
                self.subjects[i].cppn.fc3.bias.data = parent.cppn.fc3.bias.data.clone()
                
                # Regenerate CNN kernels from CPPN
                self.subjects[i].conv1.weight.data = self.subjects[i].cppn.generate_conv_weights(4, 32, 3)
                self.subjects[i].conv1.bias.data = self.subjects[i].cppn.generate_bias(32)
                self.subjects[i]._regenerate_conv2_from_cppn()
                
    def mutate(self):
        """Apply mutations to all subjects except the fittest"""
        for i in range(self.pop_size):
            if i == self.fittest_index:  # Don't mutate the fittest
                continue
            
            # Mutate CPPN parameters
            for layer in [self.subjects[i].cppn.fc1, self.subjects[i].cppn.fc2, self.subjects[i].cppn.fc3]:
                mutation_mask = torch.rand_like(layer.weight) < self.mut_rate
                mutations = (torch.rand_like(layer.weight) - 0.5) * 2 * self.mut_mag
                layer.weight.data[mutation_mask] += mutations[mutation_mask]
                
                mutation_mask = torch.rand_like(layer.bias) < self.mut_rate
                mutations = (torch.rand_like(layer.bias) - 0.5) * 2 * self.mut_mag
                layer.bias.data[mutation_mask] += mutations[mutation_mask]
            
            # Regenerate CNN kernels from mutated CPPN
            self.subjects[i].conv1.weight.data = self.subjects[i].cppn.generate_conv_weights(4, 32, 3)
            self.subjects[i].conv1.bias.data = self.subjects[i].cppn.generate_bias(32)
            self.subjects[i]._regenerate_conv2_from_cppn()
                
    def save_model(self, index, generation=None):
        """Save model weights to file"""
        if generation is not None:
            filename = f'data/cnn_{self.run_id}_gen{generation}_{self.fitness_scores[index]:.6f}.pt'
        else:
            filename = f'data/cnn_{self.run_id}_{self.fitness_scores[index]:.6f}.pt'
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        torch.save(self.subjects[index].state_dict(), filename)
        print(f"Saved model: {filename}")
        
    def load_model(self, filename):
        """Load model weights from file"""
        try:
            state_dict = torch.load(filename, map_location=self.device)
            self.subjects[0].load_state_dict(state_dict)
            self.subjects[0]._zero_hidden_channel_bias()
            print(f"Loaded model: {filename}")
        except Exception as e:
            print(f"Couldn't load {filename}: {e}")
    
    def load_latest_model(self):
        """Load the fittest model from the last generation of the last run"""
        # Find all .pt files in data directory
        pattern = 'data/cnn_*_gen*_*.pt'
        files = glob.glob(pattern)
        
        if not files:
            print("No saved models found in data/ directory")
            return False
        
        # Sort by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Get the most recent file
        latest_file = files[0]
        self.load_model(latest_file)
        return True


# Worker process initialization - set device once per process
_worker_device = None

def _init_worker(device_str):
    """Initialize worker process with device (called once per worker process)"""
    global _worker_device
    from config import DEVICE_TYPE
    if device_str.startswith('cuda'):
        _worker_device = torch.device(device_str)
    elif device_str == DEVICE_TYPE:
        _worker_device = torch.device(device_str)
    else:
        _worker_device = torch.device('cpu')
    
    # Set global device for this worker process (used by Simulation and related classes)
    import sys
    current_module = sys.modules[__name__]
    current_module.device = _worker_device

def _release_device_memory(torch_device):
    gc.collect()
    if torch_device.type == 'mps':
        torch.mps.empty_cache()
    elif torch_device.type == 'cuda':
        torch.cuda.empty_cache()

CNN_FITNESS_MODES = ("cell_count", "entropy_production", "persistence", "life_like")

def validate_cnn_fitness_mode(mode):
    if mode not in CNN_FITNESS_MODES:
        raise ValueError(f"Unknown fitness mode {mode!r}; expected one of {CNN_FITNESS_MODES}")

def set_cnn_fitness_mode(mode):
    """Set module-level CNN_FITNESS_MODE (config.py value used only at import)."""
    global CNN_FITNESS_MODE
    validate_cnn_fitness_mode(mode)
    CNN_FITNESS_MODE = mode

def cnn_fitness_mode_label(mode=None):
    mode = CNN_FITNESS_MODE if mode is None else mode
    if mode == "cell_count":
        return "cell count"
    if mode == "entropy_production":
        return "entropy production"
    if mode == "persistence":
        return "persistence"
    if mode == "life_like":
        return "life-like"
    return mode

def _configure_fitness_environment(sim, fitness_mode):
    if fitness_mode == "persistence":
        sim.environment.terrain.fill_(CNN_FITNESS_PERSISTENCE_TERRAIN)
        sim.organism_manager.terrain = sim.environment.terrain

def run_cnn_fitness_rollout(sim, max_time, collect_tick_data=False, fitness_mode=None):
    """
    Run a training rollout and return (fitness, cumulative_cell_count, tick_data).
    fitness_mode overrides CNN_FITNESS_MODE when provided.
    """
    mode = CNN_FITNESS_MODE if fitness_mode is None else fitness_mode
    validate_cnn_fitness_mode(mode)

    _configure_fitness_environment(sim, mode)
    om = sim.organism_manager
    fitness = 0.0
    cumulative_cell_count = 0.0
    tick_data = []
    entropy_produced_start = om.destroyed_entropy
    config_entropy_start = configurational_entropy(om, sim.environment.terrain)
    for t in range(max_time):
        entropy_before = om.destroyed_entropy
        sim.update_simulation()
        cell_count = torch.sum(om.topology_matrix).item()
        cumulative_cell_count += cell_count

        if mode == "cell_count":
            fitness += cell_count
        elif mode == "entropy_production":
            fitness += om.destroyed_entropy - entropy_before
        elif mode == "persistence":
            if cell_count > 0:
                fitness += 1.0

        if collect_tick_data and t % 10 == 0:
            org_energy = torch.sum(sim.organism_manager.energy_matrix).item()
            env_energy = torch.sum(sim.environment.terrain).item()
            total = org_energy + env_energy
            tick_data.append((t, cumulative_cell_count, cell_count, org_energy, env_energy, total))

        if cell_count == 0:
            break

    if mode == "life_like":
        entropy_produced = om.destroyed_entropy - entropy_produced_start
        config_entropy_end = configurational_entropy(om, sim.environment.terrain)
        delta_config_entropy = config_entropy_end - config_entropy_start
        fitness = (
            entropy_produced
            - CNN_FITNESS_LIFE_LIKE_ORDER_WEIGHT * delta_config_entropy.item()
        )
        if fitness < 0.0:
            fitness = 0.0
        if CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR > 0.0:
            if entropy_produced < CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR:
                fitness = 0.0

    return fitness, cumulative_cell_count, tick_data

def _evaluate_cnn_worker(args):
    """Worker function for multiprocessing CNN evaluation"""
    cnn_state_dict, world_size, max_time, bot_index, collect_tick_data, fitness_mode = args
    
    # Use the device that was set during worker initialization
    global _worker_device
    
    # Create CNN and load state dict
    cnn = EnergyDistributionCNN(_worker_device)
    cnn.load_state_dict(cnn_state_dict)
    cnn._zero_hidden_channel_bias()
    
    # Create simulation instance
    sim = Simulation(enable_debug=False)
    sim.organism_manager.energy_distribution_cnn = cnn
    
    fitness, _, tick_data = run_cnn_fitness_rollout(sim, max_time, collect_tick_data, fitness_mode)
    
    # Explicitly clean up GPU memory
    del cnn
    del sim
    _release_device_memory(_worker_device)
    
    return (fitness, tick_data)


class CNNEvaluator:
    """CNN evaluator for GPU-accelerated fitness computation"""
    def __init__(self, world_size, max_time, device, fitness_mode=None):
        self.world_size = world_size
        self.max_time = max_time
        self.device = device
        self.fitness_mode = CNN_FITNESS_MODE if fitness_mode is None else fitness_mode
        validate_cnn_fitness_mode(self.fitness_mode)
        self.grapher = None
        self.current_generation_max_fitness = 0.0
        self.pool = None
        
    def _ensure_pool(self):
        """Create pool if it doesn't exist"""
        if self.pool is None:
            device_str = str(self.device)
            pool_kwargs = {
                'processes': TRAIN_WORKER_COUNT,
                'initializer': _init_worker,
                'initargs': (device_str,),
            }
            if TRAIN_WORKER_MAX_TASKS is not None:
                pool_kwargs['maxtasksperchild'] = TRAIN_WORKER_MAX_TASKS
            self.pool = multiprocessing.Pool(**pool_kwargs)
    
    def close_pool(self):
        """Close and terminate the multiprocessing pool"""
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join(timeout=5)
            except Exception:
                pass
            finally:
                try:
                    self.pool.terminate()
                    self.pool.join()
                except Exception:
                    pass
                self.pool = None
    
    def __del__(self):
        """Ensure pool is cleaned up on deletion"""
        self.close_pool()
        
    def evaluate_population(self, subjects, pop_size):
        """Evaluate entire population using multiprocessing"""
        # Ensure pool is created (reused across generations)
        self._ensure_pool()
        
        # Prepare arguments for each worker
        # Move state dicts to CPU for pickling (GPU/MPS tensors can't be pickled)
        collect_tick_data = self.grapher is not None
        args_list = []
        for i in range(pop_size):
            state_dict = subjects[i].state_dict()
            cpu_state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
            args_list.append(
                (cpu_state_dict, self.world_size, self.max_time, i, collect_tick_data, self.fitness_mode)
            )
        
        # Use multiprocessing to evaluate in parallel (reuse existing pool)
        results = self.pool.map(_evaluate_cnn_worker, args_list)
        
        # Extract fitness scores and process tick data
        fitness_scores = []
        if self.grapher is not None:
            for i, (fitness, tick_data) in enumerate(results):
                fitness_scores.append(fitness)
                # Process tick data for grapher
                for t, total_cell_count, current_cell_count, org_energy, env_energy, total in tick_data:
                    self.grapher.enqueue_tick(t, self.current_generation_max_fitness, [total_cell_count], org_energy, env_energy, total)
                    self.grapher.enqueue_bot_tick(i, t, total_cell_count, current_cell_count, env_energy, total)
                # Process queued updates periodically
                try:
                    self.grapher.process_queued()
                except Exception:
                    pass
        else:
            fitness_scores = [fitness for fitness, _ in results]
        
        # Clean up to free memory
        del args_list
        del results
        _release_device_memory(self.device)
        
        return fitness_scores
    
    def _evaluate_single_cnn(self, simulation, bot_index: int | None = None):
        """Evaluate a single CNN simulation"""
        collect_tick_data = self.grapher is not None
        fitness, cumulative_cell_count, tick_data = run_cnn_fitness_rollout(
            simulation, self.max_time, collect_tick_data, self.fitness_mode
        )
        if self.grapher is not None:
            for t, total_cell_count, current_cell_count, org_energy, env_energy, total in tick_data:
                self.grapher.enqueue_tick(t, self.current_generation_max_fitness, [fitness], org_energy, env_energy, total)
                if bot_index is not None:
                    self.grapher.enqueue_bot_tick(bot_index, t, fitness, current_cell_count, env_energy, total)
        return fitness


class CNNEvolutionDriver:
    """Evolution driver for CNN training with GPU evaluation and replay"""
    def __init__(self, world_size, epochs=100, max_time=100, fitness_mode=None):
        self.world_size = world_size
        self.epochs = epochs
        self.max_time = max_time
        self.fitness_mode = CNN_FITNESS_MODE if fitness_mode is None else fitness_mode
        validate_cnn_fitness_mode(self.fitness_mode)
        
        # Genetic algorithm parameters
        self.ga = CNNGeneticAlgorithm(CNN_POPULATION_SIZE, CNN_MUTATION_RATE, CNN_MUTATION_MAGNITUDE, device)
        
        # Evaluator
        self.evaluator = CNNEvaluator(world_size, max_time, device, self.fitness_mode)
        self.grapher = None
        
        # Replay simulation for showing best organism
        self.replay_simulation = None
        
    def evaluate_cnn(self, cnn, simulation):
        """Evaluate a CNN by running simulation and measuring fitness"""
        test_sim = Simulation(enable_debug=False)
        test_sim.organism_manager.energy_distribution_cnn = cnn
        fitness, _, _ = run_cnn_fitness_rollout(test_sim, self.max_time, fitness_mode=self.fitness_mode)
        return fitness
    
    def create_replay_simulation(self, best_cnn):
        """Create a simulation for replaying the best organism"""
        global current_best_cnn, replay_mode
        
        # Create new simulation with the best CNN
        self.replay_simulation = Simulation()
        self.replay_simulation.organism_manager.energy_distribution_cnn = best_cnn
        
        # Update global variables
        current_best_cnn = best_cnn
        replay_mode = True
        
        # Store replay simulation in main simulation for access (only if OpenGL sim is running)
        if 'current_simulation' in globals():
            try:
                current_simulation.replay_simulation = self.replay_simulation
            except Exception:
                pass
        
        print(f"Created replay simulation with best CNN (fitness/{cnn_fitness_mode_label()}: {self.ga.fitness_scores[self.ga.fittest_index]:.6f})")
        print("Press 'r' to toggle replay mode and see the best organism in action!")
        
    def run_evolution(self):
        """Run the evolution process with multiprocessing GPU evaluation"""
        print(f"\nStarting CNN Evolution - {self.epochs} generations")
        print(f"Population size: {CNN_POPULATION_SIZE}")
        print(f"Fitness mode: {cnn_fitness_mode_label(self.fitness_mode)} ({self.fitness_mode})")
        print(f"Using {TRAIN_WORKER_COUNT} worker processes (maxtasksperchild={TRAIN_WORKER_MAX_TASKS})")
        
        try:
            for gen in range(self.epochs):
                print(f"\nGeneration {gen + 1}/{self.epochs}")
                
                # Evaluate entire population sequentially
                print("Evaluating population...")
                if self.grapher is not None:
                    self.evaluator.grapher = self.grapher
                fitness_scores = self.evaluator.evaluate_population(
                    self.ga.subjects, 
                    CNN_POPULATION_SIZE
                )
                
                # Update fitness scores
                self.ga.fitness_scores = fitness_scores
                
                # Print individual results
                fitness_label = cnn_fitness_mode_label(self.fitness_mode)
                for i, fitness in enumerate(fitness_scores):
                    print(f"CNN {i}: fitness ({fitness_label}) = {fitness:.6f}")
                    
                # Run one generation
                self.ga.compute_generation()
                
                # Print summary
                best_fitness = self.ga.fitness_scores[self.ga.fittest_index]
                print(f"Best fitness ({fitness_label}): {best_fitness:.6f}")
                print(f"Best CNN conv1 weight shape: {self.ga.subjects[self.ga.fittest_index].conv1.weight.data.shape}")
                print(f"Best CNN conv2 weight shape: {self.ga.subjects[self.ga.fittest_index].conv2.weight.data.shape}")
                
                # Save the fittest network every generation
                self.ga.save_model(self.ga.fittest_index, generation=gen + 1)
                
                if self.grapher is not None:
                    best_cnn = self.ga.subjects[self.ga.fittest_index]
                    self.create_replay_simulation(best_cnn)

                # Update generation plot
                if self.grapher is not None:
                    # track history of best fitness
                    if not hasattr(self, 'best_history'):
                        self.best_history = []
                    self.best_history.append(best_fitness)
                    # update evaluator context
                    self.evaluator.grapher = self.grapher
                    self.evaluator.current_generation_max_fitness = best_fitness
                    # include per-bot fitnesses for colored series
                    self.grapher.enqueue_generation(gen + 1, self.best_history, fitness_scores)
                    self.grapher.process_queued()
                    # Clear tick-series for next generation window
                    self.grapher.reset_tick_metrics()
                
                # Reset for next generation
                self.ga.reset_fitness()
        finally:
            # Always close the multiprocessing pool, even if there's an exception
            self.evaluator.close_pool()
            
        print("\nEvolution completed!")
        return self.ga.subjects[self.ga.fittest_index]


class Environment:
    def __init__(self, world_size, noise_scale, quantization_step, organism_seed_positions):
        self.world_size = world_size
        self.noise_scale = noise_scale
        self.quantization_step = quantization_step
        self.organism_seed_positions = organism_seed_positions
        self.environment_type = ENVIRONMENT_TYPE
        self.time = 0.0
        self.pump_time = 0.0
        self._reset_pump_controller()
        self.terrain = self.generate_terrain()
    
    def generate_terrain(self):
        """Generate terrain based on environment type"""
        if self.environment_type == 1:
            # Type 1: Energy masks (static terrain with energy sources)
            return self._generate_energy_mask_terrain()
        elif self.environment_type == 2:
            self.base_raw = self._sample_perlin_octaves(0.0)
            self.base_terrain = self._perlin_to_terrain(self.base_raw)
            return self.base_terrain.clone()
        elif self.environment_type == 3:
            # Type 3: Moving perlin noise
            return self._generate_perlin_terrain()
        else:
            return self._empty_terrain()
    
    def _center_3x3_mask(self):
        cy = self.world_size // 2
        cx = self.world_size // 2
        mask = torch.zeros((self.world_size, self.world_size), dtype=torch.bool, device=device)
        mask[cy - 1 : cy + 2, cx - 1 : cx + 2] = True
        return mask

    def _generate_energy_mask_terrain(self):
        """Type 1: bright 3x3 at world center only."""
        terrain = torch.zeros((self.world_size, self.world_size), dtype=torch.float32, device=device)
        terrain[self._center_3x3_mask()] = STARTING_POSITION_TERRAIN_BOOST
        return torch.clamp(terrain, 0, 1)

    def _empty_terrain(self):
        return torch.zeros(
            (self.world_size, self.world_size), dtype=torch.float32, device=device
        )
    
    def _sample_perlin_octaves(self, time_value):
        """Raw 3D perlin sum at time_value; roughly in [-1, 1] before shaping."""
        x = torch.arange(self.world_size, dtype=torch.float32)
        y = torch.arange(self.world_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx_np = xx.numpy()
        yy_np = yy.numpy()
        noise_values = np.zeros((self.world_size, self.world_size), dtype=np.float32)
        for octave in range(NOISE_OCTAVES):
            octave_time = time_value * (1.0 + octave * 0.3) + octave * 5.0
            octave_scale = PERLIN_NOISE_SCALE * (2.0 ** octave)
            octave_noise = np.zeros((self.world_size, self.world_size), dtype=np.float32)
            for i in range(self.world_size):
                for j in range(self.world_size):
                    octave_noise[i, j] = pnoise3(
                        xx_np[i, j] * octave_scale,
                        yy_np[i, j] * octave_scale,
                        octave_time,
                        octaves=1,
                        base=octave * 10,
                    )
            noise_values += octave_noise / (2.0 ** octave)
        return torch.from_numpy(noise_values).to(device)

    def _perlin_to_terrain(self, raw):
        values = (raw + 1.0) * 0.5
        values = torch.pow(values, NOISE_POWER)
        dead_mask = values > ENV_NOISE_THRESHOLD
        values = values * dead_mask
        return torch.clamp(values, 0, 1)

    def _generate_perlin_terrain_at(self, time_value):
        return self._perlin_to_terrain(self._sample_perlin_octaves(time_value))

    def _generate_perlin_terrain(self):
        return self._generate_perlin_terrain_at(self.time)

    def _reset_pump_controller(self):
        self._pump_pid_integral = 0.0
        self._pump_pid_prev_error = 0.0
        self._pump_drive = 0.0

    def _centered_pump_wave(self):
        """Time-varying perlin layer (zero mean) summed with static base perlin."""
        raw = self._sample_perlin_octaves(self.pump_time)
        self.pump_time += PUMP_PERLIN_TIME_SPEED
        return raw - raw.mean()

    def _population_setpoint(self):
        if PUMP_POPULATION_SETPOINT is not None:
            return float(PUMP_POPULATION_SETPOINT)
        return PUMP_POPULATION_FRACTION * self.world_size * self.world_size

    def _pump_population_error(self, organism_manager):
        """Normalized error: (setpoint − alive) / setpoint, clamped to [-1, 1]."""
        alive = organism_manager.topology_matrix.sum().item()
        setpoint = self._population_setpoint()
        error = (setpoint - alive) / max(setpoint, 1.0)
        if error > 1.0:
            error = 1.0
        elif error < -1.0:
            error = -1.0
        if error > PUMP_PID_DEADBAND:
            error -= PUMP_PID_DEADBAND
        elif error < -PUMP_PID_DEADBAND:
            error += PUMP_PID_DEADBAND
        else:
            error = 0.0
        return error

    def _pump_pid_step(self, error):
        """PID on population; drive in [-clamp, clamp] (positive adds perlin wave)."""
        self._pump_pid_integral += error
        if self._pump_pid_integral > PUMP_PID_INTEGRAL_CLAMP:
            self._pump_pid_integral = PUMP_PID_INTEGRAL_CLAMP
        elif self._pump_pid_integral < -PUMP_PID_INTEGRAL_CLAMP:
            self._pump_pid_integral = -PUMP_PID_INTEGRAL_CLAMP
        derivative = error - self._pump_pid_prev_error
        self._pump_pid_prev_error = error
        raw = (
            PUMP_PID_KP * error
            + PUMP_PID_KI * self._pump_pid_integral
            + PUMP_PID_KD * derivative
        )
        if raw > PUMP_PID_CLAMP:
            if error > 0.0:
                self._pump_pid_integral -= error
            raw = PUMP_PID_CLAMP
        elif raw < -PUMP_PID_CLAMP:
            if error < 0.0:
                self._pump_pid_integral -= error
            raw = -PUMP_PID_CLAMP
        delta = raw - self._pump_drive
        if delta > PUMP_DRIVE_SLEW:
            delta = PUMP_DRIVE_SLEW
        elif delta < -PUMP_DRIVE_SLEW:
            delta = -PUMP_DRIVE_SLEW
        self._pump_drive += delta
        return self._pump_drive

    def _apply_thermodynamic_pump(self, organism_manager, tick_harvest=0.0):
        """
        PID bias on perlin wave + small mean shift; capped per-cell delta.
        """
        if organism_manager is None:
            return
        error = self._pump_population_error(organism_manager)
        drive = self._pump_pid_step(error)
        cell_count = self.world_size * self.world_size
        delta = drive * PUMP_WAVE_RATE * self._centered_pump_wave()
        delta += drive * PUMP_MEAN_RATE / cell_count
        if drive > 0.0:
            pull = torch.clamp(self.base_terrain - self.terrain, min=0.0)
            delta += PUMP_TRAIL_RELAX_RATE * pull
        delta = torch.clamp(delta, -PUMP_MAX_CELL_DELTA, PUMP_MAX_CELL_DELTA)
        self.terrain += delta

    def _advance_moving_perlin(self):
        field = self._generate_perlin_terrain_at(self.time)
        self.time += PERLIN_TIME_SPEED
        return field

    def compute_environment(self, topology_matrix, harvested_energy, organism_manager=None):
        """Modify environment based on organism presence"""
        self.terrain.copy_(torch.clamp(self.terrain - (harvested_energy * topology_matrix), 0, 1))

        if self.environment_type == 3:
            self.terrain.copy_(self._advance_moving_perlin())
        elif self.environment_type == 2 and TERRAIN_PUMP_ENABLED:
            self._apply_thermodynamic_pump(organism_manager)

        if self.environment_type == 1:
            center_mask = self._center_3x3_mask()
            self.terrain[center_mask] = torch.clamp(
                self.terrain[center_mask] + TERRAIN_PUMP_RATE,
                max=1.0,
            )

        self.terrain.clamp_(0.0, 1.0)

def random_organism_positions(world_size, organism_count):
    """Pick organism_count distinct random cells (x, y) on the world grid."""
    if organism_count <= 0:
        return []
    n = min(organism_count, world_size * world_size)
    flat = torch.randperm(world_size * world_size, device=device)[:n]
    xs = flat % world_size
    ys = flat // world_size
    return [[int(xs[i].item()), int(ys[i].item())] for i in range(n)]


class OrganismManager:
    def __init__(self, world_size, organism_count, terrain, seed_positions=None):
        self.world_size = world_size
        self.terrain = terrain

        if seed_positions is None:
            seed_positions = random_organism_positions(world_size, organism_count)
        self.positions = torch.tensor(seed_positions, dtype=torch.long, device=device)
        self.topology_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.energy_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.sharing_rate_matrix = torch.full((world_size, world_size), SHARING_OFF_VALUE, dtype=torch.float32, device=device)
        self.hidden_channels = torch.zeros((1, world_size, world_size), dtype=torch.float32, device=device)
        self.rotation_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.parent_giver_dir = torch.full((world_size, world_size), -1, dtype=torch.long, device=device)
        self.new_cell_candidates = torch.zeros((world_size, world_size), dtype=torch.bool, device=device)
        self.pending_birth_energy = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.destroyed_energy = 0.0
        self.destroyed_entropy = 0.0
        self.last_tick_destroyed_energy = 0.0
        self._tick_destroyed = torch.zeros((), device=device, dtype=torch.float32)
        self._tick_entropy = torch.zeros((), device=device, dtype=torch.float32)
        self._has_new_cell_candidates = False
        self.colony_mutation_enabled = False
        self.colony_mutation_graph = None
        self._initialize_topology()
        
        # Reproduction parameters
        self.reproduction_threshold = REPRODUCTION_THRESHOLD
        
        # Energy distribution CNN
        self.energy_distribution_cnn = EnergyDistributionCNN(device)
        self._init_contribution_conv_weights()
        self._init_giver_dir_gather()

    def enable_colony_mutation(self):
        """Lineage mutations (--lineage): differential transforms composed along parent_giver_dir tree."""
        if self.colony_mutation_enabled:
            return
        self.colony_mutation_enabled = True
        self.colony_mutation_graph = ColonyMutationGraph(
            self.world_size,
            device,
            CNN_MUTATION_RATE,
            CNN_MUTATION_MAGNITUDE,
        )
        if self.topology_matrix.sum().item() > 0 and self.positions.numel() > 0:
            y_coords = self.positions[:, 1]
            x_coords = self.positions[:, 0]
            for idx in range(y_coords.shape[0]):
                y = y_coords[idx].item()
                x = x_coords[idx].item()
                if self.topology_matrix[y, x].item() > 0:
                    self.colony_mutation_graph.register_seed(y, x)
        print(
            f"Colony lineage mutations enabled "
            f"(rate={CNN_MUTATION_RATE}, magnitude={CNN_MUTATION_MAGNITUDE})"
        )

    def _register_colony_mutations(self, energy_mask, birth_mask, dominant_giver_dir):
        graph = self.colony_mutation_graph
        child_y, child_x = energy_mask.nonzero(as_tuple=True)
        if child_y.numel() == 0:
            return
        has_parent = birth_mask[child_y, child_x]
        parent_y = child_y.clone()
        parent_x = child_x.clone()
        if has_parent.any():
            g = dominant_giver_dir[child_y, child_x]
            parent_y = torch.where(
                has_parent,
                self._giver_source_y[g, child_y, child_x],
                parent_y,
            )
            parent_x = torch.where(
                has_parent,
                self._giver_source_x[g, child_y, child_x],
                parent_x,
            )
        graph.register_births(child_y, child_x, parent_y, parent_x, has_parent, self.topology_matrix)

    @staticmethod
    def _make_contrib_accum_weight(device):
        weight = torch.zeros(1, 9, 3, 3, device=device, dtype=torch.float32)
        for ci in range(3):
            for cj in range(3):
                weight[0, ci * 3 + cj, 2 - ci, 2 - cj] = 1.0
        return weight

    @staticmethod
    def _make_dest_eff_gather_weight(device):
        weight = torch.zeros(9, 1, 3, 3, device=device, dtype=torch.float32)
        for ci in range(3):
            for cj in range(3):
                weight[ci * 3 + cj, 0, ci, cj] = 1.0
        return weight

    def _init_contribution_conv_weights(self):
        self._contrib_accum_weight = self._make_contrib_accum_weight(device)
        self._dest_eff_gather_weight = self._make_dest_eff_gather_weight(device)
        self._org_avg_weight = torch.full((1, 1, 3, 3), 1.0 / 9.0, device=device, dtype=torch.float32)

    def _init_giver_dir_gather(self):
        ring_cij = EnergyDistributionCNN._RING_CIJ
        y_coords = torch.arange(self.world_size, device=device).view(self.world_size, 1).expand(self.world_size, self.world_size)
        x_coords = torch.arange(self.world_size, device=device).view(1, self.world_size).expand(self.world_size, self.world_size)
        source_y = []
        source_x = []
        oci_list = []
        ocj_list = []
        for g in range(8):
            sci, scj = ring_cij[g]
            oci, ocj = ring_cij[(g + 4) % 8]
            source_y.append((y_coords + sci - 1) % self.world_size)
            source_x.append((x_coords + scj - 1) % self.world_size)
            oci_list.append(oci)
            ocj_list.append(ocj)
        self._giver_source_y = torch.stack(source_y)
        self._giver_source_x = torch.stack(source_x)
        self._giver_oc = torch.tensor(oci_list, device=device, dtype=torch.long)
        self._giver_ocj = torch.tensor(ocj_list, device=device, dtype=torch.long)

    def _gather_inbound_by_giver_dir(self, contributions):
        """(8, H, W) inbound contribution at each cell from parent at giver direction g."""
        return contributions[
            self._giver_oc[:, None, None],
            self._giver_ocj[:, None, None],
            self._giver_source_y,
            self._giver_source_x,
        ]

    def _add_destroyed_energy(self, energy_delta):
        """Route energy loss to destroyed pool with ΔS = ΔE / T_env."""
        if isinstance(energy_delta, torch.Tensor):
            energy_delta = energy_delta.sum()
        delta = float(energy_delta)
        if delta <= 0.0:
            return
        self._tick_destroyed += delta
        self._tick_entropy += delta / THERMO_ENV_TEMPERATURE

    def _add_entropy_production(self, entropy_delta):
        """Record irreversible entropy production (e.g. decay σ_dot) without energy bookkeeping."""
        if isinstance(entropy_delta, torch.Tensor):
            entropy_delta = entropy_delta.sum()
        delta = float(entropy_delta)
        if delta <= 0.0:
            return
        self._tick_entropy += delta

    def _flush_tick_destroyed(self):
        self.last_tick_destroyed_energy = self._tick_destroyed.item()
        self.destroyed_energy += self.last_tick_destroyed_energy
        self.destroyed_entropy += self._tick_entropy.item()
        self._tick_destroyed.zero_()
        self._tick_entropy.zero_()

    def _shift_sum_contributions(self, contributions):
        """Sum shifted contribution channels onto each destination cell via circular conv."""
        stacked = contributions.reshape(1, 9, self.world_size, self.world_size)
        padded = torch.nn.functional.pad(stacked, (1, 1, 1, 1), mode='circular')
        return torch.nn.functional.conv2d(padded, self._contrib_accum_weight).squeeze(0).squeeze(0)

    def _compute_parent_incoming(self, contributions):
        """Energy each cell receives from its parent (child->parent direction stored per cell)."""
        inbound_by_dir = self._gather_inbound_by_giver_dir(contributions)
        has_parent = self.parent_giver_dir >= 0
        parent_incoming = inbound_by_dir.gather(0, self.parent_giver_dir.clamp(min=0).unsqueeze(0)).squeeze(0)
        return parent_incoming * has_parent.float()
    
    def _place_seed_cell(self, y, x):
        """Place one seed organism at (y, x) with default seed state."""
        self.topology_matrix[y, x] = 1
        self.energy_matrix[y, x] = 1
        self.sharing_rate_matrix[y, x] = SHARING_ON_VALUE if ENERGY_SHARING_RATE > 0 else SHARING_OFF_VALUE
        self.hidden_channels[:, y, x] = 0
        self.rotation_matrix[y, x] = 0
        self.parent_giver_dir[y, x] = -1
        if self.colony_mutation_enabled:
            self.colony_mutation_graph.register_seed(y, x)

    def _initialize_topology(self):
        """Initialize topology and energy with organism positions"""
        if self.positions.numel() > 0:
            y_coords, x_coords = self.positions[:, 1], self.positions[:, 0]
            for idx in range(y_coords.shape[0]):
                self._place_seed_cell(y_coords[idx].item(), x_coords[idx].item())

    def reseed_organism_positions_if_extinct(self):
        """--lineage: when the colony dies out, re-seed at initial random seed positions."""
        if not self.colony_mutation_enabled:
            return
        if self.topology_matrix.sum().item() != 0:
            return
        if self.positions.numel() == 0:
            return
        y_coords, x_coords = self.positions[:, 1], self.positions[:, 0]
        for idx in range(y_coords.shape[0]):
            self._place_seed_cell(y_coords[idx].item(), x_coords[idx].item())
        print("Colony extinct — reseeded organism position(s)")
    
    def compute_topology(self):
        """Reproduce using new_cell_candidates mask from energy sharing"""
        if not self._has_new_cell_candidates:
            if not self.new_cell_candidates.any().item():
                return
        self._has_new_cell_candidates = False
        
        # Birth when incoming shared energy at candidate meets threshold
        energy_mask = self.new_cell_candidates & (self.pending_birth_energy >= self.reproduction_threshold)
        
        # Parent, rotation, sharing rate, and hidden (from parent hidden state) for new cells
        child_sharing_rate = torch.full_like(
            self.sharing_rate_matrix,
            SHARING_ON_VALUE if ENERGY_SHARING_RATE > 0 else SHARING_OFF_VALUE,
        )
        child_hidden_at_birth = torch.zeros_like(self.sharing_rate_matrix)
        dominant_giver_dir = None
        birth_mask = torch.zeros_like(energy_mask)
        if self.new_cell_contributions is not None:
            contrib_by_giver_dir = self._gather_inbound_by_giver_dir(self.new_cell_contributions)
            total_weight = contrib_by_giver_dir.sum(dim=0)
            dominant_giver_dir = torch.argmax(contrib_by_giver_dir, dim=0)
            birth_mask = energy_mask & (total_weight > 0)
            self.parent_giver_dir[birth_mask] = dominant_giver_dir[birth_mask]
            rotation_bucket = (dominant_giver_dir - 2) % 8
            snapped_angles = rotation_bucket.float() * (torch.pi / 4)
            self.rotation_matrix[birth_mask] = snapped_angles[birth_mask]
            parent_hidden_by_g = self.hidden_channels[0, self._giver_source_y, self._giver_source_x]
            child_hidden = parent_hidden_by_g.gather(0, dominant_giver_dir.unsqueeze(0).clamp(min=0)).squeeze(0)
            # Map binary hidden (0/1) into sharing rates (OFF=0.1, ON=0.9)
            child_sharing_rate = torch.where(
                birth_mask,
                SHARING_OFF_VALUE + child_hidden * (SHARING_ON_VALUE - SHARING_OFF_VALUE),
                child_sharing_rate,
            )
            child_hidden_at_birth = torch.where(birth_mask, child_hidden, child_hidden_at_birth)
        
        # Add selected positions to topology and commit birth energy
        self.topology_matrix[energy_mask] = 1
        self.energy_matrix[energy_mask] = self.pending_birth_energy[energy_mask]
        self.pending_birth_energy[energy_mask] = 0
        
        self.sharing_rate_matrix[energy_mask] = child_sharing_rate[energy_mask]
        self.hidden_channels[0, energy_mask] = child_hidden_at_birth[energy_mask]

        if self.colony_mutation_enabled and energy_mask.any():
            if dominant_giver_dir is None:
                dominant_giver_dir = torch.zeros_like(self.parent_giver_dir)
            self._register_colony_mutations(energy_mask, birth_mask, dominant_giver_dir)
    
    def _apply_harvest_and_decay(self, terrain):
        """Harvest energy from terrain and apply decay"""
        if terrain is not None:
            self.terrain = terrain
                
        # Remove organisms with energy below threshold
        low_energy_mask = self.energy_matrix < DEATH_THRESHOLD
        self._add_destroyed_energy(self.energy_matrix * low_energy_mask.float())
        self.energy_matrix[low_energy_mask] = 0
        self.topology_matrix[low_energy_mask] = 0
        self.sharing_rate_matrix[low_energy_mask] = SHARING_OFF_VALUE
        self.hidden_channels[:, low_energy_mask] = 0
        self.rotation_matrix[low_energy_mask] = 0
        self.parent_giver_dir[low_energy_mask] = -1
        if self.colony_mutation_enabled:
            self.colony_mutation_graph.clear_cells_mask(low_energy_mask)

        harvested_energy, harvest_sigma_dot = thermodynamic_harvest_step(
            self.terrain,
            self.energy_matrix,
            self.sharing_rate_matrix,
            self.topology_matrix,
        )
        self._tick_entropy += harvest_sigma_dot.sum()
        
        energy_unsqueezed = self.energy_matrix.unsqueeze(0).unsqueeze(0)
        org_avg = torch.nn.functional.conv2d(energy_unsqueezed, self._org_avg_weight, padding=1).squeeze(0).squeeze(0)
        energy_after_harvest = self.energy_matrix + harvested_energy
        cell_decay, _sigma_dot = thermodynamic_decay_step(
            energy_after_harvest,
            self.sharing_rate_matrix,
            org_avg,
            self.topology_matrix,
        )
        energy_before_decay = energy_after_harvest * self.topology_matrix
        self.energy_matrix = torch.clamp((energy_after_harvest - cell_decay) * self.topology_matrix, 0, 1)
        self._add_destroyed_energy(energy_before_decay - self.energy_matrix)

        return harvested_energy
    
    def _compute_energy_contributions(self, shareable_energy, proportions):
        """Compute energy contributions from each source to each neighbor"""
        # (H, W) * (3, 3, H, W) -> (3, 3, H, W)
        contributions = shareable_energy.unsqueeze(0).unsqueeze(0) * proportions
        return contributions
    
    def _accumulate_contributions(self, contributions, shareable_energy):
        """Accumulate contributions to destination cells"""
        return self.energy_matrix - shareable_energy + self._shift_sum_contributions(contributions)
    
    def _compute_source_removed(self, contributions, dest_efficiency, receiving_mask):
        """Compute how much energy each source should lose based on what was received"""
        padded = torch.nn.functional.pad(
            dest_efficiency.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode='circular',
        )
        shifted_eff = torch.nn.functional.conv2d(padded, self._dest_eff_gather_weight).squeeze(0)
        return (contributions.reshape(9, self.world_size, self.world_size) * shifted_eff).sum(0)

    def _apply_capacity_constraints(self, new_energy_matrix, receiving_mask):
        """Apply capacity constraints to limit received energy"""
        capacity = (1.0 - self.energy_matrix) * receiving_mask
        energy_incoming = new_energy_matrix - self.energy_matrix
        actual_received = torch.min(energy_incoming, capacity)
        new_energy_matrix = self.energy_matrix + actual_received
        return new_energy_matrix, actual_received, energy_incoming
    
    def compute_energy(self, terrain):
        """Main energy computation: harvest, decay, and sharing"""
        self._tick_destroyed.zero_()
        self._tick_entropy.zero_()
        self._add_destroyed_energy(self.pending_birth_energy.sum())
        harvested_energy = self._apply_harvest_and_decay(terrain)
        
        # Shareable outbound energy (new tensor; energy_matrix is updated later in sharing)
        shareable_energy = self.energy_matrix * self.topology_matrix * self.sharing_rate_matrix
        
        mutation_transform = None
        mutation_apply_mask = None
        lineage_mutation_cell_count = 0
        if self.colony_mutation_enabled:
            graph = self.colony_mutation_graph
            lineage_mutation_cell_count = graph.lineage_mutation_cell_count
            if lineage_mutation_cell_count > 0:
                mutation_transform = graph.transform_field
                mutation_apply_mask = graph.mutation_apply_mask(self.topology_matrix)
        proportions, _, hidden_channels_output = self.energy_distribution_cnn(
            shareable_energy,
            terrain,
            self.sharing_rate_matrix,
            self.hidden_channels,
            self.rotation_matrix,
            mutation_transform=mutation_transform,
            mutation_apply_mask=mutation_apply_mask,
            lineage_mutation_cell_count=lineage_mutation_cell_count,
        )
        
        # Update hidden_channels with CNN output (only for cells that exist)
        topology_mask = self.topology_matrix.unsqueeze(0)  # (1, H, W)
        self.hidden_channels = self.hidden_channels * (1 - topology_mask) + hidden_channels_output * topology_mask
        
        # Compute contributions
        contributions = self._compute_energy_contributions(shareable_energy, proportions)
        
        full_distributed = self._shift_sum_contributions(contributions)
        parent_incoming = self._compute_parent_incoming(contributions)
        has_parent = (self.parent_giver_dir >= 0) & (self.topology_matrix > 0)
        is_seed = (self.topology_matrix > 0) & (self.parent_giver_dir < 0)
        distributed_total = full_distributed
        distributed_total = torch.where(has_parent, shareable_energy + parent_incoming, distributed_total)
        distributed_total = torch.where(is_seed, shareable_energy, distributed_total)
        
        # Calculate new cell candidates
        self.new_cell_candidates = (distributed_total > self.reproduction_threshold) & (self.topology_matrix == 0)
        self._has_new_cell_candidates = self.new_cell_candidates.any().item()
        self.new_cell_contributions = contributions
        
        # Receiving mask (candidates included for sharing bookkeeping, not energy_matrix)
        receiving_mask = (self.topology_matrix.bool() | self.new_cell_candidates).float()
        self.pending_birth_energy = torch.zeros_like(self.energy_matrix)
        
        capacity = (1.0 - self.energy_matrix) * receiving_mask
        energy_incoming = distributed_total - shareable_energy
        actual_received = torch.min(energy_incoming, capacity)
        self.pending_birth_energy = actual_received * self.new_cell_candidates.float()
        
        dest_efficiency = torch.where(
            energy_incoming > 0,
            actual_received / energy_incoming,
            torch.zeros_like(energy_incoming)
        ) * receiving_mask
        
        source_removed = self._compute_source_removed(contributions, dest_efficiency, receiving_mask)
        total_received = actual_received.sum()
        total_removed = source_removed.sum()
        scale = torch.where(
            total_removed > total_received,
            total_received / total_removed.clamp(min=1e-12),
            torch.ones((), device=device, dtype=torch.float32),
        )
        source_removed = source_removed * scale
        
        incoming = distributed_total - shareable_energy
        valid_mask = self.topology_matrix
        unclamped = (
            self.energy_matrix
            + incoming * dest_efficiency
            - source_removed
        ) * valid_mask
        self.energy_matrix = torch.clamp(unclamped, 0, 1)
        self._add_destroyed_energy((unclamped - self.energy_matrix).sum())
        
        self._flush_tick_destroyed()
        return harvested_energy
class Renderer:
    def __init__(self, world_size):
        self.world_size = world_size
        self.render_size = int(world_size * PIXEL_SCALE_FACTOR)
        self.render_mode = "org_energy"  # "org_top" or "org_energy"
        self.filters_enabled = True  # True = show organisms, False = environment only
        self.texture_id = None
        self.quad_vbo = None
        self.opengl_initialized = False
        self.left_margin = 300
        self.top_margin = 50
        self.bottom_margin = 50
        self.right_margin = 50
        self.debug_text_enabled = True
        self.org_energy_view_enabled = True
        self.sharing_rate_raw_view = True
        self.hidden_channel_0_view_enabled = True
        self.genome_view_enabled = False
        self.debug_panel_mode = "stats"
    
    def toggle_genome_view(self):
        """Tab: genome-colored cells (lineage) and genome list overlay."""
        self.genome_view_enabled = not self.genome_view_enabled
        self.debug_panel_mode = "genomes" if self.genome_view_enabled else "stats"
        print(
            f"Genome view: {'ON' if self.genome_view_enabled else 'OFF'} "
            f"(overlay={self.debug_panel_mode})"
        )
    
    def toggle_render_mode(self):
        """Toggle between organism topology and energy visualization"""
        self.render_mode = "org_energy" if self.render_mode == "org_top" else "org_top"
        print(f"Render mode: {self.render_mode}")
    
    def toggle_filters(self):
        """Toggle between showing organisms (enabled) and environment only (disabled)"""
        self.filters_enabled = not self.filters_enabled
        if self.filters_enabled:
            print(f"Filters enabled - showing organisms ({self.render_mode})")
        else:
            print("Filters disabled - showing environment only")
    
    def toggle_debug_text(self):
        """Toggle debug text display"""
        self.debug_text_enabled = not self.debug_text_enabled
        if self.debug_text_enabled:
            print("Debug text enabled")
        else:
            print("Debug text disabled")
    
    def toggle_org_energy_view(self):
        """Toggle organism energy view visualization"""
        self.org_energy_view_enabled = not self.org_energy_view_enabled
        if self.org_energy_view_enabled:
            print("Org energy view enabled")
        else:
            print("Org energy view disabled")
    
    def toggle_sharing_rate_view(self):
        """Toggle between raw sharing rate values and thresholded mask"""
        self.sharing_rate_raw_view = not self.sharing_rate_raw_view
        if self.sharing_rate_raw_view:
            print("Sharing rate: raw 0-1 values")
        else:
            print("Sharing rate: thresholded mask (>0.5)")

    def toggle_hidden_channel_0_view(self):
        """Toggle green overlay for hidden channel 0"""
        self.hidden_channel_0_view_enabled = not self.hidden_channel_0_view_enabled
        print(f"Hidden channel 0 view (green): {'ON' if self.hidden_channel_0_view_enabled else 'OFF'}")

    def render_text(self, x, y, text):
        """Render text at specified position"""
        try:
            # Use glRasterPos2f with current projection/modelview matrices
            glRasterPos2f(x, y)
            for char in text:
                glut.glutBitmapCharacter(glut.GLUT_BITMAP_8_BY_13, ord(char))
        except Exception as e:
            print(f"Text rendering error: {e}")
    
    def render_debug_info(self, simulation_data, current_harvest_rate, replay_mode, current_best_cnn, logger=None):
        """Render debug information overlay"""
        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        
        # Set viewport for debug text (full window)
        window_width = self.render_size + self.left_margin + self.right_margin
        window_height = self.render_size + self.top_margin + self.bottom_margin
        glViewport(0, 0, window_width, window_height)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, window_width, window_height, 0, -1, 1)  # Flip Y axis
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable texture mapping and other states that interfere with text rendering
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set text color (bright green)
        glColor4f(0.0, 1.0, 0.0, 1.0)
        
        # Calculate text position (top-left corner, flipped coordinates)
        # Add top_margin offset so text doesn't clip at the top
        text_x = 10
        text_y = 20
        line_height = 15
        
        # Get actual FPS from logger
        actual_fps = logger.get_fps() if logger else 0.0

        panel_mode = simulation_data.get("debug_panel_mode", "stats")
        if panel_mode == "genomes" and simulation_data.get("lineage_enabled"):
            header_lines = [
                f"FPS: {actual_fps:.1f}",
                "",
                f"=== GENOMES (Tab) {simulation_data.get('genome_count', 0)} present, top 10 ===",
            ]
            for i, line in enumerate(header_lines):
                glColor4f(0.0, 1.0, 0.0, 1.0)
                self.render_text(text_x, text_y + (i * line_height), line)
            genome_entries = simulation_data.get("genome_panel_lines", [("(no data)", (1.0, 1.0, 1.0))])
            row = len(header_lines)
            for entry in genome_entries:
                if isinstance(entry, tuple) and len(entry) == 2:
                    line, color = entry
                    glColor4f(color[0], color[1], color[2], 1.0)
                else:
                    line = entry
                    glColor4f(0.0, 1.0, 0.0, 1.0)
                self.render_text(text_x, text_y + (row * line_height), line)
                row += 1
            glPopAttrib()
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            return
        
        # Render debug information
        lines = [
            f"FPS: {actual_fps:.1f}",
            f"",
            f"=== TOGGLES ===",
            f"Render Mode (m): {self.render_mode}",
            f"Sharing Rate View (v): {'RAW' if self.sharing_rate_raw_view else 'MASK'}",
            f"Org Energy View (b): {'ON' if self.org_energy_view_enabled else 'OFF'}",
            f"Hidden green (q): {'ON' if self.hidden_channel_0_view_enabled else 'OFF'}",
            f"Cell colors: white=100 red=low share green=h0 blue=h1",
            f"Filters (n): {'ON' if self.filters_enabled else 'OFF'}",
            f"Harvesting (h): {'ON' if current_harvest_rate > 0 else 'OFF'}",
            f"",
            f"=== ENVIRONMENT CONFIG ===",
            f"Noise Scale: {NOISE_SCALE}",
            f"Quantization Step: {QUANTIZATION_STEP}",
            f"Noise Frequency Multiplier: {NOISE_FREQUENCY_MULTIPLIER}",
            f"Noise Octaves: {NOISE_OCTAVES}",
            f"Noise Power: {NOISE_POWER}",
            f"",
            f"=== ORGANISM CONFIG ===",
            f"Seed Count: {ORGANISM_COUNT}",
            f"Energy Sharing Rate: {ENERGY_SHARING_RATE}",
            f"Energy Harvest Rate: {ENERGY_HARVEST_RATE:.4f}",
            f"Energy Decay: {ENERGY_DECAY:.4f}",
            f"Reproduction Threshold: {REPRODUCTION_THRESHOLD:.4f}",
            f"Death Threshold: {DEATH_THRESHOLD:.4f}",
            f"Seed Boost: {STARTING_POSITION_TERRAIN_BOOST}",
            f"",
            f"=== SIMULATION STATE ===",
        ]
        
        # Add simulation data if available
        if simulation_data:
            organism_energy = torch.sum(simulation_data['energy']).item()
            terrain_energy = torch.sum(simulation_data['terrain']).item()
            pending_energy = torch.sum(simulation_data['pending_birth_energy']).item()
            destroyed_energy = simulation_data['destroyed_energy']
            destroyed_entropy = simulation_data['destroyed_entropy']
            system_energy = organism_energy + terrain_energy + pending_energy + destroyed_energy
            topology_count = torch.sum(simulation_data['topology']).item()
            new_cells = torch.sum(simulation_data['new_cell_candidates']).item()

            lines.extend([
                f"System Energy: {system_energy:.2f}",
                f"Terrain Energy: {terrain_energy:.2f}",
                f"Organism Energy: {organism_energy:.2f}",
                f"Pending Birth Energy: {pending_energy:.2f}",
                f"Destroyed Energy: {destroyed_energy:.2f}",
                f"Destroyed Entropy: {destroyed_entropy:.2f}",
                f"Entropy Produced (cum): {simulation_data['entropy_produced']:.2f}",
                f"Cells: {topology_count:.0f}",
                f"New Cell Candidates: {new_cells:.0f}",
            ])
            if 'system_entropy' in simulation_data:
                organism_entropy = simulation_data['organism_entropy'].item()
                terrain_entropy = simulation_data['terrain_entropy'].item()
                pending_entropy = simulation_data['pending_entropy'].item()
                system_entropy = simulation_data['system_entropy'].item()
                lines.extend([
                    f"Organism Entropy: {organism_entropy:.2f}",
                    f"Terrain Entropy: {terrain_entropy:.2f}",
                    f"Pending Entropy: {pending_entropy:.2f}",
                    f"System Entropy: {system_entropy:.2f}",
                ])

        # Render each line
        for i, line in enumerate(lines):
            self.render_text(text_x, text_y + (i * line_height), line)
        
        # Restore OpenGL state
        glPopAttrib()
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def render(
        self,
        environment,
        topology,
        mask,
        new_cell_candidates=None,
        sharing_rate=None,
        hidden_channels=None,
        genome_colors=None,
    ):
        """Render the current state using PyTorch tensors directly - GPU accelerated"""
        env_scaled = environment.clamp(0, 1)
        
        # Create RGBA image tensor
        image = torch.zeros((4, self.world_size, self.world_size), device=device, dtype=torch.float32)
        
        # Apply environment to all channels
        image[0] = 0.0
        image[1] = env_scaled  # Green channel  
        image[2] = env_scaled  # Blue channel
        image[3] = 1.0  # Alpha channel (fully opaque by default)
        
        # Only apply organism visualization if filters are enabled
        if self.filters_enabled and genome_colors is not None:
            org = topology.bool()
            image[0] = torch.where(org, genome_colors[0], image[0])
            image[1] = torch.where(org, genome_colors[1], image[1])
            image[2] = torch.where(org, genome_colors[2], image[2])
            if self.render_mode == "org_energy" and self.org_energy_view_enabled:
                image[3] = org.float() * torch.clamp(mask, 0.1, 1) + (~org).float() * env_scaled
            return image
        
        if self.filters_enabled and sharing_rate is not None:
            org = topology
            sh = sharing_rate.clamp(0, 1) * org
            h0 = torch.zeros_like(org)
            if hidden_channels is not None:
                h0 = hidden_channels[0] * org
            if not self.hidden_channel_0_view_enabled:
                h0 = torch.zeros_like(h0)

            # Red = sharing off, green = hidden; sharing on + hidden off -> white
            org_r = 1 - sh
            org_g = h0
            org_b = torch.zeros_like(org)
            all_off = org.bool() & (sh > 0.5) & (h0 == 0)
            org_r = torch.where(all_off, torch.ones_like(org), org_r)
            org_g = torch.where(all_off, torch.ones_like(org), org_g)
            org_b = torch.where(all_off, torch.ones_like(org), org_b)

            image[0] = torch.where(org.bool(), org_r, image[0])
            image[1] = torch.where(org.bool(), org_g, image[1])
            image[2] = torch.where(org.bool(), org_b, image[2])

            if self.render_mode == "org_energy" and self.org_energy_view_enabled:
                image[3] = org * torch.clamp(mask, 0.1, 1) + (1 - org) * env_scaled
        
        return image
    
    def _setup_opengl(self):
        """Setup OpenGL components for GPU-accelerated rendering (OpenGL 2.1 compatible)"""
        # Create texture
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # Create quad vertices for full-screen rendering (OpenGL 2.1 style)
        self.quad_vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,  # Bottom-left
             1.0, -1.0, 1.0, 0.0,  # Bottom-right
             1.0,  1.0, 1.0, 1.0,  # Top-right
            -1.0,  1.0, 0.0, 1.0   # Top-left
        ], dtype=np.float32)
        
        # Create VBO (no VAO for OpenGL 2.1 compatibility)
        self.quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.quad_vertices.nbytes, self.quad_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    def update_texture(self, image_tensor):
        """Update OpenGL texture with minimal CPU transfer using GPU-optimized operations"""
        if not self.opengl_initialized:
            return
            
        # All processing stays on GPU until the very last step
        # Ensure tensor is on GPU
        from config import DEVICE_TYPE
        if image_tensor.device.type != 'cuda' and image_tensor.device.type != DEVICE_TYPE:
            image_tensor = image_tensor.to(device)
        
        # Upscale image tensor by sampling each pixel d times in each dimension
        if PIXEL_SCALE_FACTOR > 1:
            image_tensor = image_tensor.repeat_interleave(PIXEL_SCALE_FACTOR, dim=1).repeat_interleave(PIXEL_SCALE_FACTOR, dim=2)
        
        # Process entirely on GPU: clamp, scale, permute, convert to uint8
        image_tensor = image_tensor.clamp(0, 1) * PIXEL_SCALE
        image_tensor = image_tensor.permute(1, 2, 0)  # CHW -> HWC
        image_tensor = image_tensor.byte()  # Convert to uint8 on GPU
        
        # Only transfer to CPU at the very end for OpenGL texture upload
        # This is the minimal possible CPU transfer
        image_np = image_tensor.cpu().numpy()
        
        # Update OpenGL texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.render_size, self.render_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_np)
    
    def render_opengl(self, simulation_data=None, current_harvest_rate=0, replay_mode=False, current_best_cnn=None, logger=None):
        """Render using OpenGL (OpenGL 2.1 compatible) with debug text"""
        if not self.opengl_initialized:
            return
            
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Set viewport for world rendering (offset by margins)
        # OpenGL viewport Y is measured from bottom-left corner
        # Position world so its top edge aligns with debug text (top_margin from top)
        window_height = self.render_size + self.top_margin + self.bottom_margin
        # To have top_margin at top: viewport_y = window_height - top_margin - render_size
        # This positions the world's top edge at top_margin from the window top
        viewport_y = window_height - self.top_margin - self.render_size
        glViewport(self.left_margin, viewport_y, self.render_size, self.render_size)
        
        # Ensure correct matrix mode for world rendering
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Enable texture mapping and blending for alpha
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Render full-screen quad using immediate mode (no VBOs)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, -1.0)  # Bottom-left
        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, -1.0)   # Bottom-right
        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, 1.0)     # Top-right
        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1.0, 1.0)    # Top-left
        glEnd()
        
        # Render debug text overlay
        self.render_debug_info(simulation_data, current_harvest_rate, replay_mode, current_best_cnn, logger)
        
        # Swap buffers
        glutSwapBuffers()
    

class Simulation:
    def __init__(self, enable_debug: bool = True):
        self.world_size = WORLD_SIZE
        self.seed_positions = random_organism_positions(self.world_size, ORGANISM_COUNT)
        self.environment = Environment(
            self.world_size,
            NOISE_SCALE,
            QUANTIZATION_STEP,
            self.seed_positions,
        )
        self.organism_manager = OrganismManager(
            self.world_size,
            ORGANISM_COUNT,
            self.environment.terrain,
            self.seed_positions,
        )
        self.logger = Logger()
        self.tick = 0
        self.enable_debug = enable_debug
    
    def update_simulation(self):
        """Update one simulation tick"""
                
        # Compute energy decay and sharing
        harvested_energy = self.organism_manager.compute_energy(self.environment.terrain)
        
        # DISABLE topology expansion (major CPU bottleneck)
        self.organism_manager.compute_topology()
        self.organism_manager.reseed_organism_positions_if_extinct()
        
        # Compute environment changes
        self.environment.compute_environment(
            self.organism_manager.topology_matrix,
            harvested_energy,
            self.organism_manager,
        )
        
        self.logger.update_fps()

        log_debug_metrics = self.enable_debug and self.tick % DEBUG_PRINT_INTERVAL == 0
        if log_debug_metrics:
            debug_info = self.logger.get_debug_info()
            total_energy = torch.sum(self.organism_manager.energy_matrix).item()
            total_terrain = torch.sum(self.environment.terrain).item()
            total_pending = torch.sum(self.organism_manager.pending_birth_energy).item()
            system_energy = total_energy + total_terrain + total_pending + self.organism_manager.destroyed_energy
            self.logger.log_tick(self.tick, 0, debug_info, None, None, total_energy, total_terrain, system_energy)

        # Clear GPU cache periodically (interactive sim only; skipped during training)
        if self.enable_debug and self.tick > 0 and self.tick % GPU_CACHE_CLEAR_INTERVAL == 0:
            gpu_handler.clear_cache()
            if self.environment.environment_type == 3:
                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        self.tick += 1

        om = self.organism_manager
        sim_data = {
            'terrain': self.environment.terrain,
            'topology': om.topology_matrix,
            'energy': om.energy_matrix,
            'new_cell_candidates': om.new_cell_candidates,
            'sharing_rate': om.sharing_rate_matrix,
            'hidden_channels': om.hidden_channels,
            'pending_birth_energy': om.pending_birth_energy,
            'destroyed_energy': om.destroyed_energy,
            'destroyed_entropy': om.destroyed_entropy,
            'entropy_produced': om.destroyed_entropy,
            'lineage_enabled': om.colony_mutation_enabled,
        }
        if log_debug_metrics:
            organism_s, terrain_s, pending_s = system_entropy_components(
                om.energy_matrix,
                self.environment.terrain,
                om.topology_matrix,
                om.pending_birth_energy,
                om.new_cell_candidates.float(),
            )
            sim_data['organism_entropy'] = organism_s
            sim_data['terrain_entropy'] = terrain_s
            sim_data['pending_entropy'] = pending_s
            sim_data['system_entropy'] = system_entropy_total(om, self.environment.terrain)
        return sim_data
    
    def reset_for_replay(self):
        """Reset simulation for replay with new CNN"""
        # Reset organism manager
        self.organism_manager = OrganismManager(
            self.world_size,
            ORGANISM_COUNT,
            self.environment.terrain,
            self.seed_positions,
        )
        # Reset tick counter
        self.tick = 0
    

def clear_saved_networks():
    """Clear all saved .pt network files from data directory"""
    pattern = 'data/cnn_*.pt'
    files = glob.glob(pattern)
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    if files:
        print(f"Cleared {len(files)} saved network files")

def apply_loaded_cnn(organism_manager, loaded_cnn):
    """Attach a trained CNN to the organism manager."""
    organism_manager.energy_distribution_cnn = loaded_cnn


def configure_organism_manager_from_args(organism_manager, args):
    """Apply --load and/or --lineage CLI options to an organism manager."""
    if args.load:
        loaded_cnn = load_latest_cnn()
        if loaded_cnn is not None:
            apply_loaded_cnn(organism_manager, loaded_cnn)
            print("Using loaded model in simulation")
        else:
            print("Failed to load model, using default CNN")
    if args.lineage:
        organism_manager.enable_colony_mutation()


def load_latest_cnn():
    """Load the latest saved CNN model and return it"""
    # Find all .pt files in data directory
    pattern = 'data/cnn_*_gen*_*.pt'
    files = glob.glob(pattern)
    
    if not files:
        print("No saved models found in data/ directory")
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Get the most recent file
    latest_file = files[0]
    
    try:
        # Create a CNN instance
        cnn = EnergyDistributionCNN(device)
        # Load the state dict
        state_dict = torch.load(latest_file, map_location=device)
        cnn.load_state_dict(state_dict)
        cnn._zero_hidden_channel_bias()
        print(f"Loaded model: {latest_file}")
        return cnn
    except Exception as e:
        print(f"Couldn't load {latest_file}: {e}")
        return None

def start_cnn_evolution(grapher: Grapher | None = None, load_latest=False, fitness_mode=None):
    """Start CNN evolution training"""
    mode = CNN_FITNESS_MODE if fitness_mode is None else fitness_mode
    set_cnn_fitness_mode(mode)
    print("Starting CNN Evolution Training...")
    print(f"Fitness mode: {cnn_fitness_mode_label(mode)} ({mode})")
    # Set multiprocessing start method for PyTorch compatibility
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn', force=True)
    evolution_driver = CNNEvolutionDriver(
        WORLD_SIZE,
        epochs=CNN_TRAINING_EPOCHS,
        max_time=CNN_TRAINING_MAX_TIME,
        fitness_mode=mode,
    )
    
    # Load latest model if requested
    if load_latest:
        if evolution_driver.ga.load_latest_model():
            # Copy loaded model to all subjects (as if it was the fittest from previous run)
            parent = evolution_driver.ga.subjects[0]
            for i in range(1, CNN_POPULATION_SIZE):
                # Copy CPPN parameters
                evolution_driver.ga.subjects[i].cppn.fc1.weight.data = parent.cppn.fc1.weight.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc1.bias.data = parent.cppn.fc1.bias.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc2.weight.data = parent.cppn.fc2.weight.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc2.bias.data = parent.cppn.fc2.bias.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc3.weight.data = parent.cppn.fc3.weight.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc3.bias.data = parent.cppn.fc3.bias.data.clone()
                
                # Regenerate CNN kernels from CPPN
                evolution_driver.ga.subjects[i].conv1.weight.data = evolution_driver.ga.subjects[i].cppn.generate_conv_weights(4, 32, 3)
                evolution_driver.ga.subjects[i].conv1.bias.data = evolution_driver.ga.subjects[i].cppn.generate_bias(32)
                evolution_driver.ga.subjects[i]._regenerate_conv2_from_cppn()
            print("Loaded latest model and initialized population from it")
    
    evolution_driver.grapher = grapher
    best_cnn = evolution_driver.run_evolution()
    
    print(f"\nTraining completed! Best CNN parameters:")
    print(f"conv1 weight shape: {best_cnn.conv1.weight.data.shape}")
    print(f"conv2 weight shape: {best_cnn.conv2.weight.data.shape}")
    
    return best_cnn
    

def main():
    """Main simulation loop with GPU-accelerated OpenGL rendering"""
    parser = argparse.ArgumentParser(
        description='Organism Evolution Simulation with CNN-based energy distribution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
COMMAND-LINE ARGUMENTS:
  --train              Run CNN training mode (no OpenGL simulation)
                       Uses genetic algorithm to evolve CNN weights for energy distribution
  --graph              Show matplotlib training graphs (default: headless when TRAIN_HEADLESS=True)
  --load               Load the fittest model from the last generation of the last run
                       Works with --train or normal simulation mode
  --lineage            Enable colony lineage mutations on offspring (CNN_MUTATION_*)
                       Interactive simulation only; use with or without --load

KEYBOARD CONTROLS (during simulation):
  ESC                  Quit the simulation
  m                    Toggle render mode (org_top / org_energy)
  n                    Toggle filters (show organisms / environment only)
  b                    Toggle organism energy view
  v                    Toggle sharing rate view (raw values / thresholded mask)
  q                    Toggle hidden channel (green) in cell state view
  h                    Toggle harvesting (enable/disable energy harvest rate)
  r                    Reload simulation (preserves environment type)
  1                    Switch to environment type 1 (energy masks)
  2                    Switch to environment type 2 (perlin pump)
  3                    Switch to environment type 3 (moving perlin noise)

CONFIGURATION PARAMETERS (from config.py):
  World Configuration:
    WORLD_SIZE              Grid size (default: 72)
    ORGANISM_COUNT          Number of starting organisms (default: 1)
  
  Environment Configuration:
    ENVIRONMENT_TYPE        Terrain type: 1=center mask, 2=perlin pump, 3=moving perlin (default: 2)
    NOISE_SCALE             Terrain noise scale (default: 0.01)
    NOISE_FREQUENCY_MULTIPLIER  Frequency multiplier for terrain generation (default: 8)
    NOISE_OCTAVES           Number of noise octaves (default: 6)
    NOISE_POWER             Power shaping for terrain (default: 2)
    PERLIN_NOISE_SCALE      Base scale for perlin noise (default: 0.05)
    PERLIN_TIME_SPEED       Speed of perlin noise animation (default: 0.005)
  
  Organism Configuration:
    ENERGY_HARVEST_RATE     Energy harvested per tick (default: 0.05)
    ENERGY_DECAY             Base energy decay per tick (default: 0.001)
    ENERGY_SHARING_RATE     Initial energy sharing rate (default: 0.5)
    REPRODUCTION_THRESHOLD  Energy threshold for cell reproduction (default: 0.1)
    DEATH_THRESHOLD         Energy threshold below which cells die (default: 0.05)
    STARTING_POSITION_TERRAIN_BOOST  Terrain energy boost at starting positions (default: 10.0)
  
  Rendering Configuration:
    RENDERING_FPS           Target rendering FPS (default: 30)
    PIXEL_SCALE_FACTOR      Pixel upscaling factor (default: calculated from WORLD_SIZE)
  
  CNN Training Configuration:
    CNN_POPULATION_SIZE     Number of CNNs in genetic algorithm population (default: 16)
    CNN_MUTATION_RATE       Probability of mutation per parameter (default: 0.01)
    CNN_MUTATION_MAGNITUDE  Magnitude of mutations (default: 0.01)
    CNN_TRAINING_EPOCHS     Number of generations to evolve (default: 100)
    CNN_TRAINING_MAX_TIME   Maximum simulation time per evaluation (default: 200)
    CNN_FITNESS_MODE        cell_count | entropy_production | persistence | life_like
    CNN_FITNESS_PERSISTENCE_TERRAIN  Uniform terrain level for persistence mode (default: 0.5)
    CNN_FITNESS_LIFE_LIKE_ORDER_WEIGHT  λ in fitness = ∫σ̇ − λ·ΔS_config (default: 1.0)
    CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR  Minimum ∫σ̇ or life_like fitness is 0 (default: 0)
  
  Performance Configuration:
    DEVICE_TYPE             Preferred device: "mps", "cuda", or "cpu" (default: "mps")
    GPU_CACHE_CLEAR_INTERVAL  Ticks between GPU cache clears (default: 10)

For more details, see config.py
        '''
    )
    parser.add_argument('--train', action='store_true', help='Run CNN training mode (no OpenGL sim)')
    parser.add_argument('--graph', action='store_true', help='Show matplotlib graphs during --train')
    parser.add_argument('--load', action='store_true', help='Load the fittest model from the last generation of the last run (works with --train or normal sim)')
    parser.add_argument(
        '--lineage',
        action='store_true',
        help='Enable colony lineage differential mutations on offspring (CNN_MUTATION_*; interactive sim)',
    )
    parser.add_argument(
        '--fitness-mode',
        choices=CNN_FITNESS_MODES,
        default=None,
        help='Evolution objective: cell_count (growth), entropy_production (dissipation), persistence (survival on uniform terrain)',
    )
    args = parser.parse_args()

    if args.train:
        # Clear saved networks at the beginning of training (only if not loading)
        if not args.load:
            clear_saved_networks()
        
        use_graph = (not TRAIN_HEADLESS) or args.graph
        if TRAIN_HEADLESS and not args.graph:
            print('Headless training mode (no matplotlib graphs)')
        elif args.graph:
            print('Training with matplotlib graphs')
        grapher = Grapher() if use_graph else None
        start_cnn_evolution(grapher, load_latest=args.load, fitness_mode=args.fitness_mode)
        return
    # Initialize OpenGL/GLUT
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
    left_margin = 300
    top_margin = 10
    bottom_margin = 10
    right_margin = 10
    render_size = int(WORLD_SIZE * PIXEL_SCALE_FACTOR)
    window_width = render_size + left_margin + right_margin
    window_height = render_size + top_margin + bottom_margin
    glut.glutInitWindowSize(window_width, window_height)
    glut.glutCreateWindow(b"Organism Simulation - OpenGL GPU Accelerated")
    
    # Setup OpenGL
    glEnable(GL_TEXTURE_2D)
    glClearColor(OPENGL_CLEAR_COLOR_R, OPENGL_CLEAR_COLOR_G, OPENGL_CLEAR_COLOR_B, OPENGL_CLEAR_COLOR_A)
    
    # Set up viewport and projection for full-screen rendering
    glViewport(0, 0, window_width, window_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    simulation = Simulation()
    configure_organism_manager_from_args(simulation.organism_manager, args)
    
    renderer = Renderer(WORLD_SIZE)
    renderer.top_margin = top_margin
    renderer.bottom_margin = bottom_margin
    renderer.right_margin = right_margin
    input_handler = InputHandler(renderer)
    
    # Initialize OpenGL components after context is created
    renderer._setup_opengl()
    renderer.opengl_initialized = True
    
    # Print controls
    input_handler.print_controls()
    
    # Rendering frequency - simulation runs at MAXIMUM SPEED
    rendering_fps = RENDERING_FPS
    # Calculate rendering frequency based on target FPS (simulation runs at full speed)
    rendering_frequency = max(1, RENDERING_BASE_FPS // RENDERING_FPS)  # Calculate rendering frequency
    
    # Frame rate control variables
    last_frame_time = time.time()
    
    # Global variables for OpenGL callbacks
    global current_simulation, current_renderer, current_input_handler, main_args
    current_simulation = simulation
    current_renderer = renderer
    current_input_handler = input_handler
    main_args = args  # Store args for keyboard callback
    
    def display():
        """OpenGL display callback - only handles rendering when window needs redraw"""
        global current_renderer, current_harvest_rate, replay_mode, current_best_cnn
        
        # Only render if we have data
        if last_sim_data is not None:
            # Update OpenGL texture and render
            current_renderer.update_texture(current_renderer.last_image)
            # Get logger from simulation for FPS display
            logger = current_simulation.logger if hasattr(current_simulation, 'logger') else None
            overlay_data = dict(last_sim_data)
            overlay_data["debug_panel_mode"] = current_renderer.debug_panel_mode
            if (
                current_renderer.genome_view_enabled
                and overlay_data.get("lineage_enabled")
                and current_renderer.debug_panel_mode == "genomes"
            ):
                graph = current_simulation.organism_manager.colony_mutation_graph
                genome_count, panel_lines = graph.genome_panel_lines(
                    overlay_data["topology"]
                )
                overlay_data["genome_count"] = genome_count
                overlay_data["genome_panel_lines"] = panel_lines
            current_renderer.render_opengl(
                overlay_data, current_harvest_rate, replay_mode, current_best_cnn, logger
            )
    
    def keyboard(key, x, y):
        """OpenGL keyboard callback"""
        global current_renderer, current_simulation, main_args, current_harvest_rate
        
        if key == 27:  # ESC
            glut.glutLeaveMainLoop()
        elif key == b'q':
            current_renderer.toggle_hidden_channel_0_view()
        elif key == b'm':
            current_renderer.toggle_render_mode()
        elif key == b'n':
            current_renderer.toggle_filters()
        elif key == b'b':
            current_renderer.toggle_org_energy_view()
        elif key == b'v':
            current_renderer.toggle_sharing_rate_view()
        elif key == b'\t':
            if not current_simulation.organism_manager.colony_mutation_enabled:
                print("Genome view requires --lineage")
            else:
                current_renderer.toggle_genome_view()
        elif key == b'h':
            # Toggle harvest rate between 0 and original value
            if current_harvest_rate == 0:
                current_harvest_rate = ENERGY_HARVEST_RATE
                print(f"Harvest rate enabled: {ENERGY_HARVEST_RATE}")
            else:
                current_harvest_rate = 0
                print("Harvest rate disabled: 0")
        elif key == b'r':
            # Reload simulation (preserve environment type)
            print("Reloading simulation...")
            current_env_type = current_simulation.environment.environment_type
            current_simulation = Simulation()
            # Restore environment type
            current_simulation.environment.environment_type = current_env_type
            current_simulation.environment.terrain = current_simulation.environment.generate_terrain()
            configure_organism_manager_from_args(current_simulation.organism_manager, main_args)
            print(f"Simulation reloaded (environment type: {current_env_type})")
        elif key == b'1':
            # Switch to environment type 1 (energy masks)
            print("Switching to environment type 1 (energy masks)")
            import config
            config.ENVIRONMENT_TYPE = 1
            current_simulation.environment.environment_type = 1
            current_simulation.environment.terrain = current_simulation.environment._generate_energy_mask_terrain()
        elif key == b'2':
            print("Switching to environment type 2 (perlin pump)")
            import config
            config.ENVIRONMENT_TYPE = 2
            current_simulation.environment.environment_type = 2
            current_simulation.environment.time = 0.0
            current_simulation.environment.pump_time = 0.0
            current_simulation.environment._reset_pump_controller()
            current_simulation.environment.terrain = current_simulation.environment.generate_terrain()
        elif key == b'3':
            # Switch to environment type 3 (moving perlin noise)
            print("Switching to environment type 3 (moving perlin noise)")
            import config
            config.ENVIRONMENT_TYPE = 3
            current_simulation.environment.environment_type = 3
            current_simulation.environment.time = 0.0
            current_simulation.environment.terrain = current_simulation.environment._generate_perlin_terrain()
    
    # Frame rate control - ONLY for rendering/OpenGL
    rendering_frame_counter = 0
    
    # Store last simulation data
    last_sim_data = None
    
    def idle():
        """OpenGL idle callback with proper frame rate limiting"""
        nonlocal rendering_frame_counter, last_sim_data, last_frame_time
        global current_simulation, current_renderer, current_best_cnn, replay_mode
        
        # Calculate time since last frame
        current_time = time.time()
        delta_time = current_time - last_frame_time
        
        # Update simulation at FULL SPEED - no frame limiting
        # Use replay simulation if in replay mode and it exists
        if replay_mode and hasattr(current_simulation, 'replay_simulation') and current_simulation.replay_simulation is not None:
            last_sim_data = current_simulation.replay_simulation.update_simulation()
        else:
            last_sim_data = current_simulation.update_simulation()
        
        # Render at limited frequency using last simulation data
        rendering_frame_counter += 1
        if rendering_frame_counter >= rendering_frequency and last_sim_data is not None:
            # Render using last simulation data
            # Create mask based on render mode
            if current_renderer.render_mode == "org_top":
                # Single channel mask for topology (not used in topology mode)
                mask = torch.zeros((WORLD_SIZE, WORLD_SIZE), device=device)
            else:
                # Single channel mask for energy
                mask = torch.clamp(last_sim_data['energy'], 0, 1)
            
            use_genome_view = (
                last_sim_data.get("lineage_enabled", False)
                and current_renderer.genome_view_enabled
            )
            genome_colors = None
            if use_genome_view:
                graph = current_simulation.organism_manager.colony_mutation_graph
                genome_colors = graph.genome_color_field(last_sim_data['topology'])
            image_tensor = current_renderer.render(
                last_sim_data['terrain'], 
                last_sim_data['topology'], 
                mask,
                last_sim_data['new_cell_candidates'],
                last_sim_data.get('sharing_rate', None),
                last_sim_data.get('hidden_channels', None),
                genome_colors,
            )
            
            # Store the rendered image for display() to use
            current_renderer.last_image = image_tensor
            
            # Trigger display update
            glut.glutPostRedisplay()
            
            rendering_frame_counter = 0
        
        # No frame rate limiting - simulation runs at MAXIMUM SPEED
        last_frame_time = time.time()
    
    # Set OpenGL callbacks
    glut.glutDisplayFunc(display)
    glut.glutKeyboardFunc(keyboard)
    glut.glutIdleFunc(idle)
    
    try:
        # Start OpenGL main loop
        glut.glutMainLoop()
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by Ctrl+C")
    
    finally:
        print("Simulation ended")

if __name__ == "__main__":
    main()
