import numpy as np
import torch
import time
from noise import pnoise2
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

# Initialize GPU handler
gpu_handler = GPUHandler()
device = gpu_handler.get_device()

# Global harvest rate (can be modified at runtime)
current_harvest_rate = ENERGY_HARVEST_RATE

# Global best CNN for replay
current_best_cnn = None
replay_mode = False

class EnergyDistributionCNN(torch.nn.Module):
    """CNN that outputs 3x3 distribution proportions for each source cell"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Conv layers for processing input channels directly
        # Input: 6 channels (shareable_energy + terrain + sharing_rate + 3 hidden channels)
        # Output: 13 channels (9 for 3x3 distribution matrix + 1 for sharing_rate + 3 for hidden channels)
        # conv1 now processes rotated 3x3 patches with stride=3 (skips 3 cells)
        self.conv1 = torch.nn.Conv2d(6, 32, kernel_size=3, stride=3, padding=0, device=device)
        self.conv2 = torch.nn.Conv2d(32, 13, kernel_size=1, device=device) 
        
        # Pre-compute rotation permutation lookup table on GPU (90° rotations only)
        # Each row contains the permutation indices for that rotation direction
        # For 3x3 grid: 0 1 2 / 3 4 5 / 6 7 8
        rotations = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 0° (no rotation)
            [6, 3, 0, 7, 4, 1, 8, 5, 2],  # 90° CCW
            [8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180°
            [2, 5, 8, 1, 4, 7, 0, 3, 6],  # 270° CCW (90° CW)
        ]
        self.rotation_lookup = torch.tensor(rotations, dtype=torch.long, device=device)  # (4, 9)
        
        # Inverse rotation lookup for rotating output back to world coordinates
        # Inverse of: 0°→0°, 90°→270°, 180°→180°, 270°→90°
        inverse_rotations = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],  # 0° inverse (0°)
            [2, 5, 8, 1, 4, 7, 0, 3, 6],  # 90° inverse (270°)
            [8, 7, 6, 5, 4, 3, 2, 1, 0],  # 180° inverse (180°)
            [6, 3, 0, 7, 4, 1, 8, 5, 2],  # 270° inverse (90°)
        ]
        self.inverse_rotation_lookup = torch.tensor(inverse_rotations, dtype=torch.long, device=device)  # (4, 9)
        
        # Move model to device
        self.to(device)

    def _discretize_rotations_batch(self, angles):
        """Convert continuous angles to discrete 90° directions (0-3) for all patches at once"""
        # Normalize angles to [0, 2π)
        angles = angles % (2 * torch.pi)
        # Convert to 4 directions: 0=0°, 1=90°, 2=180°, 3=270°
        directions = (angles / (2 * torch.pi) * 4).long() % 4
        return directions
    
    def forward(self, shareable_energy, terrain, sharing_rate, hidden_channels, rotation_matrix):
        """
        Input: 
            shareable_energy of shape (world_size, world_size)
            terrain of shape (world_size, world_size)
            sharing_rate of shape (world_size, world_size)
            hidden_channels of shape (3, world_size, world_size)
            rotation_matrix of shape (world_size, world_size) - rotation angles for each cell
        Output: 
            proportions of shape (3, 3, world_size, world_size)
            sharing_rate_output of shape (world_size, world_size)
            hidden_channels_output of shape (3, world_size, world_size)
        """
        world_size = shareable_energy.shape[0]
        
        # Stack input channels: (6, H, W)
        input_channels = torch.cat([
            shareable_energy.unsqueeze(0),
            terrain.unsqueeze(0),
            sharing_rate.unsqueeze(0),
            hidden_channels
        ], dim=0)  # (6, H, W)
        
        # Extract 3x3 patches for each cell with padding
        padded = torch.nn.functional.pad(input_channels, (1, 1, 1, 1), mode='circular')  # (6, H+2, W+2)
        
        # Extract patches: (6, H, W, 3, 3)
        patches = padded.unfold(1, 3, 1).unfold(2, 3, 1)  # (6, H, W, 3, 3)
        patches = patches.permute(1, 2, 0, 3, 4)  # (H, W, 6, 3, 3)
        
        # Reshape to batch: (H*W, 6, 3, 3)
        patches_flat = patches.reshape(world_size * world_size, 6, 3, 3)
        rotation_flat = rotation_matrix.reshape(world_size * world_size)
        
        # Discretize all rotation angles to directions at once (GPU operation)
        directions = self._discretize_rotations_batch(rotation_flat)  # (H*W,)
        
        # Flatten patches to (H*W, 6, 9) for gather operation
        patches_flat_2d = patches_flat.view(world_size * world_size, 6, 9)  # (H*W, 6, 9)
        
        # Get permutation indices for each patch: (H*W, 9)
        perm_indices = self.rotation_lookup[directions]  # (H*W, 9)
        
        # Expand perm_indices to match channel dimension: (H*W, 6, 9)
        perm_indices_expanded = perm_indices.unsqueeze(1).expand(-1, 6, -1)  # (H*W, 6, 9)
        
        # Gather rotated patches along the last dimension (position indices)
        # This applies the permutation to all patches in parallel
        rotated_flat = torch.gather(patches_flat_2d, 2, perm_indices_expanded)  # (H*W, 6, 9)
        
        # Reshape back to (H*W, 6, 3, 3)
        batch_patches = rotated_flat.view(world_size * world_size, 6, 3, 3)
        
        # Process through conv1 with stride=3: (H*W, 16, 1, 1)
        x = self.conv1(batch_patches)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)  # (H*W, 13, 1, 1)
        
        # Reshape back to spatial dimensions: (H, W, 13)
        x = x.squeeze(-1).squeeze(-1)  # (H*W, 13)
        x = x.view(world_size, world_size, 13)  # (H, W, 13)
        x = x.permute(2, 0, 1)  # (13, H, W)
        
        # Split output: 9 channels for proportions, 1 channel for sharing_rate, 3 channels for hidden
        proportions_flat = x[:9]  # (9, H, W)
        sharing_rate_output = x[9:10].squeeze(0)  # (H, W)
        hidden_channels_output = x[10:13]  # (3, H, W)
        
        proportions_flat = torch.nn.functional.softmax(proportions_flat, dim=0)  # (9, H, W)
        
        # Reshape to (3, 3, H, W) - each cell has its 3x3 distribution matrix
        proportions = proportions_flat.view(3, 3, world_size, world_size)  # (3, 3, H, W)
        
        # Rotate output back to world coordinates
        # Reshape proportions to (H*W, 3, 3) for batch rotation
        proportions_flat_batch = proportions.permute(2, 3, 0, 1).reshape(world_size * world_size, 9)  # (H*W, 9)
        
        # Get inverse permutation indices for each cell: (H*W, 9)
        inverse_perm_indices = self.inverse_rotation_lookup[directions]  # (H*W, 9)
        
        # Apply inverse rotation to rotate output back to world coordinates
        proportions_rotated_back = torch.gather(proportions_flat_batch, 1, inverse_perm_indices)  # (H*W, 9)
        
        # Reshape back to (3, 3, H, W)
        proportions = proportions_rotated_back.view(world_size, world_size, 9).permute(2, 0, 1).view(3, 3, world_size, world_size)  # (3, 3, H, W)
        
        # Clamp sharing_rate_output to valid range
        sharing_rate_output = torch.tanh(sharing_rate_output)  # (H, W)
        
        # Clamp hidden_channels_output to valid range
        hidden_channels_output = torch.sigmoid(hidden_channels_output)  # (3, H, W)
        
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
        """Copy parent weights to all subjects"""
        for i in range(self.pop_size):
            if i != self.fittest_index:  # Don't overwrite the parent
                # Copy conv layer parameters
                self.subjects[i].conv1.weight.data = parent.conv1.weight.data.clone()
                self.subjects[i].conv1.bias.data = parent.conv1.bias.data.clone()
                self.subjects[i].conv2.weight.data = parent.conv2.weight.data.clone()
                self.subjects[i].conv2.bias.data = parent.conv2.bias.data.clone()
                
    def mutate(self):
        """Apply mutations to all subjects except the fittest"""
        for i in range(self.pop_size):
            if i == self.fittest_index:  # Don't mutate the fittest
                continue
            
            # Mutate conv1 parameters
            mutation_mask = torch.rand_like(self.subjects[i].conv1.weight) < self.mut_rate
            mutations = (torch.rand_like(self.subjects[i].conv1.weight) - 0.5) * 2 * self.mut_mag
            self.subjects[i].conv1.weight.data[mutation_mask] += mutations[mutation_mask]
            
            mutation_mask = torch.rand_like(self.subjects[i].conv1.bias) < self.mut_rate
            mutations = (torch.rand_like(self.subjects[i].conv1.bias) - 0.5) * 2 * self.mut_mag
            self.subjects[i].conv1.bias.data[mutation_mask] += mutations[mutation_mask]
            
            # Mutate conv2 parameters
            mutation_mask = torch.rand_like(self.subjects[i].conv2.weight) < self.mut_rate
            mutations = (torch.rand_like(self.subjects[i].conv2.weight) - 0.5) * 2 * self.mut_mag
            self.subjects[i].conv2.weight.data[mutation_mask] += mutations[mutation_mask]
            
            mutation_mask = torch.rand_like(self.subjects[i].conv2.bias) < self.mut_rate
            mutations = (torch.rand_like(self.subjects[i].conv2.bias) - 0.5) * 2 * self.mut_mag
            self.subjects[i].conv2.bias.data[mutation_mask] += mutations[mutation_mask]
                
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
    if device_str.startswith('cuda'):
        _worker_device = torch.device(device_str)
    elif device_str == 'mps':
        _worker_device = torch.device('mps')
    else:
        _worker_device = torch.device('cpu')
    
    # Set global device for this worker process (used by Simulation and related classes)
    import sys
    current_module = sys.modules[__name__]
    current_module.device = _worker_device

def _evaluate_cnn_worker(args):
    """Worker function for multiprocessing CNN evaluation"""
    cnn_state_dict, world_size, max_time, bot_index, collect_tick_data = args
    
    # Use the device that was set during worker initialization
    global _worker_device
    
    # Create CNN and load state dict
    cnn = EnergyDistributionCNN(_worker_device)
    cnn.load_state_dict(cnn_state_dict)
    
    # Create simulation instance
    sim = Simulation(enable_debug=False)
    sim.organism_manager.energy_distribution_cnn = cnn
    
    # Evaluate CNN
    total_cell_count = 0.0
    tick_data = []
    for t in range(max_time):
        sim.update_simulation()
        current_cell_count = torch.sum(sim.organism_manager.topology_matrix).item()
        total_cell_count += current_cell_count
        
        # Collect tick data every 10 ticks if requested
        if collect_tick_data and t % 10 == 0:
            org_energy = torch.sum(sim.organism_manager.energy_matrix).item()
            env_energy = torch.sum(sim.environment.terrain).item()
            total = org_energy + env_energy
            tick_data.append((t, total_cell_count, current_cell_count, org_energy, env_energy, total))
        
        if current_cell_count == 0:
            break
    
    # Explicitly clean up GPU memory
    del cnn
    del sim
    gc.collect()
    if _worker_device.type == 'mps':
        torch.mps.empty_cache()
    elif _worker_device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return (total_cell_count, tick_data)


class CNNEvaluator:
    """CNN evaluator for GPU-accelerated fitness computation"""
    def __init__(self, world_size, max_time, device):
        self.world_size = world_size
        self.max_time = max_time
        self.device = device
        self.grapher = None
        self.current_generation_max_fitness = 0.0
        self.pool = None
        
    def _ensure_pool(self):
        """Create pool if it doesn't exist"""
        if self.pool is None:
            device_str = str(self.device)
            self.pool = multiprocessing.Pool(initializer=_init_worker, initargs=(device_str,))
    
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
            args_list.append((cpu_state_dict, self.world_size, self.max_time, i, collect_tick_data))
        
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
        gc.collect()
        
        # Clear GPU cache in main process after evaluation
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Reset pool to drop any lingering worker caches before next call
        self.close_pool()
        
        return fitness_scores
    
    def _evaluate_single_cnn(self, simulation, bot_index: int | None = None):
        """Evaluate a single CNN simulation"""
        total_cell_count = 0.0
        
        # Run simulation for max_time steps
        for t in range(self.max_time):
            sim_data = simulation.update_simulation()
            
            # Calculate fitness based on total cell count
            current_cell_count = torch.sum(simulation.organism_manager.topology_matrix).item()
            total_cell_count += current_cell_count

            # Tick graph update every 10 ticks if grapher attached
            if self.grapher is not None and t % 10 == 0:
                org_energy = torch.sum(simulation.organism_manager.energy_matrix).item()
                env_energy = torch.sum(simulation.environment.terrain).item()
                total = org_energy + env_energy
                # enqueue only; GUI update happens in main thread
                # For fitness series, use cumulative fitness (total accumulated cell count so far)
                self.grapher.enqueue_tick(t, self.current_generation_max_fitness, [total_cell_count], org_energy, env_energy, total)
                if bot_index is not None:
                    # For per-bot fitness series, use cumulative fitness
                    # Pass current_cell_count for org_energy parameter so organism graph shows cell count
                    self.grapher.enqueue_bot_tick(bot_index, t, total_cell_count, current_cell_count, env_energy, total)
            
            # Early termination if cell count drops to zero
            if current_cell_count == 0:
                break
                
        # Fitness is total accumulated cell count over time (rewards high cell count sustained longer)
        fitness = total_cell_count
        return fitness


class CNNEvolutionDriver:
    """Evolution driver for CNN training with GPU evaluation and replay"""
    def __init__(self, world_size, epochs=100, max_time=100):
        self.world_size = world_size
        self.epochs = epochs
        self.max_time = max_time
        
        # Genetic algorithm parameters
        self.ga = CNNGeneticAlgorithm(CNN_POPULATION_SIZE, CNN_MUTATION_RATE, CNN_MUTATION_MAGNITUDE, device)
        
        # Evaluator
        self.evaluator = CNNEvaluator(world_size, max_time, device)
        self.grapher = None
        
        # Replay simulation for showing best organism
        self.replay_simulation = None
        
    def evaluate_cnn(self, cnn, simulation):
        """Evaluate a CNN by running simulation and measuring fitness"""
        # Create a copy of simulation with this CNN (debug disabled during training)
        test_sim = Simulation(enable_debug=False)
        test_sim.organism_manager.energy_distribution_cnn = cnn
        
        total_cell_count = 0.0
        
        # Run simulation for max_time steps
        for t in range(self.max_time):
            sim_data = test_sim.update_simulation()
            
            # Calculate fitness based on total cell count
            current_cell_count = torch.sum(test_sim.organism_manager.topology_matrix).item()
            total_cell_count += current_cell_count
            
            # Early termination if cell count drops to zero
            if current_cell_count == 0:
                break
                
        # Fitness is total accumulated cell count over time
        fitness = total_cell_count
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
        
        print(f"Created replay simulation with best CNN (fitness/cell count: {self.ga.fitness_scores[self.ga.fittest_index]:.6f})")
        print("Press 'r' to toggle replay mode and see the best organism in action!")
        
    def run_evolution(self):
        """Run the evolution process with multiprocessing GPU evaluation"""
        print(f"\nStarting CNN Evolution - {self.epochs} generations")
        print(f"Population size: {CNN_POPULATION_SIZE}")
        print(f"Using multiprocessing evaluation with {multiprocessing.cpu_count()} processes")
        
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
                for i, fitness in enumerate(fitness_scores):
                    print(f"CNN {i}: fitness (cell count) = {fitness:.6f}")
                    
                # Run one generation
                self.ga.compute_generation()
                
                # Print summary
                best_fitness = self.ga.fitness_scores[self.ga.fittest_index]
                print(f"Best fitness (cell count): {best_fitness:.6f}")
                print(f"Best CNN conv1 weight shape: {self.ga.subjects[self.ga.fittest_index].conv1.weight.data.shape}")
                print(f"Best CNN conv2 weight shape: {self.ga.subjects[self.ga.fittest_index].conv2.weight.data.shape}")
                
                # Save the fittest network every generation
                self.ga.save_model(self.ga.fittest_index, generation=gen + 1)
                
                # Create replay simulation with best organism
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
    def __init__(self, world_size, noise_scale, quantization_step):
        self.world_size = world_size
        self.noise_scale = noise_scale
        self.quantization_step = quantization_step
        self.terrain = self.generate_terrain()
    
    def generate_terrain(self):
        """Generate 2D sine-based terrain"""
        x = torch.arange(self.world_size, dtype=torch.float32, device=device)
        y = torch.arange(self.world_size, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        freq = self.noise_scale * NOISE_FREQUENCY_MULTIPLIER * (2.0 * np.pi)
        cx = self.world_size // 2
        cy = self.world_size // 2
        # Exponential radial decay from center to dampen amplitude
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = torch.tensor(self.world_size / 2, dtype=torch.float32, device=device)
        radial_decay = torch.exp(-dist / max_dist)
        # Use cosine so that values are 1 at center (cos(0) = 1)
        values = torch.cos((xx - cx) * freq) * torch.cos((yy - cy) * freq)
        # Normalize to 0-1 (cosine ranges from -1 to 1)
        values = (values + 1.0) * 0.5
        # Shape distribution using power
        values = torch.pow(values, NOISE_POWER)
        # Apply radial decay
        values = values * radial_decay
        
        # Add 4 more energy sources 3px away from center in cardinal directions
        offset = 1
        sources = [
            (cx, cy - offset),  # up
            (cx, cy + offset),  # down
            (cx - offset, cy),   # left
            (cx + offset, cy)   # right
        ]
        
        for sx, sy in sources:
            source_values = torch.cos((xx - sx) * freq) * torch.cos((yy - sy) * freq)
            source_values = (source_values + 1.0) * 0.5
            # Apply same shaping and radial decay to source peaks
            source_values = torch.pow(source_values, NOISE_POWER) * (radial_decay**2)
            values = torch.maximum(values, source_values)
        
        # Apply dead mask similar to previous behavior
        dead_mask = values > ENV_NOISE_THRESHOLD
        values = values * dead_mask
        return torch.clamp(values, 0, 1)
    
    def compute_environment(self, topology_matrix, harvested_energy):
        """Modify environment based on organism presence"""
        # Deplete terrain energy by the amount harvested
        self.terrain = torch.clamp(self.terrain - (harvested_energy * 0.1 * topology_matrix), 0, 1)
        
        # # Apply terrain boost at organism starting positions
        # positions = torch.tensor(ORGANISM_POSITIONS, dtype=torch.long, device=device)
        # if positions.numel() > 0:
        #     # Create 4 additional sources 3px away from each original position
        #     additional_positions = []
        #     for pos in positions:
        #         x, y = pos[0].item(), pos[1].item()
        #         offsets = [(-5, 0), (5, 0), (0, -5), (0, 5)]
        #         for dx, dy in offsets:
        #             new_x, new_y = x + dx, y + dy
        #             if 0 <= new_x < self.world_size and 0 <= new_y < self.world_size:
        #                 additional_positions.append([new_x, new_y])
            
        #     if additional_positions:
        #         additional_positions_tensor = torch.tensor(additional_positions, dtype=torch.long, device=device)
        #         all_positions = torch.cat([positions, additional_positions_tensor], dim=0)
        #     else:
        #         all_positions = positions
            
        #     y_coords, x_coords = all_positions[:, 1], all_positions[:, 0]
        #     energy_radius = 1
        #     circle_mask = torch.zeros((self.world_size, self.world_size), device=device)
        #     circle_mask[y_coords, x_coords] = True
        #     circle_mask = circle_mask.unsqueeze(0).unsqueeze(0)
        #     circle_mask = torch.nn.functional.conv2d(circle_mask, torch.ones((1, 1, 2 * energy_radius + 1, 2 * energy_radius + 1), device=device), padding=energy_radius)
        #     circle_mask = circle_mask.squeeze(0).squeeze(0)
        #     circle_mask = circle_mask > 0
        #     self.terrain[circle_mask] = self.terrain[circle_mask] + STARTING_POSITION_TERRAIN_BOOST
            

class OrganismManager:
    def __init__(self, world_size, organism_count, terrain):
        self.world_size = world_size
        self.terrain = terrain

        self.positions = torch.tensor(ORGANISM_POSITIONS, dtype=torch.long, device=device)
        self.topology_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.energy_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.sharing_rate_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.hidden_channels = torch.zeros((3, world_size, world_size), dtype=torch.float32, device=device)
        self.rotation_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.new_cell_candidates = torch.zeros((world_size, world_size), dtype=torch.bool, device=device)
        self._initialize_topology()
        
        # Reproduction parameters
        self.reproduction_threshold = REPRODUCTION_THRESHOLD
        
        # Energy distribution CNN
        self.energy_distribution_cnn = EnergyDistributionCNN(device)
    
    def _initialize_topology(self):
        """Initialize topology and energy with organism positions"""
        if self.positions.numel() > 0:
            y_coords, x_coords = self.positions[:, 1], self.positions[:, 0]
            self.topology_matrix[y_coords, x_coords] = 1
            self.energy_matrix[y_coords, x_coords] = 1
            self.sharing_rate_matrix[y_coords, x_coords] = ENERGY_SHARING_RATE
            self.hidden_channels[:, y_coords, x_coords] = 0
            self.rotation_matrix[y_coords, x_coords] = 0
    
    def compute_topology(self):
        """Reproduce using new_cell_candidates mask from energy sharing"""
        # Use new_cell_candidates mask for reproduction instead of random selection
        if not torch.any(self.new_cell_candidates):
            return
        
        # Check energy threshold on the new cell candidates
        energy_mask = (self.energy_matrix >= self.reproduction_threshold) * self.new_cell_candidates
        
        # Calculate rotation for new cells based on parent energy contributions
        if torch.any(energy_mask) and self.new_cell_contributions is not None:
            offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),  (0, 0),  (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
            
            y_coords = torch.arange(self.world_size, device=device).unsqueeze(1).expand(-1, self.world_size)
            x_coords = torch.arange(self.world_size, device=device).unsqueeze(0).expand(self.world_size, -1)
            
            # Calculate weighted direction vectors for new cells
            weighted_dx = torch.zeros_like(self.energy_matrix)
            weighted_dy = torch.zeros_like(self.energy_matrix)
            total_weight = torch.zeros_like(self.energy_matrix)
            
            for idx, (dy, dx) in enumerate(offsets):
                ci, cj = idx // 3, idx % 3
                # For each new cell at (y, x), contributions[ci, cj][y, x] is the energy
                # that arrived at (y, x) from parent at (y-dy, x-dx)
                # Get contributions that went to new cell candidates
                contrib = self.new_cell_contributions[ci, cj] * energy_mask
                
                # Weight by contribution amount, direction is (dx, dy) - away from parent
                weighted_dx += contrib * dx
                weighted_dy += contrib * dy
                total_weight += contrib
            
            # Calculate rotation angle (atan2 gives angle from positive x-axis)
            # Only calculate where there are new cells and non-zero weights
            rotation_mask = energy_mask & (total_weight > 0)
            rotation_angles = torch.atan2(weighted_dy, weighted_dx)
            self.rotation_matrix[rotation_mask] = rotation_angles[rotation_mask]
        
        # Add selected positions to topology
        self.topology_matrix[energy_mask] = 1
        
        # Initialize sharing_rate and hidden_channels for new cells
        self.sharing_rate_matrix[energy_mask] = ENERGY_SHARING_RATE
        self.hidden_channels[:, energy_mask] = 0
    
    def _apply_harvest_and_decay(self, terrain):
        """Harvest energy from terrain and apply decay"""
        if terrain is not None:
            self.terrain = terrain
                
        # Remove organisms with energy below threshold
        low_energy_mask = self.energy_matrix < DEATH_THRESHOLD
        self.topology_matrix[low_energy_mask] = 0
        self.sharing_rate_matrix[low_energy_mask] = 0
        self.hidden_channels[:, low_energy_mask] = 0
        self.rotation_matrix[low_energy_mask] = 0
        
        # Harvest energy from terrain
        harvested_energy = torch.minimum(self.terrain, self.sharing_rate_matrix * ENERGY_HARVEST_RATE)
        
        # Calculate average energy in 3x3 neighborhood for decay scaling
        energy_unsqueezed = self.energy_matrix.unsqueeze(0).unsqueeze(0) + terrain.unsqueeze(0).unsqueeze(0)
        average_energy = torch.nn.functional.conv2d(energy_unsqueezed, torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32), padding=1).squeeze(0).squeeze(0) / 9
    
        # Apply decay with factor proportional to cell's distribution rate and number of 0.0 cells in neighborhood
        decay_amount = self.sharing_rate_matrix**2 * (1-average_energy) * 0.1
        # Apply decay and harvest
        self.energy_matrix = torch.clamp((self.energy_matrix + harvested_energy - decay_amount - ENERGY_DECAY) * self.topology_matrix, 0, 1)

        return harvested_energy
    
    def _compute_energy_contributions(self, shareable_energy, proportions):
        """Compute energy contributions from each source to each neighbor"""
        # (H, W) * (3, 3, H, W) -> (3, 3, H, W)
        contributions = shareable_energy.unsqueeze(0).unsqueeze(0) * proportions
        return contributions
    
    def _accumulate_contributions(self, contributions, shareable_energy):
        """Accumulate contributions to destination cells"""
        # Subtract shareable_energy from sources first (since it's being redistributed)
        # Then add contributions which redistribute that energy (including center self-transfer)
        new_energy_matrix = self.energy_matrix - shareable_energy
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),  (0, 0),  (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
        
        # Use coordinate-based indexing for correct mapping with wrapping
        y_coords = torch.arange(self.world_size, device=device).unsqueeze(1).expand(-1, self.world_size)
        x_coords = torch.arange(self.world_size, device=device).unsqueeze(0).expand(self.world_size, -1)
        for idx, (dy, dx) in enumerate(offsets):
            ci, cj = idx // 3, idx % 3
            # For each destination (y, x), get contribution from source at (y-dy, x-dx) with wrapping
            y_src = (y_coords - dy) % self.world_size
            x_src = (x_coords - dx) % self.world_size
            new_energy_matrix += contributions[ci, cj][y_src, x_src]
        
        return new_energy_matrix
    
    def _apply_capacity_constraints(self, new_energy_matrix, receiving_mask):
        """Apply capacity constraints to limit received energy"""
        capacity = (1.0 - self.energy_matrix) * receiving_mask
        energy_incoming = new_energy_matrix - self.energy_matrix
        actual_received = torch.min(energy_incoming, capacity)
        new_energy_matrix = self.energy_matrix + actual_received
        return new_energy_matrix, actual_received, energy_incoming
    
    def _compute_source_removed(self, contributions, dest_efficiency, receiving_mask):
        """Compute how much energy each source should lose based on what was received"""
        source_removed = torch.zeros_like(contributions[0, 0])
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),  (0, 0),  (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
        
        for idx, (dy, dx) in enumerate(offsets):
            ci, cj = idx // 3, idx % 3
            y_coords = torch.arange(self.world_size, device=device)
            x_coords = torch.arange(self.world_size, device=device)
            y_dest = (y_coords.unsqueeze(1) + dy) % self.world_size
            x_dest = (x_coords.unsqueeze(0) + dx) % self.world_size
            
            dest_eff = dest_efficiency[y_dest, x_dest]
            contrib_amount = contributions[ci, cj]
            source_removed += contrib_amount * dest_eff
        
        return source_removed
    
    def compute_energy(self, terrain):
        """Main energy computation: harvest, decay, and sharing"""
        harvested_energy = self._apply_harvest_and_decay(terrain)
        
        # Preserve source energy for reads (avoid race conditions)
        source_energy = self.energy_matrix.clone()
        
        # Calculate shareable energy using per-cell sharing rates (only organisms can share)
        shareable_energy = source_energy * self.topology_matrix * self.sharing_rate_matrix
        
        # Get 3x3 proportions, sharing_rate output, and hidden_channels output from CNN
        proportions, sharing_rate_output, hidden_channels_output = self.energy_distribution_cnn(shareable_energy, terrain, self.sharing_rate_matrix, self.hidden_channels, self.rotation_matrix)
        
        # Update sharing_rate_matrix with CNN output (only for cells that exist)
        self.sharing_rate_matrix += self.sharing_rate_matrix * (1 - self.topology_matrix) + sharing_rate_output/50 * self.topology_matrix
        self.sharing_rate_matrix = torch.clamp(self.sharing_rate_matrix, 0.2, 0.999)
        
        # Update hidden_channels with CNN output (only for cells that exist)
        topology_mask = self.topology_matrix.unsqueeze(0)  # (1, H, W)
        self.hidden_channels = self.hidden_channels * (1 - topology_mask) + hidden_channels_output * topology_mask
        
        # Compute contributions
        contributions = self._compute_energy_contributions(shareable_energy, proportions)
        
        # Calculate new cell candidates - shift contributions to compute received energy at each destination
        distributed_total = torch.zeros_like(self.energy_matrix)
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),  (0, 0),  (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
        # Use coordinate-based indexing like _compute_source_removed for correct mapping with wrapping
        y_coords = torch.arange(self.world_size, device=device).unsqueeze(1).expand(-1, self.world_size)
        x_coords = torch.arange(self.world_size, device=device).unsqueeze(0).expand(self.world_size, -1)
        for idx, (dy, dx) in enumerate(offsets):
            ci, cj = idx // 3, idx % 3
            # For each destination (y, x), get contribution from source at (y-dy, x-dx) with wrapping
            y_src = (y_coords - dy) % self.world_size
            x_src = (x_coords - dx) % self.world_size
            distributed_total += contributions[ci, cj][y_src, x_src]
        
        # Calculate new cell candidates
        self.new_cell_candidates = (distributed_total > self.reproduction_threshold) & (self.topology_matrix == 0)
        
        # Store contributions to new cell candidates for rotation calculation
        self.new_cell_contributions = contributions.clone() if torch.any(self.new_cell_candidates) else None
        
        # Receiving mask
        receiving_mask = (self.topology_matrix.bool() | self.new_cell_candidates).float()
        
        if torch.any(receiving_mask):
            # Calculate what can actually be received at each destination (before accumulating)
            # This prevents energy loss when destinations hit capacity
            capacity = (1.0 - self.energy_matrix) * receiving_mask
            
            # Calculate energy that would be incoming at each destination
            temp_energy_matrix = self._accumulate_contributions(contributions, shareable_energy)
            energy_incoming = temp_energy_matrix - self.energy_matrix
            actual_received = torch.min(energy_incoming, capacity)
            
            # Calculate destination efficiency based on what can be received
            dest_efficiency = torch.where(
                energy_incoming > 0,
                actual_received / energy_incoming,
                torch.zeros_like(energy_incoming)
            ) * receiving_mask
            
            # Compute source removal based on what will actually be received
            source_removed = self._compute_source_removed(contributions, dest_efficiency, receiving_mask)
            
            # Build final energy matrix: start with current energy, subtract shareable_energy,
            # then add only the portion of contributions that will actually be received
            new_energy_matrix = self.energy_matrix - shareable_energy
            
            # Add contributions scaled by destination efficiency (only what can be received)
            offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),  (0, 0),  (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
            y_coords = torch.arange(self.world_size, device=device).unsqueeze(1).expand(-1, self.world_size)
            x_coords = torch.arange(self.world_size, device=device).unsqueeze(0).expand(self.world_size, -1)
            for idx, (dy, dx) in enumerate(offsets):
                ci, cj = idx // 3, idx % 3
                # For each destination (y, x), get contribution from source at (y-dy, x-dx) with wrapping
                y_src = (y_coords - dy) % self.world_size
                x_src = (x_coords - dx) % self.world_size
                # Scale by destination efficiency to only add what can be received
                y_dest = y_coords
                x_dest = x_coords
                dest_eff = dest_efficiency[y_dest, x_dest]
                new_energy_matrix += contributions[ci, cj][y_src, x_src] * dest_eff
            
            # Sources only lose what was actually sent and received (source_removed)
            # Energy that couldn't be received stays with sources (shareable_energy - source_removed)
            # Since we subtracted shareable_energy and only added scaled contributions,
            # we need to add back the energy sources kept
            new_energy_matrix = new_energy_matrix + (shareable_energy - source_removed)
            
            # Final update - preserve energy in existing cells and new candidates
            valid_mask = (self.topology_matrix.bool() | self.new_cell_candidates).float()
            self.energy_matrix = torch.clamp(new_energy_matrix * valid_mask, 0, 1)
        
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
#        print(f"Rendering debug info: {len(lines)} lines")  # Debug output
            
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
        
        # Render debug information
        lines = [
            f"FPS: {actual_fps:.1f}",
            f"",
            f"=== TOGGLES ===",
            f"Render Mode (m): {self.render_mode}",
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
            f"Energy Density Decay Modifier: {ENERGY_DENSITY_DECAY_MODIFIER}",
            f"Seed Boost: {STARTING_POSITION_TERRAIN_BOOST}",
            f"",
            f"=== SIMULATION STATE ===",
        ]
        
        # Add simulation data if available
        if simulation_data:
            organism_energy = torch.sum(simulation_data['energy']).item()
            terrain_energy = torch.sum(simulation_data['terrain']).item()
            total_energy = organism_energy + terrain_energy
            topology_count = torch.sum(simulation_data['topology']).item()
            new_cells = torch.sum(simulation_data['new_cell_candidates']).item()
            
            lines.extend([
                f"System Energy: {total_energy:.2f}",
                f"Terrain Energy: {terrain_energy:.2f}",
                f"Organism Energy: {organism_energy:.2f}",
                f"Cells: {topology_count:.0f}",
                f"New Cell Candidates: {new_cells:.0f}",
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
    
    def render(self, environment, topology, mask, new_cell_candidates=None, sharing_rate=None):
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
        if self.filters_enabled:
            # Apply mask where topology = 1
            if self.render_mode == "org_top":
                # Red topology
                image[0] = topology * 1.0 + (1 - topology) * env_scaled
                image[1] = topology * 0.0 + (1 - topology) * env_scaled
                image[2] = topology * 0.0 + (1 - topology) * env_scaled
            else:
                sharing_rate_clamped = sharing_rate.clamp(0, 1)
                print(sharing_rate_clamped)
                image[0] = topology + (1 - topology) * 0.0
                image[1] = topology * sharing_rate_clamped + (1 - topology) * env_scaled
                image[2] = topology * sharing_rate_clamped + (1 - topology) * env_scaled
                #image[3] = topology * torch.clamp(mask, 0.3, 1) + (1 - topology) * env_scaled # Alpha channel set to mask
        
            # Render new cell candidates as blue
            if new_cell_candidates is not None:
                image[0] = new_cell_candidates.float() * 0.0 + (1 - new_cell_candidates.float()) * image[0]
                image[1] = new_cell_candidates.float() * 0.0 + (1 - new_cell_candidates.float()) * image[1]
                image[2] = new_cell_candidates.float() * 1.0 + (1 - new_cell_candidates.float()) * image[2]
        
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
        if image_tensor.device.type != 'cuda' and image_tensor.device.type != 'mps':
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
        self.environment = Environment(self.world_size, NOISE_SCALE, QUANTIZATION_STEP)
        self.organism_manager = OrganismManager(self.world_size, ORGANISM_COUNT, self.environment.terrain)
        self.logger = Logger()
        self.tick = 0
        self.enable_debug = enable_debug
    
    def update_simulation(self):
        """Update one simulation tick"""
                
        # Compute energy decay and sharing
        harvested_energy = self.organism_manager.compute_energy(self.environment.terrain)
        
        # DISABLE topology expansion (major CPU bottleneck)
        self.organism_manager.compute_topology()
        
        # Compute environment changes
        self.environment.compute_environment(self.organism_manager.topology_matrix, harvested_energy)
        
        self.logger.update_fps()
       
        if self.enable_debug and self.tick % DEBUG_PRINT_INTERVAL == 0:
            debug_info = self.logger.get_debug_info()
            # Debug: Log total energy in system and terrain
            total_energy = torch.sum(self.organism_manager.energy_matrix).item()
            total_terrain = torch.sum(self.environment.terrain).item()
            system_energy = total_energy + total_terrain
            self.logger.log_tick(self.tick, 0, debug_info, None, None, total_energy, total_terrain, system_energy)
       
        # Clear GPU cache periodically
        if self.tick % GPU_CACHE_CLEAR_INTERVAL == 0:
            gpu_handler.clear_cache()
        
        self.tick += 1

        return {
            'terrain': self.environment.terrain,
            'topology': self.organism_manager.topology_matrix,
            'energy': self.organism_manager.energy_matrix,
            'new_cell_candidates': self.organism_manager.new_cell_candidates,
            'sharing_rate': self.organism_manager.sharing_rate_matrix
        }
    
    def reset_for_replay(self):
        """Reset simulation for replay with new CNN"""
        # Reset organism manager
        self.organism_manager = OrganismManager(self.world_size, ORGANISM_COUNT, self.environment.terrain)
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
        print(f"Loaded model: {latest_file}")
        return cnn
    except Exception as e:
        print(f"Couldn't load {latest_file}: {e}")
        return None

def start_cnn_evolution(grapher: Grapher | None = None, load_latest=False):
    """Start CNN evolution training"""
    print("Starting CNN Evolution Training...")
    # Set multiprocessing start method for PyTorch compatibility
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn', force=True)
    evolution_driver = CNNEvolutionDriver(WORLD_SIZE, epochs=CNN_TRAINING_EPOCHS, max_time=CNN_TRAINING_MAX_TIME)
    
    # Load latest model if requested
    if load_latest:
        if evolution_driver.ga.load_latest_model():
            # Copy loaded model to all subjects (as if it was the fittest from previous run)
            parent = evolution_driver.ga.subjects[0]
            for i in range(1, CNN_POPULATION_SIZE):
                evolution_driver.ga.subjects[i].conv1.weight.data = parent.conv1.weight.data.clone()
                evolution_driver.ga.subjects[i].conv1.bias.data = parent.conv1.bias.data.clone()
                evolution_driver.ga.subjects[i].conv2.weight.data = parent.conv2.weight.data.clone()
                evolution_driver.ga.subjects[i].conv2.bias.data = parent.conv2.bias.data.clone()
            print("Loaded latest model and initialized population from it")
    
    evolution_driver.grapher = grapher
    best_cnn = evolution_driver.run_evolution()
    
    print(f"\nTraining completed! Best CNN parameters:")
    print(f"conv1 weight shape: {best_cnn.conv1.weight.data.shape}")
    print(f"conv2 weight shape: {best_cnn.conv2.weight.data.shape}")
    
    return best_cnn
    

def main():
    """Main simulation loop with GPU-accelerated OpenGL rendering"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run CNN training mode (no OpenGL sim)')
    parser.add_argument('--load', action='store_true', help='Load the fittest model from the last generation of the last run (works with --train or normal sim)')
    args, _ = parser.parse_known_args()

    if args.train:
        # Clear saved networks at the beginning of training (only if not loading)
        if not args.load:
            clear_saved_networks()
        
        # Training mode with matplotlib graphs
        grapher = Grapher()
        start_cnn_evolution(grapher, load_latest=args.load)
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
    
    # Load model if requested
    if args.load:
        loaded_cnn = load_latest_cnn()
        if loaded_cnn is not None:
            simulation.organism_manager.energy_distribution_cnn = loaded_cnn
            print("Using loaded model in simulation")
        else:
            print("Failed to load model, using default CNN")
    
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
    global current_simulation, current_renderer, current_input_handler
    current_simulation = simulation
    current_renderer = renderer
    current_input_handler = input_handler
    
    def display():
        """OpenGL display callback - only handles rendering when window needs redraw"""
        global current_renderer, current_harvest_rate, replay_mode, current_best_cnn
        
        # Only render if we have data
        if last_sim_data is not None:
            # Update OpenGL texture and render
            current_renderer.update_texture(current_renderer.last_image)
            # Get logger from simulation for FPS display
            logger = current_simulation.logger if hasattr(current_simulation, 'logger') else None
            current_renderer.render_opengl(last_sim_data, current_harvest_rate, replay_mode, current_best_cnn, logger)
    
    def keyboard(key, x, y):
        """OpenGL keyboard callback"""
        global current_renderer, current_simulation
        
        if key == b'q' or key == 27:  # 'q' or ESC
            glut.glutLeaveMainLoop()
        elif key == b'm':
            current_renderer.toggle_render_mode()
        elif key == b'n':
            current_renderer.toggle_filters()
        elif key == b'h':
            # Toggle harvest rate between 0 and original value
            global current_harvest_rate
            if current_harvest_rate == 0:
                current_harvest_rate = ENERGY_HARVEST_RATE
                print(f"Harvest rate enabled: {ENERGY_HARVEST_RATE}")
            else:
                current_harvest_rate = 0
                print("Harvest rate disabled: 0")
    
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
            
            image_tensor = current_renderer.render(
                last_sim_data['terrain'], 
                last_sim_data['topology'], 
                mask,
                last_sim_data['new_cell_candidates'],
                last_sim_data.get('sharing_rate', None)
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
