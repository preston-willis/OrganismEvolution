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

    def forward(self, shareable_energy, terrain, sharing_rate, hidden_channels, rotation_matrix):
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

def _evaluate_cnn_worker(args):
    """Worker function for multiprocessing CNN evaluation"""
    cnn_state_dict, world_size, max_time, bot_index, collect_tick_data = args
    
    # Use the device that was set during worker initialization
    global _worker_device
    
    # Create CNN and load state dict
    cnn = EnergyDistributionCNN(_worker_device)
    cnn.load_state_dict(cnn_state_dict)
    cnn._zero_hidden_channel_bias()
    
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
    _release_device_memory(_worker_device)
    
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
        _release_device_memory(self.device)
        
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
    def __init__(self, world_size, noise_scale, quantization_step):
        self.world_size = world_size
        self.noise_scale = noise_scale
        self.quantization_step = quantization_step
        self.environment_type = ENVIRONMENT_TYPE
        self.time = 0.0  # For moving perlin noise
        self.terrain = self.generate_terrain()
    
    def generate_terrain(self):
        """Generate terrain based on environment type"""
        if self.environment_type == 1:
            # Type 1: Energy masks (static terrain with energy sources)
            return self._generate_energy_mask_terrain()
        elif self.environment_type == 2:
            # Type 2: Sine waves
            return self._generate_sine_terrain()
        elif self.environment_type == 3:
            # Type 3: Moving perlin noise
            return self._generate_perlin_terrain()
        else:
            # Default to sine waves
            return self._generate_sine_terrain()
    
    def _generate_energy_mask_terrain(self):
        """Generate static terrain with energy masks at starting positions"""
        terrain = torch.zeros((self.world_size, self.world_size), dtype=torch.float32, device=device)
        positions = torch.tensor(ORGANISM_POSITIONS, dtype=torch.long, device=device)
        if positions.numel() > 0:
            y_coords, x_coords = positions[:, 1], positions[:, 0]
            energy_radius = 1
            circle_mask = torch.zeros((self.world_size, self.world_size), device=device)
            circle_mask[y_coords, x_coords] = True
            circle_mask = circle_mask.unsqueeze(0).unsqueeze(0)
            circle_mask = torch.nn.functional.conv2d(circle_mask, torch.ones((1, 1, 2 * energy_radius + 1, 2 * energy_radius + 1), device=device), padding=energy_radius)
            circle_mask = circle_mask.squeeze(0).squeeze(0)
            circle_mask = circle_mask > 0
            terrain[circle_mask] = STARTING_POSITION_TERRAIN_BOOST
        return torch.clamp(terrain, 0, 1)
    
    def _generate_sine_terrain(self):
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
        radial_decay = torch.exp(-dist / (max_dist * TERRAIN_RADIAL_DECAY_SCALE))
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
    
    def _generate_perlin_terrain(self):
        """Generate perlin noise terrain by taking 2D slices of 3D perlin noise through time"""
        # Use CPU for coordinate generation to avoid GPU memory
        x = torch.arange(self.world_size, dtype=torch.float32)
        y = torch.arange(self.world_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Convert to numpy for pnoise3 (keep on CPU)
        xx_np = xx.numpy()
        yy_np = yy.numpy()
        
        # Generate each octave separately with independent time progression
        noise_values = np.zeros((self.world_size, self.world_size), dtype=np.float32)
        
        for octave in range(NOISE_OCTAVES):
            # Each octave has its own time progression for independent morphing
            octave_time = self.time * (1.0 + octave * 0.3) + octave * 5.0
            
            # Base scale for this octave
            octave_scale = PERLIN_NOISE_SCALE * (2.0 ** octave)
            
            # Generate noise for this octave using 3D perlin noise
            # Sample 2D slices (x, y) at different time values (z)
            octave_noise = np.zeros((self.world_size, self.world_size), dtype=np.float32)
            for i in range(self.world_size):
                for j in range(self.world_size):
                    # Sample 3D perlin noise: (x, y, time)
                    # Time is the Z dimension, creating natural morphing as we move through time
                    octave_noise[i, j] = pnoise3(
                        xx_np[i, j] * octave_scale,
                        yy_np[i, j] * octave_scale,
                        octave_time,
                        octaves=1,
                        base=octave * 10
                    )
            
            # Add this octave with its independent amplitude modulation
            noise_values += octave_noise / (2.0 ** octave)
        
        # Normalize from perlin range to [0, 1]
        # Perlin noise with multiple octaves typically ranges roughly [-1, 1]
        values = torch.from_numpy(noise_values).to(device)
        values = (values + 1.0) * 0.5
        
        # Apply power shaping
        values = torch.pow(values, NOISE_POWER)
        
        # Apply dead mask
        dead_mask = values > ENV_NOISE_THRESHOLD
        values = values * dead_mask
        return torch.clamp(values, 0, 1)
    
    def compute_environment(self, topology_matrix, harvested_energy):
        """Modify environment based on organism presence"""
        # Deplete terrain by the amount organisms harvested
        self.terrain.copy_(torch.clamp(self.terrain - (harvested_energy * topology_matrix), 0, 1))
        
        if self.environment_type == 1:
            # Type 1: Apply terrain boost at organism starting positions
            positions = torch.tensor(ORGANISM_POSITIONS, dtype=torch.long, device=device)
            if positions.numel() > 0:
                # Create 4 additional sources 3px away from each original position
                additional_positions = []
                for pos in positions:
                    x, y = pos[0].item(), pos[1].item()
                    offsets = [(-5, 0), (5, 0), (0, -5), (0, 5)]
                    for dx, dy in offsets:
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < self.world_size and 0 <= new_y < self.world_size:
                            additional_positions.append([new_x, new_y])
                
                if additional_positions:
                    additional_positions_tensor = torch.tensor(additional_positions, dtype=torch.long, device=device)
                    all_positions = torch.cat([positions, additional_positions_tensor], dim=0)
                else:
                    all_positions = positions
                
                y_coords, x_coords = all_positions[:, 1], all_positions[:, 0]
                energy_radius = 1
                circle_mask = torch.zeros((self.world_size, self.world_size), device=device)
                circle_mask[y_coords, x_coords] = True
                circle_mask = circle_mask.unsqueeze(0).unsqueeze(0)
                circle_mask = torch.nn.functional.conv2d(circle_mask, torch.ones((1, 1, 2 * energy_radius + 1, 2 * energy_radius + 1), device=device), padding=energy_radius)
                circle_mask = circle_mask.squeeze(0).squeeze(0)
                circle_mask = circle_mask > 0
                self.terrain[circle_mask] = self.terrain[circle_mask] + STARTING_POSITION_TERRAIN_BOOST
        
        elif self.environment_type == 3:
            # Type 3: Update moving perlin noise
            self.time += PERLIN_TIME_SPEED
            self.terrain = self._generate_perlin_terrain()
            

class OrganismManager:
    def __init__(self, world_size, organism_count, terrain):
        self.world_size = world_size
        self.terrain = terrain

        self.positions = torch.tensor(ORGANISM_POSITIONS, dtype=torch.long, device=device)
        self.topology_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.energy_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.sharing_rate_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.hidden_channels = torch.zeros((1, world_size, world_size), dtype=torch.float32, device=device)
        self.rotation_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.parent_giver_dir = torch.full((world_size, world_size), -1, dtype=torch.long, device=device)
        self.new_cell_candidates = torch.zeros((world_size, world_size), dtype=torch.bool, device=device)
        self.pending_birth_energy = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.destroyed_energy = 0.0
        self._tick_destroyed = torch.zeros((), device=device, dtype=torch.float32)
        self._has_new_cell_candidates = False
        self._initialize_topology()
        
        # Reproduction parameters
        self.reproduction_threshold = REPRODUCTION_THRESHOLD
        
        # Energy distribution CNN
        self.energy_distribution_cnn = EnergyDistributionCNN(device)
        self._init_contribution_conv_weights()
        self._init_giver_dir_gather()

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

    def _flush_tick_destroyed(self):
        self.destroyed_energy += self._tick_destroyed.item()
        self._tick_destroyed.zero_()

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
    
    def _initialize_topology(self):
        """Initialize topology and energy with organism positions"""
        if self.positions.numel() > 0:
            y_coords, x_coords = self.positions[:, 1], self.positions[:, 0]
            self.topology_matrix[y_coords, x_coords] = 1
            self.energy_matrix[y_coords, x_coords] = 1
            self.sharing_rate_matrix[y_coords, x_coords] = float(ENERGY_SHARING_RATE > 0)
            self.hidden_channels[:, y_coords, x_coords] = 0
            self.rotation_matrix[y_coords, x_coords] = 0
    
    def compute_topology(self):
        """Reproduce using new_cell_candidates mask from energy sharing"""
        if not self._has_new_cell_candidates:
            if not self.new_cell_candidates.any().item():
                return
        self._has_new_cell_candidates = False
        
        # Birth when incoming shared energy at candidate meets threshold
        energy_mask = self.new_cell_candidates & (self.pending_birth_energy >= self.reproduction_threshold)
        
        # Parent, rotation, and sharing rate (from parent hidden state) for new cells
        child_sharing_rate = torch.full_like(self.sharing_rate_matrix, float(ENERGY_SHARING_RATE > 0))
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
            child_sharing_rate = torch.where(birth_mask, child_hidden, child_sharing_rate)
        
        # Add selected positions to topology and commit birth energy
        self.topology_matrix[energy_mask] = 1
        self.energy_matrix[energy_mask] = self.pending_birth_energy[energy_mask]
        self.pending_birth_energy[energy_mask] = 0
        
        self.sharing_rate_matrix[energy_mask] = child_sharing_rate[energy_mask]
        self.hidden_channels[:, energy_mask] = 0
    
    def _apply_harvest_and_decay(self, terrain):
        """Harvest energy from terrain and apply decay"""
        if terrain is not None:
            self.terrain = terrain
                
        # Remove organisms with energy below threshold
        low_energy_mask = self.energy_matrix < DEATH_THRESHOLD
        self._tick_destroyed += (self.energy_matrix * low_energy_mask.float()).sum()
        self.energy_matrix[low_energy_mask] = 0
        self.topology_matrix[low_energy_mask] = 0
        self.sharing_rate_matrix[low_energy_mask] = 0
        self.hidden_channels[:, low_energy_mask] = 0
        self.rotation_matrix[low_energy_mask] = 0
        self.parent_giver_dir[low_energy_mask] = -1
        
        # Harvest energy from terrain
        harvested_energy = torch.minimum(self.terrain, self.sharing_rate_matrix * ENERGY_HARVEST_RATE)
        
        # Decay: sparse organism neighborhood + low local terrain → higher loss
        energy_unsqueezed = self.energy_matrix.unsqueeze(0).unsqueeze(0)
        org_avg = torch.nn.functional.conv2d(energy_unsqueezed, self._org_avg_weight, padding=1).squeeze(0).squeeze(0)
        decay_amount = ENERGY_DENSITY_DECAY_MODIFIER * self.sharing_rate_matrix**2 * (1.0 - org_avg) * (1.0 - self.terrain)
        cell_decay = (decay_amount + ENERGY_DECAY) * self.topology_matrix
        energy_after_harvest = self.energy_matrix + harvested_energy
        energy_before_decay = energy_after_harvest * self.topology_matrix
        self.energy_matrix = torch.clamp((energy_after_harvest - cell_decay) * self.topology_matrix, 0, 1)
        self._tick_destroyed += (energy_before_decay - self.energy_matrix).sum()

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
        harvested_energy = self._apply_harvest_and_decay(terrain)
        
        # Shareable outbound energy (new tensor; energy_matrix is updated later in sharing)
        shareable_energy = self.energy_matrix * self.topology_matrix * self.sharing_rate_matrix
        
        # Get 3x3 proportions, sharing_rate output, and hidden_channels output from CNN
        proportions, _, hidden_channels_output = self.energy_distribution_cnn(shareable_energy, terrain, self.sharing_rate_matrix, self.hidden_channels, self.rotation_matrix)
        
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
        self._tick_destroyed += (unclamped - self.energy_matrix).sum()
        
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
            f"Energy Density Decay Modifier: {ENERGY_DENSITY_DECAY_MODIFIER}",
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
            system_energy = organism_energy + terrain_energy + pending_energy + destroyed_energy
            topology_count = torch.sum(simulation_data['topology']).item()
            new_cells = torch.sum(simulation_data['new_cell_candidates']).item()
            
            lines.extend([
                f"System Energy: {system_energy:.2f}",
                f"Terrain Energy: {terrain_energy:.2f}",
                f"Organism Energy: {organism_energy:.2f}",
                f"Pending Birth Energy: {pending_energy:.2f}",
                f"Destroyed Energy: {destroyed_energy:.2f}",
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
    
    def render(self, environment, topology, mask, new_cell_candidates=None, sharing_rate=None, hidden_channels=None):
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
            total_pending = torch.sum(self.organism_manager.pending_birth_energy).item()
            system_energy = total_energy + total_terrain + total_pending + self.organism_manager.destroyed_energy
            self.logger.log_tick(self.tick, 0, debug_info, None, None, total_energy, total_terrain, system_energy)
       
        # Clear GPU cache periodically (interactive sim only; skipped during training)
        if self.enable_debug and self.tick % GPU_CACHE_CLEAR_INTERVAL == 0:
            gpu_handler.clear_cache()
            if self.environment.environment_type == 3:
                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        self.tick += 1

        return {
            'terrain': self.environment.terrain,
            'topology': self.organism_manager.topology_matrix,
            'energy': self.organism_manager.energy_matrix,
            'new_cell_candidates': self.organism_manager.new_cell_candidates,
            'sharing_rate': self.organism_manager.sharing_rate_matrix,
            'hidden_channels': self.organism_manager.hidden_channels,
            'pending_birth_energy': self.organism_manager.pending_birth_energy,
            'destroyed_energy': self.organism_manager.destroyed_energy,
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
        cnn._zero_hidden_channel_bias()
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
  2                    Switch to environment type 2 (sine waves)
  3                    Switch to environment type 3 (moving perlin noise)

CONFIGURATION PARAMETERS (from config.py):
  World Configuration:
    WORLD_SIZE              Grid size (default: 72)
    ORGANISM_COUNT          Number of starting organisms (default: 1)
  
  Environment Configuration:
    ENVIRONMENT_TYPE        Terrain type: 1=energy masks, 2=sine waves, 3=moving perlin noise (default: 2)
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
  
  Performance Configuration:
    DEVICE_TYPE             Preferred device: "mps", "cuda", or "cpu" (default: "mps")
    GPU_CACHE_CLEAR_INTERVAL  Ticks between GPU cache clears (default: 10)

For more details, see config.py
        '''
    )
    parser.add_argument('--train', action='store_true', help='Run CNN training mode (no OpenGL sim)')
    parser.add_argument('--graph', action='store_true', help='Show matplotlib graphs during --train')
    parser.add_argument('--load', action='store_true', help='Load the fittest model from the last generation of the last run (works with --train or normal sim)')
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
            current_renderer.render_opengl(last_sim_data, current_harvest_rate, replay_mode, current_best_cnn, logger)
    
    def keyboard(key, x, y):
        """OpenGL keyboard callback"""
        global current_renderer, current_simulation, main_args
        
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
        elif key == b'h':
            # Toggle harvest rate between 0 and original value
            global current_harvest_rate
            if current_harvest_rate == 0:
                current_harvest_rate = ENERGY_HARVEST_RATE
                print(f"Harvest rate enabled: {ENERGY_HARVEST_RATE}")
            else:
                current_harvest_rate = 0
                print("Harvest rate disabled: 0")
        elif key == b'r':
            # Reload simulation (preserve environment type)
            print("Reloading simulation...")
            global current_simulation
            current_env_type = current_simulation.environment.environment_type
            current_simulation = Simulation()
            # Restore environment type
            current_simulation.environment.environment_type = current_env_type
            current_simulation.environment.terrain = current_simulation.environment.generate_terrain()
            if main_args.load:
                loaded_cnn = load_latest_cnn()
                if loaded_cnn is not None:
                    current_simulation.organism_manager.energy_distribution_cnn = loaded_cnn
                    print("Using loaded model in simulation")
            print(f"Simulation reloaded (environment type: {current_env_type})")
        elif key == b'1':
            # Switch to environment type 1 (energy masks)
            print("Switching to environment type 1 (energy masks)")
            import config
            config.ENVIRONMENT_TYPE = 1
            current_simulation.environment.environment_type = 1
            current_simulation.environment.terrain = current_simulation.environment._generate_energy_mask_terrain()
        elif key == b'2':
            # Switch to environment type 2 (sine waves)
            print("Switching to environment type 2 (sine waves)")
            import config
            config.ENVIRONMENT_TYPE = 2
            current_simulation.environment.environment_type = 2
            current_simulation.environment.terrain = current_simulation.environment._generate_sine_terrain()
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
            
            image_tensor = current_renderer.render(
                last_sim_data['terrain'], 
                last_sim_data['topology'], 
                mask,
                last_sim_data['new_cell_candidates'],
                last_sim_data.get('sharing_rate', None),
                last_sim_data.get('hidden_channels', None)
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
