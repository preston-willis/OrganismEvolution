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
from concurrent.futures import ThreadPoolExecutor

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
    """CNN that distributes energy across the entire world grid using 3x3 convolution"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # Create 3x3 kernel for energy distribution (excluding center)
        kernel = torch.tensor([[-12.8405, -10.0789, -13.9634],
        [ -5.2370, -13.8395, -10.9243],
        [ -4.8459,  -7.6581,  -6.9305]], device='mps:0')
        # kernel[1, 1] = 0  # Don't include center cell
        
        # Register as buffer for conv2d
        self.register_buffer('kernel', kernel)
        
        # Move model to device
        self.to(device)
        
    def forward(self, shareable_energy):
        """
        Input: shareable_energy of shape (world_size, world_size)
        Output: distributed energies of shape (world_size, world_size)
        """
        # Prepare input with batch/channel dims
        x = shareable_energy.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Convolution using the stored kernel
        out = torch.nn.functional.conv2d(x, self.kernel.unsqueeze(0).unsqueeze(0), padding=1)

        # Softmax each 3x3 locality on the OUTPUT (destination gets a per-source 3x3 distribution)
        patches_out = torch.nn.functional.unfold(out, kernel_size=3, padding=1)  # (1, 9, H*W)
        local_probs = torch.nn.functional.softmax(patches_out, dim=1)            # (1, 9, H*W)
        center_idx = 4
        H, W = out.shape[-2:]

        # Source-only gating and scaling: use the input shareable center per source
        patches_in = torch.nn.functional.unfold(x, kernel_size=3, padding=1)     # (1, 9, H*W)
        center_energy = patches_in[:, center_idx:center_idx+1, :]                # (1,1,H*W)
        gated_weighted = local_probs * center_energy                              # (1,9,H*W)

        # Fold back to accumulate contributions from all sources into destinations
        distributed = torch.nn.functional.fold(gated_weighted, output_size=(H, W), kernel_size=3, padding=1)  # (1,1,H,W)

        return distributed.squeeze(0).squeeze(0)


class CNNGeneticAlgorithm:
    """Genetic algorithm for training CNN weights with parallel GPU evaluation"""
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
        
        # Create batched kernel for parallel evaluation
        self.batched_kernels = torch.stack([cnn.kernel for cnn in self.subjects])  # (pop_size, 3, 3)
        
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
        
    def update_batched_kernels(self):
        """Update the batched kernels from individual CNNs"""
        self.batched_kernels = torch.stack([cnn.kernel for cnn in self.subjects])
        
    def crossover(self, parent):
        """Copy parent weights to all subjects"""
        for i in range(self.pop_size):
            if i != self.fittest_index:  # Don't overwrite the parent
                # Copy kernel weights
                self.subjects[i].kernel.data = parent.kernel.data.clone()
        
        # Update batched kernels
        self.update_batched_kernels()
                
    def mutate(self):
        """Apply mutations to all subjects except the fittest"""
        for i in range(self.pop_size):
            if i != self.fittest_index:  # Don't mutate the fittest
                # Create mutation mask
                mutation_mask = torch.rand_like(self.subjects[i].kernel) < self.mut_rate
                
                # Generate random mutations
                mutations = (torch.rand_like(self.subjects[i].kernel) - 0.5) * 2 * self.mut_mag
                
                # Apply mutations
                self.subjects[i].kernel.data[mutation_mask] += mutations[mutation_mask]
                
                # # Ensure center remains 0 (no self-connection)
                # self.subjects[i].kernel.data[1, 1] = 0
        
        # Update batched kernels
        self.update_batched_kernels()
                
    def save_model(self, index):
        """Save model weights to file"""
        filename = f'data/cnn_{self.run_id}_{self.fitness_scores[index]:.6f}.pt'
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


class ParallelCNNEvaluator:
    """Parallel CNN evaluator for GPU-accelerated fitness computation"""
    def __init__(self, world_size, max_time, device):
        self.world_size = world_size
        self.max_time = max_time
        self.device = device
        self.grapher = None
        self.current_generation_max_fitness = 0.0
        
    def evaluate_population_parallel(self, batched_kernels, pop_size):
        """Evaluate entire population in parallel on GPU using threading"""
        
        # Create simulation instances for each CNN (debug disabled during training)
        simulations = []
        for i in range(pop_size):
            sim = Simulation(enable_debug=False)
            # Replace the CNN with a custom one using the batched kernel
            sim.organism_manager.energy_distribution_cnn = self._create_cnn_from_kernel(batched_kernels[i])
            simulations.append(sim)
        
        # Run simulations in parallel using ThreadPoolExecutor
        fitness_scores = [0.0] * pop_size
        
        def evaluate_single(i, sim):
            fitness_scores[i] = self._evaluate_single_cnn(sim, bot_index=i)
        
        # Use ThreadPoolExecutor for parallel execution, continuously pump GUI
        import time as _time
        with ThreadPoolExecutor(max_workers=min(pop_size, 64)) as executor:
            futures = [executor.submit(evaluate_single, i, sim) for i, sim in enumerate(simulations)]
            # While evaluations are running, keep the matplotlib window responsive
            while True:
                all_done = True
                for f in futures:
                    if not f.done():
                        all_done = False
                        break
                if self.grapher is not None:
                    try:
                        self.grapher.process_queued()
                    except Exception:
                        pass
                if all_done:
                    break
                _time.sleep(0.03)

            # Ensure exceptions are surfaced
            for f in futures:
                f.result()
        
        return fitness_scores
    
    def _create_cnn_from_kernel(self, kernel):
        """Create a CNN with a specific kernel"""
        cnn = EnergyDistributionCNN(self.device)
        cnn.kernel.data = kernel.clone()
        return cnn
    
    def _evaluate_single_cnn(self, simulation, bot_index: int | None = None):
        """Evaluate a single CNN simulation"""
        total_energy = 0
        energy_history = []
        
        # Run simulation for max_time steps
        for t in range(self.max_time):
            sim_data = simulation.update_simulation()
            
            # Calculate fitness based on total energy and stability
            current_energy = torch.sum(simulation.organism_manager.energy_matrix).item()
            total_energy += current_energy
            energy_history.append(current_energy)

            # Tick graph update every 10 ticks if grapher attached
            if self.grapher is not None and t % 10 == 0:
                org_energy = current_energy
                env_energy = torch.sum(simulation.environment.terrain).item()
                total = org_energy + env_energy
                # enqueue only; GUI update happens in main thread
                self.grapher.enqueue_tick(t, self.current_generation_max_fitness, [current_energy], org_energy, env_energy, total)
                if bot_index is not None:
                    self.grapher.enqueue_bot_tick(bot_index, t, current_energy, org_energy, env_energy, total)
            
            # Early termination if energy drops too low
            if current_energy < CNN_FITNESS_EARLY_TERMINATION_THRESHOLD:
                break
                
        # Fitness is based on total energy and energy stability
        avg_energy = total_energy / len(energy_history) if energy_history else 0
        
        fitness = avg_energy
        return fitness


class CNNEvolutionDriver:
    """Evolution driver for CNN training with parallel GPU evaluation and replay"""
    def __init__(self, world_size, epochs=CNN_DEFAULT_EPOCHS, max_time=CNN_DEFAULT_MAX_TIME):
        self.world_size = world_size
        self.epochs = epochs
        self.max_time = max_time
        
        # Genetic algorithm parameters
        self.ga = CNNGeneticAlgorithm(CNN_POPULATION_SIZE, CNN_MUTATION_RATE, CNN_MUTATION_MAGNITUDE, device)
        
        # Parallel evaluator
        self.evaluator = ParallelCNNEvaluator(world_size, max_time, device)
        self.grapher = None
        
        # Replay simulation for showing best organism
        self.replay_simulation = None
        
    def evaluate_cnn(self, cnn, simulation):
        """Evaluate a CNN by running simulation and measuring fitness"""
        # Create a copy of simulation with this CNN (debug disabled during training)
        test_sim = Simulation(enable_debug=False)
        test_sim.organism_manager.energy_distribution_cnn = cnn
        
        total_energy = 0
        energy_history = []
        
        # Run simulation for max_time steps
        for t in range(self.max_time):
            sim_data = test_sim.update_simulation()
            
            # Calculate fitness based on total energy and stability
            current_energy = torch.sum(test_sim.organism_manager.energy_matrix).item()
            total_energy += current_energy
            energy_history.append(current_energy)
            
            # Early termination if energy drops too low
            if current_energy < CNN_FITNESS_EARLY_TERMINATION_THRESHOLD:
                break
                
        # Fitness is based on total energy and energy stability
        avg_energy = total_energy / len(energy_history) if energy_history else 0

        fitness = avg_energy
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
        
        print(f"Created replay simulation with best CNN (fitness: {self.ga.fitness_scores[self.ga.fittest_index]:.6f})")
        print("Press 'r' to toggle replay mode and see the best organism in action!")
        
    def run_evolution(self):
        """Run the evolution process with parallel GPU evaluation"""
        print(f"\nStarting CNN Evolution - {self.epochs} generations")
        print(f"Population size: {CNN_POPULATION_SIZE}")
        print(f"Using parallel GPU evaluation")
        
        for gen in range(self.epochs):
            print(f"\nGeneration {gen + 1}/{self.epochs}")
            
            # Evaluate entire population in parallel
            print("Evaluating population in parallel...")
            if self.grapher is not None:
                self.evaluator.grapher = self.grapher
            fitness_scores = self.evaluator.evaluate_population_parallel(
                self.ga.batched_kernels, 
                CNN_POPULATION_SIZE
            )
            
            # Update fitness scores
            self.ga.fitness_scores = fitness_scores
            
            # Print individual results
            for i, fitness in enumerate(fitness_scores):
                print(f"CNN {i}: fitness = {fitness:.6f}")
                
            # Run one generation
            self.ga.compute_generation()
            
            # Print summary
            best_fitness = self.ga.fitness_scores[self.ga.fittest_index]
            print(f"Best fitness: {best_fitness:.6f}")
            print(f"Best CNN kernel:\n{self.ga.subjects[self.ga.fittest_index].kernel.data}")
            
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
            
        print("\nEvolution completed!")
        return self.ga.subjects[self.ga.fittest_index]


class Environment:
    def __init__(self, world_size, noise_scale, quantization_step):
        self.world_size = world_size
        self.noise_scale = noise_scale
        self.quantization_step = quantization_step
        self.terrain = self.generate_terrain()
    
    def generate_terrain(self):
        """Generate Perlin noise terrain"""
        perlin_values = np.zeros((self.world_size, self.world_size))
        for i in range(self.world_size):
            for j in range(self.world_size):
                # Higher frequency noise with more octaves
                perlin_values[i, j] = pnoise2(i * self.noise_scale * NOISE_FREQUENCY_MULTIPLIER, j * self.noise_scale * NOISE_FREQUENCY_MULTIPLIER, octaves=NOISE_OCTAVES)
        
        # Normalize to 0-1
        perlin_values = (perlin_values - perlin_values.min()) / (max(perlin_values.max() - perlin_values.min(), 1e-6))
        
        # Apply power function to create more low energy zones
        perlin_values = np.power(perlin_values, NOISE_POWER)

        perlin_values = torch.tensor(perlin_values, dtype=torch.float32, device=device)
        dead_mask = perlin_values > 0.3
        perlin_values = perlin_values * dead_mask    
        # Convert to tensor
        return torch.clamp(perlin_values, 0, 1)
    
    def compute_environment(self, topology_matrix, harvested_energy):
        """Modify environment based on organism presence"""
        # Deplete terrain energy by the amount harvested
        self.terrain = torch.clamp(self.terrain - (harvested_energy * topology_matrix), 0, 1)

class OrganismManager:
    def __init__(self, world_size, organism_count, terrain):
        self.world_size = world_size
        self.terrain = terrain

        self.positions = torch.tensor(ORGANISM_POSITIONS, dtype=torch.long, device=device)
        self.topology_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
        self.energy_matrix = torch.zeros((world_size, world_size), dtype=torch.float32, device=device)
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
    
    def compute_topology(self):
        """Reproduce using new_cell_candidates mask from energy sharing"""
        # Use new_cell_candidates mask for reproduction instead of random selection
        if not torch.any(self.new_cell_candidates):
            return
        
        # Check energy threshold on the new cell candidates
        energy_mask = (self.energy_matrix >= self.reproduction_threshold) * self.new_cell_candidates
        
        # Add selected positions to topology
        self.topology_matrix[energy_mask] = 1
    
    def compute_energy(self, terrain):
        """Deplete all energies by ENERGY_DECAY and allow energy sharing between adjacent cells"""
        # Set organism energy to 1 in a 20-radius area BEFORE energy sharing calculation
        if terrain is not None:
            self.terrain = terrain
                
        # Remove organisms with energy below quantization step
        low_energy_mask = self.energy_matrix < DEATH_THRESHOLD
        self.topology_matrix[low_energy_mask] = 0
        
        # Apply energy decay only where topology = 1 (organisms)
        # Harvest energy from terrain where topology = 1, max harvest = ENERGY_HARVEST_RATE
        harvested_energy = torch.minimum(self.terrain, torch.tensor(current_harvest_rate, device=device))
        self.energy_matrix = torch.clamp((self.energy_matrix + harvested_energy - ENERGY_DECAY) * self.topology_matrix, 0, 1)
        
        
        # Energy sharing between adjacent cells (conserving total energy)
        sharing_rate = ENERGY_SHARING_RATE
        
        # Calculate how much energy each cell can share (only organisms can share)
        shareable_energy = self.energy_matrix * self.topology_matrix * sharing_rate
        
        # Distribute energy to adjacent cells using CNN
        distributed_energy = self.energy_distribution_cnn(shareable_energy)
        
        # Store mask of where energy is shared but topology = 0 (new cell candidates)
        self.new_cell_candidates = (distributed_energy > self.reproduction_threshold) & (self.topology_matrix == 0)
        
        # Only organisms can receive energy
        receiving_mask = (self.topology_matrix + self.new_cell_candidates) > 0
        
        if torch.any(receiving_mask):
            # Remove shared energy from source cells FIRST
            self.energy_matrix = torch.clamp(self.energy_matrix - shareable_energy, 0, 1)
            
            # Add distributed energy to destination cells (scale by 1/8 to conserve energy)
            # Each cell shares with 8 neighbors, so we need to divide by 8
            self.energy_matrix = torch.clamp(self.energy_matrix + (distributed_energy) * receiving_mask.float(), 0, 1)
        
        return harvested_energy
class Renderer:
    def __init__(self, world_size):
        self.world_size = world_size
        self.render_mode = "org_energy"  # "org_top" or "org_energy"
        self.filters_enabled = True  # True = show organisms, False = environment only
        self.texture_id = None
        self.quad_vbo = None
        self.opengl_initialized = False
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
            glRasterPos2f(x, y)
            for char in text:
                glut.glutBitmapCharacter(glut.GLUT_BITMAP_8_BY_13, ord(char))
        except Exception as e:
            print(f"Text rendering error: {e}")
    
    def render_debug_info(self, simulation_data, current_harvest_rate, replay_mode, current_best_cnn):
        """Render debug information overlay"""
        if not self.debug_text_enabled:
            return
        
#        print(f"Rendering debug info: {len(lines)} lines")  # Debug output
            
        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.world_size, self.world_size, 0, -1, 1)  # Flip Y axis
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing and enable blending for text
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set text color (bright green with slight transparency)
        glColor4f(0.0*255, 1.0*255, 0.0*255, 0.95*255)
        
        # Calculate text position (top-left corner, flipped coordinates)
        text_x = 10
        text_y = 20
        line_height = 15
        
        # Render debug information
        lines = [
            f"DEBUG TEXT TEST - VISIBLE?",
            f"=== SIMULATION DEBUG ===",
            f"Render Mode: {self.render_mode}",
            f"Filters: {'ON' if self.filters_enabled else 'OFF'}",
            f"Debug Text: {'ON' if self.debug_text_enabled else 'OFF'}",
            f"Replay Mode: {'ON' if replay_mode else 'OFF'}",
            f"",
            f"=== CONFIGURATION ===",
            f"World Size: {WORLD_SIZE}",
            f"Organism Count: {ORGANISM_COUNT}",
            f"Harvest Rate: {current_harvest_rate:.4f}",
            f"Energy Decay: {ENERGY_DECAY:.4f}",
            f"Energy Sharing Rate: {ENERGY_SHARING_RATE}",
            f"Reproduction Threshold: {REPRODUCTION_THRESHOLD:.4f}",
            f"Death Threshold: {DEATH_THRESHOLD:.4f}",
            f"",
            f"=== SIMULATION STATE ===",
        ]
        
        # Add simulation data if available
        if simulation_data:
            total_energy = torch.sum(simulation_data['energy']).item()
            total_terrain = torch.sum(simulation_data['terrain']).item()
            topology_count = torch.sum(simulation_data['topology']).item()
            new_cells = torch.sum(simulation_data['new_cell_candidates']).item()
            
            lines.extend([
                f"Total Energy: {total_energy:.2f}",
                f"Total Terrain: {total_terrain:.2f}",
                f"Organisms: {topology_count:.0f}",
                f"New Cell Candidates: {new_cells:.0f}",
            ])
        
        # Add CNN info if available
        if current_best_cnn is not None:
            lines.extend([
                f"",
                f"=== CNN STATE ===",
                f"Best CNN Active: YES",
                f"Kernel Center: {current_best_cnn.kernel.data[1, 1].item():.3f}",
            ])
        else:
            lines.extend([
                f"",
                f"=== CNN STATE ===",
                f"Best CNN Active: NO",
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
    
    def render(self, environment, topology, mask, new_cell_candidates=None):
        """Render the current state using PyTorch tensors directly - GPU accelerated"""
        env_scaled = environment.clamp(0, 1)
        
        # Create RGB image tensor
        image = torch.zeros((3, self.world_size, self.world_size), device=device, dtype=torch.float32)
        
        # Apply environment to all channels
        image[0] = env_scaled  # Red channel
        image[1] = env_scaled  # Green channel  
        image[2] = env_scaled  # Blue channel
        
        # Only apply organism visualization if filters are enabled
        if self.filters_enabled:
            # Apply mask where topology = 1
            if self.render_mode == "org_top":
                # Red topology
                image[0] = topology * 1.0 + (1 - topology) * env_scaled
                image[1] = topology * 0.0 + (1 - topology) * env_scaled
                image[2] = topology * 0.0 + (1 - topology) * env_scaled
            else:
                # Green energy
                image[0] = topology * 0.0 + (1 - topology) * env_scaled
                image[1] = topology * mask + (1 - topology) * env_scaled
                image[2] = topology * 0.0 + (1 - topology) * env_scaled
            
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
        
        # Process entirely on GPU: clamp, scale, permute, convert to uint8
        image_tensor = image_tensor.clamp(0, 1) * PIXEL_SCALE
        image_tensor = image_tensor.permute(1, 2, 0)  # CHW -> HWC
        image_tensor = image_tensor.byte()  # Convert to uint8 on GPU
        
        # Only transfer to CPU at the very end for OpenGL texture upload
        # This is the minimal possible CPU transfer
        image_np = image_tensor.cpu().numpy()
        
        # Update OpenGL texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.world_size, self.world_size, 0, GL_RGB, GL_UNSIGNED_BYTE, image_np)
    
    def render_opengl(self, simulation_data=None, current_harvest_rate=0, replay_mode=False, current_best_cnn=None):
        """Render using OpenGL (OpenGL 2.1 compatible) with debug text"""
        if not self.opengl_initialized:
            return
            
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Enable texture mapping
        glEnable(GL_TEXTURE_2D)
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
        self.render_debug_info(simulation_data, current_harvest_rate, replay_mode, current_best_cnn)
        
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
            'new_cell_candidates': self.organism_manager.new_cell_candidates
        }
    
    def reset_for_replay(self):
        """Reset simulation for replay with new CNN"""
        # Reset organism manager
        self.organism_manager = OrganismManager(self.world_size, ORGANISM_COUNT, self.environment.terrain)
        # Reset tick counter
        self.tick = 0
    

def start_cnn_evolution(grapher: Grapher | None = None):
    """Start CNN evolution training"""
    print("Starting CNN Evolution Training...")
    evolution_driver = CNNEvolutionDriver(WORLD_SIZE, epochs=CNN_TRAINING_EPOCHS, max_time=CNN_TRAINING_MAX_TIME)
    evolution_driver.grapher = grapher
    best_cnn = evolution_driver.run_evolution()
    
    print(f"\nTraining completed! Best CNN kernel:")
    print(best_cnn.kernel.data)
    
    return best_cnn
    

def main():
    """Main simulation loop with GPU-accelerated OpenGL rendering"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run CNN training mode (no OpenGL sim)')
    args, _ = parser.parse_known_args()

    if args.train:
        # Training mode with matplotlib graphs
        grapher = Grapher()
        start_cnn_evolution(grapher)
        return
    # Initialize OpenGL/GLUT
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
    glut.glutInitWindowSize(WORLD_SIZE, WORLD_SIZE)
    glut.glutCreateWindow(b"Organism Simulation - OpenGL GPU Accelerated")
    
    # Setup OpenGL
    glEnable(GL_TEXTURE_2D)
    glClearColor(OPENGL_CLEAR_COLOR_R, OPENGL_CLEAR_COLOR_G, OPENGL_CLEAR_COLOR_B, OPENGL_CLEAR_COLOR_A)
    
    # Set up viewport and projection for full-screen rendering
    glViewport(0, 0, WORLD_SIZE, WORLD_SIZE)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    simulation = Simulation()
    renderer = Renderer(WORLD_SIZE)
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
            current_renderer.render_opengl(last_sim_data, current_harvest_rate, replay_mode, current_best_cnn)
    
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
        elif key == b't':
            # Start CNN training
            print("Starting CNN evolution training...")
            best_cnn = start_cnn_evolution()
            # Replace current CNN with trained one
            current_simulation.organism_manager.energy_distribution_cnn = best_cnn
            print("CNN training completed and applied!")
        elif key == b'r':
            # Toggle replay mode
            global replay_mode
            replay_mode = not replay_mode
            if replay_mode:
                print("Replay mode enabled - showing best organism from last generation")
            else:
                print("Replay mode disabled - showing normal simulation")
        elif key == b'd':
            # Toggle debug text
            current_renderer.toggle_debug_text()
    
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
                last_sim_data['new_cell_candidates']
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
