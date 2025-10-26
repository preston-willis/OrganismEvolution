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
import ctypes

# Import our modules
from config import *
from gpu_handler import GPUHandler
from logger import Logger
from input_handler import InputHandler

# Initialize GPU handler
gpu_handler = GPUHandler()
device = gpu_handler.get_device()

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
        perlin_values = (perlin_values - perlin_values.min()) / (perlin_values.max() - perlin_values.min())
        
        # Apply power function to create more low energy zones
        perlin_values = np.power(perlin_values, NOISE_POWER)
        
        # Quantize
        terrain = np.round(perlin_values / self.quantization_step) * self.quantization_step
        terrain = np.clip(terrain, 0, 1)
        
        # Convert to tensor
        return torch.tensor(terrain, dtype=torch.float16, device=device)
    
    def compute_environment(self, topology_matrix):
        """Modify environment based on organism presence"""
        # Set environment to 0 where organisms exist
        self.terrain = torch.where(topology_matrix > 0, torch.tensor(0.0, device=device), self.terrain)

class OrganismManager:
    def __init__(self, world_size, organism_count, terrain):
        self.world_size = world_size
        self.terrain = terrain

        self.positions = torch.tensor(ORGANISM_POSITIONS, dtype=torch.long, device=device)
        self.topology_matrix = torch.zeros((world_size, world_size), dtype=torch.float16, device=device)
        self.energy_matrix = terrain.clone()
        self.new_cell_candidates = torch.zeros((world_size, world_size), dtype=torch.bool, device=device)
        self._initialize_topology()
        
        # Reproduction parameters
        self.reproduction_threshold = REPRODUCTION_THRESHOLD
        self.reproduction_probability = REPRODUCTION_PROBABILITY
    
    def _initialize_topology(self):
        """Initialize topology and energy with organism positions"""
        if self.positions.numel() > 0:
            y_coords, x_coords = self.positions[:, 1], self.positions[:, 0]
            self.topology_matrix[y_coords, x_coords] = 1
    
    def compute_topology(self):
        """Add random adjacent cells to topology using efficient convolution"""
        # Use convolution to find adjacent empty cells efficiently
        # Create a 3x3 kernel to find adjacent positions
        kernel = torch.ones((3, 3), device=device, dtype=torch.float16)
        kernel[1, 1] = 0  # Don't count the center cell itself
        
        # Convolve to find adjacent cells to existing organisms
        adjacent_count = torch.nn.functional.conv2d(
            self.topology_matrix.unsqueeze(0).unsqueeze(0), 
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=1
        ).squeeze()
        
        # Find positions that are adjacent to organisms but not occupied
        adjacent_mask = (adjacent_count > 0) & (self.topology_matrix == 0)
        
        if not torch.any(adjacent_mask):
            return
        
        # Check energy threshold directly on the mask
        energy_mask = (self.energy_matrix >= self.reproduction_threshold) & adjacent_mask
        
        if not torch.any(energy_mask):
            return
        
        # Random selection using probability - apply directly to the mask
        random_values = torch.rand_like(self.topology_matrix, dtype=torch.float16)
        selected_mask = (random_values < self.reproduction_probability) & energy_mask
        
        if torch.any(selected_mask):
            # Count how many cells are being spawned
            num_spawned = torch.sum(selected_mask).item()
            
            # Reduce energy of parent cells by REPRODUCTION_THRESHOLD * spawned cells
            # Find parent cells (organisms that are adjacent to the new cells)
            parent_energy_reduction = REPRODUCTION_THRESHOLD * num_spawned/8
            
            # Create a mask for parent cells (organisms adjacent to new cells)
            parent_kernel = torch.ones((3, 3), device=device, dtype=torch.float16)
            parent_kernel[1, 1] = 0  # Don't include center cell
            
            # Find organisms adjacent to new cells
            parent_mask = torch.nn.functional.conv2d(
                selected_mask.unsqueeze(0).unsqueeze(0).to(torch.float16), 
                parent_kernel.unsqueeze(0).unsqueeze(0), 
                padding=1
            ).squeeze() > 0
            
            # Reduce energy of parent cells
            self.energy_matrix = torch.clamp(self.energy_matrix - parent_energy_reduction * parent_mask.float(), 0, 1)
            
            # Add selected positions to topology
            self.topology_matrix[selected_mask] = 1
    
    def compute_energy(self):
        """Deplete all energies by ENERGY_DECAY and allow energy sharing between adjacent cells"""
        # Apply energy decay only where topology = 1 (organisms)
        self.energy_matrix = torch.clamp(self.energy_matrix - ENERGY_DECAY * self.topology_matrix, 0, 1)
        
        # Remove organisms with energy below quantization step
        low_energy_mask = self.energy_matrix < QUANTIZATION_STEP
        self.topology_matrix[low_energy_mask] = 0
        
        
        # Energy sharing between adjacent cells (conserving total energy)
        sharing_rate = ENERGY_SHARING_RATE
        
        # Calculate how much energy each cell can share (only organisms can share)
        shareable_energy = self.energy_matrix * self.topology_matrix * sharing_rate
        
        # Create a 3x3 kernel for energy distribution (excluding center)
        kernel = torch.ones((3, 3), device=device, dtype=torch.float16)
        kernel[1, 1] = 0  # Don't include center cell
        
        # Distribute energy to adjacent cells
        distributed_energy = torch.nn.functional.conv2d(
            shareable_energy.unsqueeze(0).unsqueeze(0).to(torch.float16), 
            kernel.unsqueeze(0).unsqueeze(0), 
            padding=1
        ).squeeze()
        
        # Only organisms can receive energy
        receiving_mask = self.topology_matrix > 0
        
        # Store mask of where energy is shared but topology = 0 (new cell candidates)
        self.new_cell_candidates = (distributed_energy > 0) & (self.topology_matrix == 0)
        
        if torch.any(receiving_mask):
            # Calculate how much energy each cell can receive (proportional to available space)
            energy_deficit = 1.0 - self.energy_matrix
            total_deficit = torch.nn.functional.conv2d(
                energy_deficit.unsqueeze(0).unsqueeze(0).to(torch.float16), 
                kernel.unsqueeze(0).unsqueeze(0), 
                padding=1
            ).squeeze()
            
            # Avoid division by zero
            total_deficit = torch.clamp(total_deficit, min=1e-6)
            
            # Calculate energy to receive based on proportional deficit
            energy_to_receive = distributed_energy * (energy_deficit / total_deficit) * receiving_mask.float()
            
            # Remove shared energy from source cells
            self.energy_matrix = torch.clamp(self.energy_matrix - shareable_energy, 0, 1)
            
            # Add received energy to destination cells
            self.energy_matrix = torch.clamp(self.energy_matrix + energy_to_receive, 0, 1)
            
class Renderer:
    def __init__(self, world_size):
        self.world_size = world_size
        self.render_mode = "org_energy"  # "org_top" or "org_energy"
        self.texture_id = None
        self.quad_vbo = None
        self.opengl_initialized = False
    
    def toggle_render_mode(self):
        """Toggle between organism topology and energy visualization"""
        self.render_mode = "org_energy" if self.render_mode == "org_top" else "org_top"
        print(f"Render mode: {self.render_mode}")
    
    def render(self, environment, topology, mask, new_cell_candidates=None):
        """Render the current state using PyTorch tensors directly - GPU accelerated"""
        env_scaled = environment.clamp(0, 1)
        
        # Create RGB image tensor
        image = torch.zeros((3, self.world_size, self.world_size), device=device, dtype=torch.float16)
        
        # Apply environment to all channels
        image[0] = env_scaled  # Red channel
        image[1] = env_scaled  # Green channel  
        image[2] = env_scaled  # Blue channel
        
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
    
    def render_opengl(self):
        """Render using OpenGL (OpenGL 2.1 compatible)"""
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
        
        # Swap buffers
        glutSwapBuffers()
    

class Simulation:
    def __init__(self):
        self.world_size = WORLD_SIZE
        self.environment = Environment(self.world_size, NOISE_SCALE, QUANTIZATION_STEP)
        self.organism_manager = OrganismManager(self.world_size, ORGANISM_COUNT, self.environment.terrain)
        self.logger = Logger()
        self.tick = 0
    
    def update_simulation(self):
        """Update one simulation tick"""
        # DISABLE topology expansion (major CPU bottleneck)
        self.organism_manager.compute_topology()
        
        # Compute energy decay and sharing
        self.organism_manager.compute_energy()
        
        # Compute environment changes
        self.environment.compute_environment(self.organism_manager.topology_matrix)
        
        self.logger.update_fps()
       
        debug_info = self.logger.get_debug_info()
        if self.tick % DEBUG_PRINT_INTERVAL == 0:
            self.logger.log_tick(self.tick, 0, debug_info, None, None)
       
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
    

def main():
    """Main simulation loop with GPU-accelerated OpenGL rendering"""
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
    rendering_frequency = max(1, 60 // RENDERING_FPS)  # Assume 60 FPS base for calculation
    
    # Frame rate control variables
    last_frame_time = time.time()
    
    # Global variables for OpenGL callbacks
    global current_simulation, current_renderer, current_input_handler
    current_simulation = simulation
    current_renderer = renderer
    current_input_handler = input_handler
    
    def display():
        """OpenGL display callback - only handles rendering when window needs redraw"""
        global current_renderer
        
        # Only render if we have data
        if last_sim_data is not None:
            # Update OpenGL texture and render
            current_renderer.update_texture(current_renderer.last_image)
            current_renderer.render_opengl()
    
    def keyboard(key, x, y):
        """OpenGL keyboard callback"""
        global current_renderer
        
        if key == b'q' or key == 27:  # 'q' or ESC
            glut.glutLeaveMainLoop()
        elif key == b'm':
            current_renderer.toggle_render_mode()
    
    # Frame rate control - ONLY for rendering/OpenGL
    rendering_frame_counter = 0
    
    # Store last simulation data
    last_sim_data = None
    
    def idle():
        """OpenGL idle callback with proper frame rate limiting"""
        nonlocal rendering_frame_counter, last_sim_data, last_frame_time
        global current_simulation, current_renderer
        
        # Calculate time since last frame
        current_time = time.time()
        delta_time = current_time - last_frame_time
        
        # Update simulation at FULL SPEED - no frame limiting
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
