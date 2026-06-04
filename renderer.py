import numpy as np
import torch
import OpenGL.GL as gl
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from OpenGL.GL import glRasterPos2f, glCallLists

from config import (
    CNN_HIDDEN_CHANNELS,
    THERMO_CONDUCTANCE,
    NOISE_OCTAVES,
    NOISE_POWER,
    ORGANISM_COUNT,
    PERLIN_NOISE_SCALE,
    PIXEL_SCALE,
    PIXEL_SCALE_FACTOR,
    REPRODUCTION_THRESHOLD,
    DEATH_THRESHOLD,
    SEED_ORGANISM_ENERGY,
    STARTING_POSITION_TERRAIN_BOOST,
    THERMO_CONDUCTANCE,
)
from runtime import get_device

device = get_device()

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
        self.hidden_channels_view_enabled = True
        self.hidden_channel_0_view_enabled = True
        self.coupling_view_enabled = False
        self.debug_panel_mode = "stats"
    
    def toggle_coupling_view(self):
        """Tab: RGB = frequency bins 0,1,2 (CNN hidden coupling bits)."""
        self.coupling_view_enabled = not self.coupling_view_enabled
        if self.coupling_view_enabled:
            self.hidden_channels_view_enabled = True
        self.debug_panel_mode = "coupling" if self.coupling_view_enabled else "stats"
        print(f"Coupling view (Tab): {'ON' if self.coupling_view_enabled else 'OFF'}")
    
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
    
    def toggle_hidden_channels_view(self):
        """Toggle RGB hidden-channel overlay on organisms."""
        self.hidden_channels_view_enabled = not self.hidden_channels_view_enabled
        print(f"Hidden channels view: {'ON' if self.hidden_channels_view_enabled else 'OFF'}")

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
                glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(char))
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
        if panel_mode == "coupling":
            header_lines = [
                f"FPS: {actual_fps:.1f}",
                "",
                "=== COUPLING (Tab) ===",
                "RGB = frequency bins 0, 1, 2 (CNN hidden)",
                "Domain bin 0 forced on in physics",
                "Dissipation = destructive interference (1-alpha)",
            ]
            for i, line in enumerate(header_lines):
                glColor4f(0.0, 1.0, 0.0, 1.0)
                self.render_text(text_x, text_y + (i * line_height), line)
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
            f"Coupling view (Tab): {'ON' if self.coupling_view_enabled else 'OFF'}",
            f"Hidden overlay (v): {'ON' if self.hidden_channels_view_enabled else 'OFF'}",
            f"Org Energy View (b): {'ON' if self.org_energy_view_enabled else 'OFF'}",
            f"Filters (n): {'ON' if self.filters_enabled else 'OFF'}",
            f"Harvesting (h): {'ON' if current_harvest_rate > 0 else 'OFF'}",
            f"",
            f"=== ENVIRONMENT CONFIG ===",
            f"Noise Octaves: {NOISE_OCTAVES}",
            f"Noise Power: {NOISE_POWER}",
            f"Perlin Scale: {PERLIN_NOISE_SCALE}",
            f"",
            f"=== ORGANISM CONFIG ===",
            f"Seed Count: {ORGANISM_COUNT}",
            f"Energy Harvest Rate: {THERMO_CONDUCTANCE:.4f}",
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
        hidden_channels=None,
        coupling_view=False,
    ):
        """Render the current state using PyTorch tensors directly - GPU accelerated"""
        env_scaled = environment.clamp(0, 1)
        
        # Create RGBA image tensor
        image = torch.zeros((4, self.world_size, self.world_size), device=device, dtype=torch.float32)
        
        org = topology > 0

        if coupling_view and self.filters_enabled and hidden_channels is not None:
            # Tab: full-saturation R/G/B = frequency bins 0, 1, 2 (no terrain grey wash)
            h = hidden_channels * org.unsqueeze(0).float()
            image[0] = torch.where(org, h[0], torch.zeros_like(env_scaled))
            image[1] = torch.where(
                org,
                h[1] if h.shape[0] > 1 else torch.zeros_like(env_scaled),
                torch.zeros_like(env_scaled),
            )
            image[2] = torch.where(
                org,
                h[2] if h.shape[0] > 2 else torch.zeros_like(env_scaled),
                torch.zeros_like(env_scaled),
            )
            image[3] = 1.0
            return image

        # Default: terrain on G/B
        image[0] = 0.0
        image[1] = env_scaled
        image[2] = env_scaled
        image[3] = 1.0

        if self.filters_enabled and hidden_channels is not None and self.hidden_channels_view_enabled:
            h = hidden_channels * org.unsqueeze(0).float()
            image[0] = torch.where(org, h[0], image[0])
            image[1] = torch.where(
                org,
                h[1] if h.shape[0] > 1 else torch.zeros_like(env_scaled),
                image[1],
            )
            image[2] = torch.where(
                org,
                h[2] if h.shape[0] > 2 else torch.zeros_like(env_scaled),
                image[2],
            )
            if self.render_mode == "org_energy" and self.org_energy_view_enabled:
                image[3] = org.float() * torch.clamp(mask, 0.1, 1) + (~org).float() * env_scaled

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
    
