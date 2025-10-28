# Configuration constants for the organism simulation
import random
# World Configuration
WORLD_SIZE = 1000
ORGANISM_COUNT = 40

# Environment Configuration
NOISE_SCALE = 0.01
QUANTIZATION_STEP = 0.01
NOISE_FREQUENCY_MULTIPLIER = 3
NOISE_OCTAVES = 6
NOISE_POWER = 4

# Organism Configuration
ENERGY_HARVEST_RATE = 0.02
ENERGY_DECAY = 0.0001
ORGANISM_POSITIONS = [(random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1)) for i in range(ORGANISM_COUNT)]

# Reproduction Configuration
REPRODUCTION_THRESHOLD = 0.001
DEATH_THRESHOLD = 0.001

# Energy Configuration
ENERGY_SHARING_RATE = 0.1


# Rendering Configuration
RENDERING_FPS = 30   # Desired rendering FPS
PIXEL_SCALE = 255

# OpenGL Configuration
OPENGL_CLEAR_COLOR_R = 0.0
OPENGL_CLEAR_COLOR_G = 0.0
OPENGL_CLEAR_COLOR_B = 0.0
OPENGL_CLEAR_COLOR_A = 1.0

# Performance Configuration
GPU_CACHE_CLEAR_INTERVAL = 10

# Debug Configuration
DEBUG_PRINT_INTERVAL = 10
