# Configuration constants for the organism simulation
import random
# World Configuration
WORLD_SIZE = 32
ORGANISM_COUNT = 1

# Environment Configuration
NOISE_SCALE = 0.01
QUANTIZATION_STEP = 0.01
NOISE_FREQUENCY_MULTIPLIER = 4
NOISE_OCTAVES = 6
NOISE_POWER = 2

# Organism Configuration
ENERGY_HARVEST_RATE = 0.01
ENERGY_DECAY = 0.005
# Coefficient for locality-based decay modulation (0 disables effect, 1 full strength)
ENERGY_DENSITY_DECAY_MODIFIER = 0.0
# Spawn organism at sine terrain peak by default
# For default NOISE_SCALE = 0.01, NOISE_FREQUENCY_MULTIPLIER = 3, peaks are at multiples of about 8
CENTER_X = WORLD_SIZE // 2
CENTER_Y = WORLD_SIZE // 2
ORGANISM_POSITIONS = [(CENTER_X, CENTER_Y) for i in range(ORGANISM_COUNT)]

# Reproduction Configuration
REPRODUCTION_THRESHOLD = 0.01
DEATH_THRESHOLD = 0.01

# Energy Configuration
ENERGY_SHARING_RATE = 0.5

# Terrain energy at the organism's starting position
STARTING_POSITION_TERRAIN_BOOST = 10.0


# Rendering Configuration
RENDERING_FPS = 30   # Desired rendering FPS
RENDERING_BASE_FPS = 60  # Base FPS for frequency calculation
PIXEL_SCALE = 255
PIXEL_SCALE_FACTOR = int(WORLD_SIZE/32*20) # Factor to upscale pixels in OpenGL rendering

# OpenGL Configuration
OPENGL_CLEAR_COLOR_R = 0.0
OPENGL_CLEAR_COLOR_G = 0.0
OPENGL_CLEAR_COLOR_B = 0.0
OPENGL_CLEAR_COLOR_A = 1.0

# Performance Configuration
GPU_CACHE_CLEAR_INTERVAL = 10

# Debug Configuration
DEBUG_PRINT_INTERVAL = 10

# CNN Training Configuration
CNN_POPULATION_SIZE = 8
CNN_MUTATION_RATE = 1
CNN_MUTATION_MAGNITUDE = 1
CNN_TRAINING_EPOCHS = 100
CNN_TRAINING_MAX_TIME = 100
CNN_FITNESS_EARLY_TERMINATION_THRESHOLD = 0.1
