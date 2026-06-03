# Configuration constants for the organism simulation

# Seeding Configuration
WORLD_SIZE = 32
ORGANISM_COUNT = 32  # Seeds placed at random distinct cells on the grid


# Organism Configuration
INTERACTION_RATE = 0.1  # P(crossover/mutation) when birth touches a different genome_id
ENERGY_HARVEST_RATE = 0.05
ENERGY_DECAY = 0.0035
SHARING_ON_VALUE = 0.8
SHARING_OFF_VALUE = 0.2
REPRODUCTION_THRESHOLD = 0.05
DEATH_THRESHOLD = 0.05
ENERGY_SHARING_RATE = 1


# Fitness modes (evolution objective; interactive sim unchanged):
#   cell_count           - sum of alive cells each tick (default; growth-selected)
#   entropy_production   - sum of destroyed_energy gained per tick (dissipation-selected)
#   persistence          - ticks alive before extinction on uniform terrain (depletes over rollout)
#   life_like            - ∫σ̇ − λ·ΔS_config (dissipation + configurational order); optional σ̇ floor

# CNN Training Configuration
CNN_POPULATION_SIZE = 16 
CNN_MUTATION_RATE = 0.01
CNN_MUTATION_MAGNITUDE = 0.5
CNN_TRAINING_EPOCHS = 100
CNN_TRAINING_MAX_TIME = 400
CNN_FITNESS_MODE = "life_like"
CNN_FITNESS_PERSISTENCE_TERRAIN = 0.5 # Uniform terrain level for persistence mode
CNN_FITNESS_LIFE_LIKE_ORDER_WEIGHT = 1.0 # λ in fitness = ∫σ̇ − λ·ΔS_config
CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR = 0.0 # Minimum ∫σ̇ or life_like fitness is 0
TRAIN_HEADLESS = True
TRAIN_WORKER_COUNT = None
TRAIN_WORKER_MAX_TASKS = 1


# Environment Configuration
ENVIRONMENT_TYPE = 2  # 1 = center mask, 2 = perlin pump (refill depletion), 3 = moving perlin
ENV_NOISE_THRESHOLD = 0 # Terrain threshold for dead cells
NOISE_SCALE = 0.01 
QUANTIZATION_STEP = 0.01
NOISE_FREQUENCY_MULTIPLIER = 8
NOISE_OCTAVES = 6 
NOISE_POWER = 1
PERLIN_NOISE_SCALE = 0.05 
THERMO_ENV_TEMPERATURE = 1 # Thermodynamic decay (Onsager: sigma_dot = L * mu^2, ΔE = T * sigma_dot * dt)
ENTROPY_EPSILON = 1e-8
STARTING_POSITION_TERRAIN_BOOST = 10.0
TERRAIN_PUMP_ENABLED = True
TERRAIN_PUMP_RATE = 0.03  # Type 1 center refill
PERLIN_TIME_SPEED = 0.002  # Type 3 field morph
PUMP_PERLIN_TIME_SPEED = 0.0005  # Type 2 pump wave advance (faster = less visual lag)
PUMP_WAVE_RATE = 0.004  # Zero-mean perlin layer (× PID drive)
PUMP_MEAN_RATE = 0.06  # Uniform mean shift at |drive|=1 (population actuator)
PUMP_TRAIL_RELAX_RATE = 0.02  # Local heal: += rate × (base − terrain)+ when drive > 0
PUMP_MAX_CELL_DELTA = 0.01  # Cap per-cell terrain change per tick (limits peak/depression depth)
PUMP_POPULATION_FRACTION = 0.5  # Setpoint when PUMP_POPULATION_SETPOINT is None: fraction × n²
PUMP_POPULATION_SETPOINT = None  # None → ½ n²; or set explicit cell count
PUMP_PID_KP = 0.75
PUMP_PID_KI = 0.001
PUMP_PID_KD = 0.1
PUMP_PID_DEADBAND = 0.02
PUMP_PID_CLAMP = 0.7
PUMP_PID_INTEGRAL_CLAMP = 4.0
PUMP_DRIVE_SLEW = 0.12  # Faster drive tracking = less delayed response


# Rendering Configuration
RENDERING_FPS = 30   # Desired rendering FPS
RENDERING_BASE_FPS = 60  # Base FPS for frequency calculation
PIXEL_SCALE = 255
PIXEL_SCALE_FACTOR = int(32/WORLD_SIZE*20) # Factor to upscale pixels in OpenGL rendering
OPENGL_CLEAR_COLOR_R = 0.0
OPENGL_CLEAR_COLOR_G = 0.0
OPENGL_CLEAR_COLOR_B = 0.0
OPENGL_CLEAR_COLOR_A = 1.0
GPU_CACHE_CLEAR_INTERVAL = 10
DEVICE_TYPE = "mps"  # Preferred device type: "mps", "cuda", or "cpu" (will fallback if unavailable)
DEBUG_PRINT_INTERVAL = 10


