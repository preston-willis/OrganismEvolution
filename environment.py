import config
import numpy as np
import torch
from noise import pnoise3
from runtime import get_device


class Environment:
    def __init__(self, world_size, organism_seed_positions):
        self.world_size = world_size
        self.organism_seed_positions = organism_seed_positions
        self.environment_type = config.ENVIRONMENT_TYPE
        self.time = 0.0
        self.pump_time = 0.0
        self._reset_pump_controller()
        self.terrain = self.generate_terrain()

    def reset_perlin(self):
        self.time = 0.0
        self.pump_time = 0.0
        self._reset_pump_controller()
        self.terrain = self.generate_terrain()

    def generate_terrain(self):
        if self.environment_type == 1:
            return self._generate_energy_mask_terrain()
        if self.environment_type == 2:
            self.base_raw = self._sample_perlin_octaves(0.0)
            self.base_terrain = self._perlin_to_terrain(self.base_raw)
            return self.base_terrain.clone()
        if self.environment_type == 3:
            return self._generate_perlin_terrain()
        return self._empty_terrain()

    def _center_3x3_mask(self):
        device = get_device()
        cy = self.world_size // 2
        cx = self.world_size // 2
        mask = torch.zeros((self.world_size, self.world_size), dtype=torch.bool, device=device)
        mask[cy - 1 : cy + 2, cx - 1 : cx + 2] = True
        return mask

    def _generate_energy_mask_terrain(self):
        device = get_device()
        terrain = torch.zeros((self.world_size, self.world_size), dtype=torch.float32, device=device)
        terrain[self._center_3x3_mask()] = config.STARTING_POSITION_TERRAIN_BOOST
        return torch.minimum(terrain, torch.ones_like(terrain))

    def _empty_terrain(self):
        device = get_device()
        return torch.zeros((self.world_size, self.world_size), dtype=torch.float32, device=device)

    def _sample_perlin_octaves(self, time_value):
        x = torch.arange(self.world_size, dtype=torch.float32)
        y = torch.arange(self.world_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx_np = xx.numpy()
        yy_np = yy.numpy()
        noise_values = np.zeros((self.world_size, self.world_size), dtype=np.float32)
        for octave in range(config.NOISE_OCTAVES):
            octave_time = time_value * (1.0 + octave * 0.3) + octave * 5.0
            octave_scale = config.PERLIN_NOISE_SCALE * (2.0**octave)
            octave_noise = np.zeros((self.world_size, self.world_size), dtype=np.float32)
            for i in range(self.world_size):
                for j in range(self.world_size):
                    octave_noise[i, j] = pnoise3(
                        xx_np[i, j] * octave_scale,
                        yy_np[i, j] * octave_scale,
                        octave_time,
                        octaves=1,
                        base=config.PERLIN_SEED + octave * 10,
                    )
            noise_values += octave_noise / (2.0**octave)
        return torch.from_numpy(noise_values).to(get_device())

    def _perlin_to_terrain(self, raw):
        values = (raw + 1.0) * 0.5
        values = torch.pow(values, config.NOISE_POWER)
        dead_mask = values > config.ENV_NOISE_THRESHOLD
        values = values * dead_mask
        return torch.clamp(values * config.TERRAIN_ENERGY_SCALE, 0, 1)

    def _generate_perlin_terrain_at(self, time_value):
        return self._perlin_to_terrain(self._sample_perlin_octaves(time_value))

    def _generate_perlin_terrain(self):
        return self._generate_perlin_terrain_at(self.time)

    def _reset_pump_controller(self):
        self._pump_pid_integral = 0.0
        self._pump_pid_prev_error = 0.0
        self._pump_drive = 0.0
        self._pump_osc_phase = 0.0

    def _centered_pump_wave(self):
        raw = self._sample_perlin_octaves(self.pump_time)
        self.pump_time += config.PUMP_PERLIN_TIME_SPEED
        return raw - raw.mean() + config.PUMP_WAVE_BIAS

    def _population_setpoint(self):
        if config.PUMP_POPULATION_SETPOINT is not None:
            return float(config.PUMP_POPULATION_SETPOINT)
        return config.PUMP_POPULATION_FRACTION * self.world_size * self.world_size

    def _pump_population_error(self, organism_manager):
        alive = organism_manager.topology_matrix.sum().item()
        setpoint = self._population_setpoint()
        error = (setpoint - alive) / max(setpoint, 1.0)
        if error > 1.0:
            error = 1.0
        elif error < -1.0:
            error = -1.0
        if error > config.PUMP_PID_DEADBAND:
            error -= config.PUMP_PID_DEADBAND
        elif error < -config.PUMP_PID_DEADBAND:
            error += config.PUMP_PID_DEADBAND
        else:
            error = 0.0
        return error

    def _pump_pid_step(self, error):
        self._pump_pid_integral += error
        if self._pump_pid_integral > config.PUMP_PID_INTEGRAL_CLAMP:
            self._pump_pid_integral = config.PUMP_PID_INTEGRAL_CLAMP
        elif self._pump_pid_integral < -config.PUMP_PID_INTEGRAL_CLAMP:
            self._pump_pid_integral = -config.PUMP_PID_INTEGRAL_CLAMP
        derivative = error - self._pump_pid_prev_error
        self._pump_pid_prev_error = error
        raw = (
            config.PUMP_PID_KP * error
            + config.PUMP_PID_KI * self._pump_pid_integral
            + config.PUMP_PID_KD * derivative
        )
        if raw > config.PUMP_PID_CLAMP:
            if error > 0.0:
                self._pump_pid_integral -= error
            raw = config.PUMP_PID_CLAMP
        elif raw < -config.PUMP_PID_CLAMP:
            if error < 0.0:
                self._pump_pid_integral -= error
            raw = -config.PUMP_PID_CLAMP
        delta = raw - self._pump_drive
        if delta > config.PUMP_DRIVE_SLEW:
            delta = config.PUMP_DRIVE_SLEW
        elif delta < -config.PUMP_DRIVE_SLEW:
            delta = -config.PUMP_DRIVE_SLEW
        self._pump_drive += delta
        if config.PUMP_PID_OSCILLATION_AMPLITUDE > 0.0:
            self._pump_osc_phase += 2.0 * np.pi / config.PUMP_PID_OSCILLATION_PERIOD
            self._pump_drive += config.PUMP_PID_OSCILLATION_AMPLITUDE * np.sin(self._pump_osc_phase)
            if self._pump_drive > config.PUMP_PID_CLAMP:
                self._pump_drive = config.PUMP_PID_CLAMP
            elif self._pump_drive < -config.PUMP_PID_CLAMP:
                self._pump_drive = -config.PUMP_PID_CLAMP
        return self._pump_drive

    def _apply_thermodynamic_pump(self, organism_manager, terrain_debit=None):
        if organism_manager is None:
            return
        error = self._pump_population_error(organism_manager)
        drive = self._pump_pid_step(error)
        cell_count = self.world_size * self.world_size
        wave = drive * config.PUMP_WAVE_RATE * self._centered_pump_wave()
        wave = torch.clamp(wave, -config.PUMP_MAX_CELL_DELTA, config.PUMP_MAX_CELL_DELTA)
        trail = torch.zeros_like(self.terrain)
        if drive > 0.0:
            deficit = torch.clamp(self.base_terrain - self.terrain, min=0.0)
            trail = drive * config.PUMP_TRAIL_RELAX_RATE * deficit
        mean_drain = torch.zeros_like(self.terrain)
        if drive < 0.0:
            mean_drain = drive * config.PUMP_MEAN_RATE / cell_count
        self.terrain += wave + trail + mean_drain

    def _advance_moving_perlin(self):
        field = self._generate_perlin_terrain_at(self.time)
        self.time += config.PERLIN_TIME_SPEED
        return field

    def compute_environment(self, topology_matrix, harvested_energy, organism_manager=None):
        self.terrain.copy_(torch.clamp(self.terrain - harvested_energy, 0, 1))
        if self.environment_type == 3:
            self.terrain.copy_(self._advance_moving_perlin())
        elif self.environment_type == 2 and config.TERRAIN_PUMP_ENABLED:
            self.base_terrain = self._generate_perlin_terrain_at(self.time)
            self.time += config.PERLIN_TIME_SPEED
            self._apply_thermodynamic_pump(organism_manager, harvested_energy)
        if self.environment_type == 1:
            center_mask = self._center_3x3_mask()
            self.terrain[center_mask] = torch.clamp(
                self.terrain[center_mask] + config.TERRAIN_PUMP_RATE,
                max=1.0,
            )
        self.terrain.clamp_(0.0, 1.0)
