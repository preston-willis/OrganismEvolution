import gc
import glob
import multiprocessing
import os

import config
import torch

from config import (
    CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR,
    CNN_FITNESS_LIFE_LIKE_ORDER_WEIGHT,
    CNN_FITNESS_MODE,
    CNN_FITNESS_PERSISTENCE_TERRAIN,
    CNN_MUTATION_MAGNITUDE,
    CNN_MUTATION_RATE,
    CNN_POPULATION_SIZE,
    CNN_TRAINING_EPOCHS,
    CNN_TRAINING_MAX_TIME,
    TRAIN_WORKER_COUNT,
    TRAIN_WORKER_MAX_TASKS,
    WORLD_SIZE,
)
from cnn import CNNGeneticAlgorithm, EnergyDistributionCNN
from entropy import configurational_entropy
from runtime import get_device, set_device
from simulation import Simulation

CNN_FITNESS_MODES = ("cell_count", "entropy_production", "persistence", "life_like")

_worker_device = None


def _init_worker(device_str):
    global _worker_device
    from config import DEVICE_TYPE

    if device_str.startswith("cuda"):
        _worker_device = torch.device(device_str)
    elif device_str == DEVICE_TYPE:
        _worker_device = torch.device(device_str)
    else:
        _worker_device = torch.device("cpu")
    set_device(_worker_device)


def _release_device_memory(torch_device):
    gc.collect()
    if torch_device.type == "mps":
        torch.mps.empty_cache()
    elif torch_device.type == "cuda":
        torch.cuda.empty_cache()


def validate_cnn_fitness_mode(mode):
    if mode not in CNN_FITNESS_MODES:
        raise ValueError(f"Unknown fitness mode {mode!r}; expected one of {CNN_FITNESS_MODES}")


def set_cnn_fitness_mode(mode):
    import config

    if mode is None:
        mode = config.CNN_FITNESS_MODE
    validate_cnn_fitness_mode(mode)
    config.CNN_FITNESS_MODE = mode


def cnn_fitness_mode_label(mode=None):
    mode = config.CNN_FITNESS_MODE if mode is None else mode
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
            org_energy = torch.sum(om.energy_matrix).item()
            env_energy = torch.sum(sim.environment.terrain).item()
            total = org_energy + env_energy
            tick_data.append((t, cumulative_cell_count, cell_count, org_energy, env_energy, total))
        if cell_count == 0:
            break
    if mode == "life_like":
        entropy_produced = om.destroyed_entropy - entropy_produced_start
        config_entropy_end = configurational_entropy(om, sim.environment.terrain)
        delta_config_entropy = config_entropy_end - config_entropy_start
        fitness = entropy_produced - CNN_FITNESS_LIFE_LIKE_ORDER_WEIGHT * delta_config_entropy.item()
        if fitness < 0.0:
            fitness = 0.0
        if CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR > 0.0:
            if entropy_produced < CNN_FITNESS_LIFE_LIKE_ENTROPY_FLOOR:
                fitness = 0.0
    return fitness, cumulative_cell_count, tick_data


def _evaluate_cnn_worker(args):
    global _worker_device
    cnn_state_dict, world_size, max_time, bot_index, collect_tick_data, fitness_mode = args
    cnn = EnergyDistributionCNN(_worker_device)
    cnn.load_state_dict(cnn_state_dict)
    cnn._zero_hidden_channel_bias()
    sim = Simulation(enable_debug=False)
    sim.organism_manager.energy_distribution_cnn = cnn
    fitness, _, tick_data = run_cnn_fitness_rollout(sim, max_time, collect_tick_data, fitness_mode)
    del cnn
    del sim
    _release_device_memory(_worker_device)
    return (fitness, tick_data)


class CNNEvaluator:
    def __init__(self, world_size, max_time, device, fitness_mode=None):
        self.world_size = world_size
        self.max_time = max_time
        self.device = device
        self.fitness_mode = config.CNN_FITNESS_MODE if fitness_mode is None else fitness_mode
        validate_cnn_fitness_mode(self.fitness_mode)
        self.grapher = None
        self.current_generation_max_fitness = 0.0
        self.pool = None

    def _ensure_pool(self):
        if self.pool is None:
            device_str = str(self.device)
            pool_kwargs = {
                "processes": TRAIN_WORKER_COUNT,
                "initializer": _init_worker,
                "initargs": (device_str,),
            }
            if TRAIN_WORKER_MAX_TASKS is not None:
                pool_kwargs["maxtasksperchild"] = TRAIN_WORKER_MAX_TASKS
            self.pool = multiprocessing.Pool(**pool_kwargs)

    def close_pool(self):
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
        self.close_pool()

    def evaluate_population(self, subjects, pop_size):
        self._ensure_pool()
        collect_tick_data = self.grapher is not None
        args_list = []
        for i in range(pop_size):
            state_dict = subjects[i].state_dict()
            cpu_state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
            args_list.append(
                (cpu_state_dict, self.world_size, self.max_time, i, collect_tick_data, self.fitness_mode)
            )
        results = self.pool.map(_evaluate_cnn_worker, args_list)
        if self.grapher is not None:
            fitness_scores = []
            for i, (fitness, tick_data) in enumerate(results):
                fitness_scores.append(fitness)
                for t, total_cell_count, current_cell_count, org_energy, env_energy, total in tick_data:
                    self.grapher.enqueue_tick(
                        t, self.current_generation_max_fitness, [total_cell_count], org_energy, env_energy, total
                    )
                    self.grapher.enqueue_bot_tick(i, t, total_cell_count, current_cell_count, env_energy, total)
                try:
                    self.grapher.process_queued()
                except Exception:
                    pass
        else:
            fitness_scores = [fitness for fitness, _ in results]
        del args_list
        del results
        _release_device_memory(self.device)
        return fitness_scores

    def _evaluate_single_cnn(self, simulation, bot_index=None):
        collect_tick_data = self.grapher is not None
        fitness, cumulative_cell_count, tick_data = run_cnn_fitness_rollout(
            simulation, self.max_time, collect_tick_data, self.fitness_mode
        )
        if self.grapher is not None:
            for t, total_cell_count, current_cell_count, org_energy, env_energy, total in tick_data:
                self.grapher.enqueue_tick(
                    t, self.current_generation_max_fitness, [fitness], org_energy, env_energy, total
                )
                if bot_index is not None:
                    self.grapher.enqueue_bot_tick(
                        bot_index, t, fitness, current_cell_count, env_energy, total
                    )
        return fitness


class CNNEvolutionDriver:
    def __init__(self, world_size, epochs=100, max_time=100, fitness_mode=None):
        self.world_size = world_size
        self.epochs = epochs
        self.max_time = max_time
        self.fitness_mode = config.CNN_FITNESS_MODE if fitness_mode is None else fitness_mode
        validate_cnn_fitness_mode(self.fitness_mode)
        self.ga = CNNGeneticAlgorithm(CNN_POPULATION_SIZE, CNN_MUTATION_RATE, CNN_MUTATION_MAGNITUDE, get_device())
        self.evaluator = CNNEvaluator(world_size, max_time, get_device(), self.fitness_mode)
        self.grapher = None
        self.replay_simulation = None

    def evaluate_cnn(self, cnn, simulation):
        test_sim = Simulation(enable_debug=False)
        test_sim.organism_manager.energy_distribution_cnn = cnn
        fitness, _, _ = run_cnn_fitness_rollout(test_sim, self.max_time, fitness_mode=self.fitness_mode)
        return fitness

    def create_replay_simulation(self, best_cnn):
        self.replay_simulation = Simulation(enable_debug=False)
        self.replay_simulation.organism_manager.energy_distribution_cnn = best_cnn
        print(
            f"Created replay simulation (fitness/{cnn_fitness_mode_label()}: "
            f"{self.ga.fitness_scores[self.ga.fittest_index]:.6f})"
        )

    def run_evolution(self):
        print(f"\nStarting CNN Evolution - {self.epochs} generations")
        print(f"Population size: {CNN_POPULATION_SIZE}")
        print(f"Fitness mode: {cnn_fitness_mode_label(self.fitness_mode)} ({self.fitness_mode})")
        print(f"Using {TRAIN_WORKER_COUNT} worker processes (maxtasksperchild={TRAIN_WORKER_MAX_TASKS})")
        try:
            for gen in range(self.epochs):
                print(f"\nGeneration {gen + 1}/{self.epochs}")
                if self.grapher is not None:
                    self.evaluator.grapher = self.grapher
                fitness_scores = self.evaluator.evaluate_population(self.ga.subjects, CNN_POPULATION_SIZE)
                self.ga.fitness_scores = fitness_scores
                fitness_label = cnn_fitness_mode_label(self.fitness_mode)
                for i, fitness in enumerate(fitness_scores):
                    print(f"CNN {i}: fitness ({fitness_label}) = {fitness:.6f}")
                self.ga.compute_generation()
                best_fitness = self.ga.fitness_scores[self.ga.fittest_index]
                print(f"Best fitness ({fitness_label}): {best_fitness:.6f}")
                self.ga.save_model(self.ga.fittest_index, generation=gen + 1)
                if self.grapher is not None:
                    self.create_replay_simulation(self.ga.subjects[self.ga.fittest_index])
                    if not hasattr(self, "best_history"):
                        self.best_history = []
                    self.best_history.append(best_fitness)
                    self.evaluator.current_generation_max_fitness = best_fitness
                    self.grapher.enqueue_generation(gen + 1, self.best_history, fitness_scores)
                    self.grapher.process_queued()
                    self.grapher.reset_tick_metrics()
                self.ga.reset_fitness()
        finally:
            self.evaluator.close_pool()
        print("\nEvolution completed!")
        return self.ga.subjects[self.ga.fittest_index]
