from GeneticAlgorithm import GeneticAlgorithm
from Plotter import Plotter
from Organism import Organism
import numpy as np
import time
import collections
import threading
from RenderThread import RenderThread
import queue
from multiprocessing import Pool, freeze_support
from itertools import repeat
import random

# CONFIGURATION
epochs = 1000
max_time = 100
graphing_enabled = True
model_file = "data\\6e23_103.23271445473337.csv"

# GENETIC ALGORITHM
population = 0
if graphing_enabled:
    population = 1
    max_time = 2000
else:
    population = 10
bots_mutated = population-1
mutation_rate = 0.1
mutation_magnitude = 1

# ORGANISM
org_x, org_y = (10, 10)
decay = 0.1
cell_limit = 10000

# ENVIRONMENT
if graphing_enabled:
    include_energy_ratio = 0.2
else:
    include_energy_ratio = 1
env_scale = 10

class EnergySource:
    def __init__(self, x, y):
        self.energy = 1
        self.x = x
        self.y = y


class EvolutionDriver:
    def __init__(self) -> None:
        self.ga = GeneticAlgorithm(
            population, mutation_rate, mutation_magnitude, bots_mutated, model_file)
        if graphing_enabled:
            self.q = queue.Queue()
            self.render_thread = RenderThread(self.q)
            self.render_thread.start()

    # TRAIN MODEL\

    def simulate(self, bot):
        orgSpawned = False
        org = None
        org_xpos = 0
        org_ypos = 0
        fitness = 0
        padding = (300)+int(env_scale/2)
        epos = []
        for i in range(int((800/env_scale)-(600/env_scale))):
            for p in range(int((800/env_scale)-(600/env_scale))):
                y = padding+(env_scale*i)
                x = padding+(env_scale*p)
                if random.random() > include_energy_ratio:
                    continue
                if not orgSpawned and i >= org_x and p >= org_y:
                    org = Organism([x, y], bot, cell_limit, decay, 0.5)
                    org_xpos = x
                    org_ypos = y
                    orgSpawned = True

                epos.append(EnergySource(x, y))

        energy_sum = 0
        for t in range(max_time):
            start_time = time.time()
            energy = generate_energy(org, t, epos)
            org.tick(t, energy)
            energy_sum += org.sum_energies()
            if org.sum_energies() < 0.001:
                break
            if graphing_enabled:
                self.q.put((org, epos))
            # print(time.time() - start_time)

        fitness = org.sum_energies()

        org.kill()

        return fitness

    def run(self):
        print("\nGEN: 0")
        for gen in range(epochs):
            # LOOP THROUGH BOTS

            if graphing_enabled:
                for i in range(population):
                    time.sleep(2)
                    fitness = self.simulate(self.ga.subjects[i])
                    self.q.put(())
                    print("bot "+str(i)+": "+str(round(fitness)))
            else:
                pool = Pool(4)
                fitness = pool.starmap(self.simulate, zip(
                    [self.ga.subjects[i] for i in range(population)]))
                for i, val in enumerate(fitness):
                    self.ga.subjects[i].fitness = val
                    print("bot "+str(i)+": "+str(round(val)))

            # RESET
            self.ga.compute_generation()
            self.print_summary(gen)
            self.ga.reset()

    def print_summary(self, gen):
        if not graphing_enabled:
            print("FITTEST: "+str(self.ga.fittestIndex)+": " +
                  str(round(self.ga.subjects[self.ga.fittestIndex].fitness)))
            print("NETWORK: " +
                  str(self.ga.subjects[self.ga.fittestIndex].network))
            print("-----")
            print()
            print("GEN: "+str(gen+1))
            print()


def generate_energy(org, ticks, source_pos):
    food_energy = collections.defaultdict(dict)
    for idx in list(org.positions):
        for idy in list(org.positions[idx]):
            cell = org.positions[idx][idy]
            if ticks < 1:
                food_energy[cell.x][cell.y] = 1
            else:
                food_energy[cell.x][cell.y] = 0
                for s in source_pos:
                    if s.energy > 0:
                        if cell.y > s.y-5 and cell.y < s.y+5 and cell.x < s.x+5 and cell.x > s.x-5:
                            transfer = (1/(np.abs(cell.y-s.y+0.0001)) +
                                        (np.abs(cell.x-s.x+0.0001)))*0.02
                            food_energy[cell.x][cell.y] = transfer
                            s.energy -= transfer*0.00012
                        if s.energy < 0:
                            s.energy = 0
    return food_energy
