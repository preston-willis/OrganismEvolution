from GeneticAlgorithm import GeneticAlgorithm
from Plotter import Plotter
from Renderer import Renderer
from Organism import Organism
import numpy as np
import uuid
import time
import collections
import threading
from copy import deepcopy

epochs = 1000
graphing_enabled = False
population = 10
max_time = 500
mutation_rate = 0.5
mutation_magnitude = 1
bots_mutated = population - 1
weight = 0.5
bounds = 10
degree_interval = 5

class EvolutionDriver:
    def __init__(self) -> None:
        if graphing_enabled: self.plt = Plotter(max_time)
        self.rendering_enabled = True
        if self.rendering_enabled: 
            self.pg_client = Renderer()
            global population
            population = 1
            global bots_mutated
            bots_mutated = 0
        self.ga = GeneticAlgorithm(population, mutation_rate, mutation_magnitude, bots_mutated)

    # TRAIN MODEL
    def simulate(self, bot, gen):
        org = Organism([400, 400], self.ga.subjects[bot], 10000, 0.01, 0)
        eposx = [400, 400]
        eposy = [370, 400]
        tick = 0
        for time in range(max_time):
            tick += 1
            energy = generate_energy(org, time, [[eposx[0], eposy[0]],[eposx[1], eposy[1]]])
            org.tick(time, energy)
            if org.members <= 0:
                break
            if self.rendering_enabled == True:
                self.pg_client.reset()
                self.pg_client.render_cells(org, [[eposx[0], eposy[0]],[eposx[1], eposy[1]]])
                self.pg_client.handle_pygame()

            # PLOTTER
            if graphing_enabled == True:
                self.plt.plot(time, self.env.deg)

        self.ga.subjects[bot].fitness += (tick*(1+org.members)+org.sum_energies())
        print("bot "+str(bot)+": "+str(org.members)+" "+str(org.members*org.sum_energies())+" "+str(tick))
        org.kill()
        if self.rendering_enabled: self.pg_client.reset()  
        
        # SAVE TRAINED
        if self.ga.subjects[bot].fitness > 10000000:
            file = open('trained_model_'+str(uuid.uuid1())+'.csv', "w")
            file.write(str(self.ga.subjects[bot].network))
            file.close()
            return 
        # SAVE TIMEOUT
        if gen == epochs-1:
            file = open('timeout_model_'+str(uuid.uuid1())+'.csv', "w")
            file.write(str(self.ga.subjects[bot].network))
            file.close()
            return

    def run(self):
        for gen in range(epochs):
            print("\nGEN: "+str(gen))
            print()
            # LOOP THROUGH BOTS

            for i in range(self.ga.popSize):
                thread = threading.Thread(target=self.simulate, args=[i, gen])
                thread.start()

            while threading.active_count() > 1:
                time.sleep(0.5)

            print()
            print("thread_count: "+str(threading.active_count()))

            # RESET
            self.ga.compute_generation()
            print()
            print("FITTEST: "+str(self.ga.fittestIndex)+": "+str(self.ga.subjects[self.ga.fittestIndex].fitness))
            print("NETWORK: "+str(self.ga.subjects[self.ga.fittestIndex].network))
            print("-----")
            self.ga.reset()

def generate_energy(org, ticks, source_pos):
    food_energy = collections.defaultdict(dict)
    for idx in list(org.positions):
            for idy in list(org.positions[idx]):
                cell = org.positions[idx][idy]
                if cell.y not in food_energy[cell.x]:
                    food_energy[cell.x][cell.y] = 0
                if ticks < 0:
                    food_energy[cell.x][cell.y] += 1
                if ticks >= 0:
                    food_energy[cell.x][cell.y] += 0
                    for s in source_pos:
                        if cell.y == s[1] and cell.x == s[0]:
                            food_energy[cell.x][cell.y] += 1
                        elif cell.y < s[1] + 5 and cell.y > s[1] - 5 and cell.x < s[0] + 5 and cell.x > s[0] - 5:
                            food_energy[cell.x][cell.y] += 1
                        elif cell.y < s[1] + 15 and cell.y > s[1] - 15 and cell.x < s[0] + 15 and cell.x > s[0] - 15:
                            food_energy[cell.x][cell.y] += 0.1
                    
    return food_energy