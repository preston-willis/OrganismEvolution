from GeneticAlgorithm import GeneticAlgorithm
from Plotter import Plotter
from Renderer import Renderer
from Organism import Organism
import numpy as np
import uuid
import time
import collections
import threading

epochs = 1000
graphing_enabled = False
population = 100
max_time = 225
mutation_rate = 0.1
mutation_magnitude = 1
bots_mutated = 99
weight = 0.1
bounds = 10
degree_interval = 5

class EvolutionDriver:
    def __init__(self) -> None:
        if graphing_enabled: self.plt = Plotter(max_time)
        self.ga = GeneticAlgorithm(population, mutation_rate, mutation_magnitude, bots_mutated)
        self.rendering_enabled = False
        if self.rendering_enabled: self.pg_client = Renderer()

    # TRAIN MODEL
    def simulate(self, bot, gen):
        org = Organism([400, 400], self.ga.subjects[bot], 10000, 0.01, 0.5)
        eposx = [410, 390]
        eposy = [420, 380]
        tick = 0
        for time in range(max_time):
            tick += 1
            if time % 2 == 0 and time < max_time/2: 
                eposy[0] -= 1
                eposy[1] += 1
            elif time % 2 == 0 and time >= max_time/2: 
                eposx[0] -= 1
                eposx[1] += 1
            energy = generate_energy(org, time, [[eposx[0], eposy[0]], [eposx[1], eposy[1]]])
            org.tick(time, energy)
            if org.sum_energies() < 1:
                break
            if self.rendering_enabled == True:
                self.pg_client.reset()
                self.pg_client.render_cells(org, [[eposx[0], eposy[0]], [eposx[1], eposy[1]]])
                self.pg_client.handle_pygame()

            # PLOTTER
            if graphing_enabled == True:
                self.plt.plot(time, self.env.deg)

        self.ga.subjects[bot].fitness += (tick*org.members)
        org.kill()
        if self.rendering_enabled: self.pg_client.reset()  
        
        # SAVE TRAINED
        print("bot "+str(bot)+": "+str(self.ga.subjects[bot].fitness))
        if self.ga.subjects[bot].fitness > 100000:
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
                if ticks < 30:
                    food_energy[cell.x][cell.y] = 1
                else:
                    food_energy[cell.x][cell.y] = 0
                    for s in source_pos:
                        if cell.y == s[1] and cell.x == s[0]:
                            food_energy[cell.x][cell.y] = 1
                        elif cell.y < s[1] + 10 and cell.y > s[1] - 10 and cell.x < s[0] + 10 and cell.x > s[0] - 10:
                            food_energy[cell.x][cell.y] = (1/(np.abs(cell.y-s[1]+0.0001))+(np.abs(cell.x-s[0]+0.0001)))*0.001
                    
    return food_energy