from Sequential import Sequential
import math
import numpy as np
import time
import uuid

run_id = str(uuid.uuid1())[:4]


class GeneticAlgorithm:
    def __init__(self, popSize, mut_rate, mut_mag, bots_mut, model_file = ""):
        self.fittestIndex = 0
        self.popSize = popSize
        self.subjects = [Sequential() for _ in range(popSize)]
        try:
            self.load(model_file)
            print("Loaded "+model_file)
        except:
            print("Couldn't load "+model_file)
        time.sleep(1)

        self.mut_rate = mut_rate
        self.bots_mutated = bots_mut
        self.mutation_magnitude = mut_mag

    def reset(self):
         for i in range(len(self.subjects)):
            self.subjects[i].fitness = 0

    def compute_generation(self):
        self.calc_fittest()

        # SAVE TRAINED
        if self.subjects[self.fittestIndex].fitness > 0:
            file = open('data/'+run_id+'_'+str(self.subjects[self.fittestIndex].fitness)+'.csv', "w")
            file.write(str(self.subjects[self.fittestIndex].network))
            file.close()
        self.crossover(self.subjects[self.fittestIndex])
        self.mutate(self.mut_rate)

    def calc_fittest(self):
        index = 0
        temp = 0
        for count, i in enumerate(self.subjects):
            if i.fitness > temp:
                temp = i.fitness
                index = count
        self.fittestIndex = index

    def load(self, fileName):
        file = open(fileName, "r")
        layers = file.read().split('array(')[1:]
        network = []
        for layerIndex in range(len(layers)):
            data = layers[layerIndex][:-3]
            if (layerIndex == len(layers) -1):
                data += "]"
            layer = np.array(eval(data))
            network.append(layer)
        self.subjects[0].network = network

    def crossover(self, parent1):
        for bot in self.subjects:
            bot.network = parent1.network.copy()

    def mutate(self, mut_rate):
        for i in range(len(self.subjects)):
            if i >= len(self.subjects)-self.bots_mutated:
                for layerIndex in range(len(self.subjects[i].network)):

                    # Copy original net
                    original = self.subjects[i].network[layerIndex].copy()

                    # Create random samples
                    random_mask = np.random.random_sample((self.subjects[i].nodesByLayer[layerIndex], self.subjects[i].nodesByLayer[layerIndex+1]))
                    random_values = np.subtract(np.dot(np.random.random_sample((self.subjects[i].nodesByLayer[layerIndex], self.subjects[i].nodesByLayer[layerIndex+1])), 2), 1)

                    # Create mask of random indexes
                    m = np.ma.masked_where(mut_rate > random_mask, self.subjects[i].network[layerIndex])

                    # Assign random values to the masked indexes
                    original[np.where(m.mask)] = random_values[np.where(m.mask)]

                    # Set new net to the old copy
                    self.subjects[i].network[layerIndex] = original.copy()