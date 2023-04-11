import numpy as np
import uuid
import collections
import time
from copy import deepcopy
import threading


class Organism:
    def __init__(self, origin, dna, cell_limit, energy_decay, replication_cost) -> None:
        self.cell_limit = cell_limit
        self.energy_decay = energy_decay
        self.origin = origin
        self.originCell = Cell(origin[0], origin[1], uuid.uuid1(), (400, 400))
        self.originCell.ableToReplicate = [1,0,0,0]
        self.positions = collections.defaultdict(dict)
        self.members = 0
        self.positions[self.originCell.x][self.originCell.y] = self.originCell
        self.previous_energy_matrix = []
        self.dna = dna
        self.replication_cost = 0.5

    def tick(self, ticks, food_energy):
        energies = self.compute_energies(deepcopy(food_energy), ticks)
        updated_positions = deepcopy(self.positions)
        self.compute_cells(updated_positions, energies)

    def compute_cells(self, cells, energies):
        for idx in list(self.positions):
            for idy in list(self.positions[idx]):
                cell = cells[idx][idy]
                newCells = cell.tick(energies[cell.x][cell
                                                      .y], self.energy_decay, self.replication_cost)
                if not cell.dead and newCells is not None:
                    if len(newCells) > 0:
                        if self.members < self.cell_limit:
                            for newCell in newCells:
                                self.members += 1
                                cells[newCell.x][newCell.y] = newCell
                if cell.dead: 
                    self.members -= 1
                    cells[cell.x].pop(cell.y)
        self.positions = cells

    def compute_energies(self, energy_matrix, ticks):
        if ticks == 0:
            self.previous_energy_matrix = energy_matrix
            return self.previous_energy_matrix
        elif ticks > 0:
            new_energy_matrix = energy_matrix
            if self.members > 0:
                threads = []
                for idx in list(self.previous_energy_matrix):
                    self.distribute_energies(idx, new_energy_matrix)

                self.previous_energy_matrix = new_energy_matrix
            else: return energy_matrix
            return new_energy_matrix
    
    def distribute_energies(self, idx, new_energy_matrix):
        for idy in list(self.previous_energy_matrix[idx]):
            cell_energy = self.previous_energy_matrix[idx][idy]
            if idx in self.positions:
                if idy in self.positions[idx]:

                    self.calculate_replication_possibilities(idx, idy, new_energy_matrix)
                    self_energy = self.positions[idx][idy].energy
                    peer_energies = self.calculate_peer_energies(idx, idy)

                    input = np.concatenate([peer_energies, [self_energy]])

                    # Bound from 0:1
                    distributed_energies = self.compute(input, idx, idy)
                    distributed_energies = np.multiply(distributed_energies, 0.95)
                    distributed_energies = self.orient_data(distributed_energies, idx, idy, -1)
                    self.positions[idx][idy].replication_energy = distributed_energies

                    self.exchange_energies(new_energy_matrix, cell_energy, distributed_energies[0], idx, idy, idx+1, idy+1)
                    self.exchange_energies(new_energy_matrix, cell_energy, distributed_energies[1], idx, idy, idx+1, idy-1)
                    self.exchange_energies(new_energy_matrix, cell_energy, distributed_energies[2], idx, idy, idx-1, idy+1)
                    self.exchange_energies(new_energy_matrix, cell_energy, distributed_energies[3], idx, idy, idx-1, idy-1)

    def orient_data(self, input, x, y, dir):
        if (self.positions[x][y].parent[0] > x):
            if self.positions[x][y].parent[1] > y:
                input = np.roll(input, (dir*2))
            if self.positions[x][y].parent[1] < y:
                input = np.roll(input, (dir*1))
        if (self.positions[x][y].parent[0] < x):
            if self.positions[x][y].parent[1] > y:
                input = np.roll(input, (dir*3))
        return input

    def calculate_peer_energies(self, x, y):
        peer_energies = [0,0,0,0]
        if x+1 in self.positions:
            if y+1 in self.positions[x+1]:
                peer_energies[0] = self.positions[x+1][y+1].energy
            if y-1 in self.positions[x+1]:
                peer_energies[1] = self.positions[x+1][y-1].energy

        if x-1 in self.positions:
            if y+1 in self.positions[x-1]:
                peer_energies[2] = self.positions[x-1][y+1].energy
            if y-1 in self.positions[x-1]:
                peer_energies[3] = self.positions[x-1][y-1].energy

        peer_energies = self.orient_data(peer_energies, x, y, -1)

        return peer_energies

    def compute(self, input, x, y):
        out = self.dna.compute(input)
        #out = self.orient_data(out, x, y, -1)
        return out

    def exchange_energies(self, em, ce, de, x, y, x1, y1):
        if x1 in em:
            if y1 in em[x1]:
                em[x1][y1] += ce * de
                em[x][y] -= ce * de

                if em[x1][y1] > 1:
                    em[x1][y1] = 1
                if em[x][y] < 0:
                    em[x][y] = 0
        

    def calculate_replication_possibilities(self, x, y, energies):
        self.positions[x][y].ableToReplicate = [0]*4
        if not y+1 in energies[x+1]:
            self.positions[x][y].ableToReplicate[0] = 1
        if not y-1 in energies[x+1]:
            self.positions[x][y].ableToReplicate[1] = 1
        if not y+1 in energies[x-1]:
            self.positions[x][y].ableToReplicate[2] = 1
        if not y-1 in energies[x-1]:
            self.positions[x][y].ableToReplicate[3] = 1

    def sum_energies(self):
        sum = 0
        for idx in list(self.previous_energy_matrix):
            for idy in list(self.previous_energy_matrix[idx]):
                sum += self.previous_energy_matrix[idx][idy]
        return sum
    
    def kill(self):
        for idx in list(self.positions):
            for idy in list(self.positions[idx]):
                self.positions[idx][idy].die()
class Cell:
    def __init__(self, x, y, id, parent) -> None:
        self.x = x
        self.y = y
        self.id = str(id)
        self.energy = 0
        self.parent = parent
        self.is_new_cell = 0
        self.is_source = (parent == (400, 400))
        self.dead = False
        self.ableToReplicate = [0,0,0,0]
        self.replication_energy = [0,0,0,0]

    def tick(self, food_energy, energy_decay, replication_cost):
        self.energy += food_energy
        self.energy -= energy_decay
        newCell = None
        
        if self.energy >= 1:
            self.energy = 1

        if self.energy > 0.01:
            rep_e = np.multiply(self.replication_energy, self.ableToReplicate)
            transferred_energy = self.energy * np.sum(rep_e)
            newCell = self.replicate(rep_e, self.energy)
        if self.energy < 0.001:
            self.die()
            self.energy = 0
        return newCell

    def die(self):
        self.dead = True
    
    def replicate(self, rep_e, host_energy):
        xdir = 1
        ydir = 1
        cells = []
        if self.ableToReplicate.count(1) > 2 or self.is_new_cell == 1 or self.is_source:
            for _ in range(2):
                i = np.argmax(rep_e)
                if self.ableToReplicate[i] == 1 and (rep_e[i] > 0.25 or rep_e[i] == 0):
                    if i == 0:
                        xdir = 1
                        ydir = 1
                    elif i == 1:
                        xdir = 1
                        ydir = -1
                    elif i == 2:
                        xdir = -1
                        ydir = 1
                    elif i == 3:
                        xdir = -1
                        ydir = -1
                    cells.append(Cell(self.x+xdir, self.y+ydir, uuid.uuid1(), (self.x, self.y)))
                    cells[len(cells)-1].energy = rep_e[i]*host_energy
                    self.energy -= rep_e[i]*host_energy
                    rep_e[i] = 0
            self.is_new_cell += 1
        return cells


