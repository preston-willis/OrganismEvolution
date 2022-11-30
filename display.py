import pygame
import numpy as np
import uuid
import collections
import time
from copy import deepcopy

pygame.init()

white = (255,255,255)
black = (0,0,0)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

gameDisplay = pygame.display.set_mode((800,800))
gameDisplay.fill(black)

ENERGY_LOSS = 0.4
ENERGY_DECAY = 0.1
ENERGY_CONSERVED = 0.5
CELL_LIMIT = 10000
ORIGIN = [400, 100]

pixAr = pygame.PixelArray(gameDisplay)

def render_cells(org):
    for idx in list(org.positions):
        for idy in list(org.positions[idx]):
            cell = org.positions[idx][idy]
            if cell.x < len(pixAr) and cell.y < len(pixAr[0]) and cell.dead == False:
                color = int(cell.energy*255)
                if color > 255: color = 255
                pixAr[cell.x][cell.y] = (0, color, 0)
            elif cell.dead:
                pixAr[cell.x][cell.y] = black

def handle_pygame():
    energy_mod = 0
    pos = None
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            if event.mod == pygame.KMOD_NONE:
                pass
            else:
                if event.mod & pygame.KMOD_LSHIFT:
                    energy_mod += 1
                if event.mod & pygame.KMOD_RSHIFT:
                    energy_mod -= 1
                if event.mod & pygame.KMOD_CTRL:
                    energy_mod -= 1
        if event.type == pygame.MOUSEBUTTONUP:
            pos = pygame.mouse.get_pos()
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    return (energy_mod, pos)

class Organism:
    def __init__(self, origin) -> None:
        self.origin = origin
        self.originCell = Cell(origin[0], origin[1], uuid.uuid1())
        self.originCell.ableToReplicate = [1,0,0,0]
        self.positions = collections.defaultdict(dict)
        self.members = 0
        self.positions[self.originCell.x][self.originCell.y] = self.originCell
        self.previous_energy_matrix = []

    def tick(self, ticks, food_energy):
        energies = self.compute_energies(deepcopy(food_energy), ticks)
        updated_positions = deepcopy(self.positions)
        self.compute_cells(updated_positions, energies)

    def compute_cells(self, cells, energies):
        for idx in list(self.positions):
            for idy in list(self.positions[idx]):
                cell = cells[idx][idy]
                newCells = cell.tick(energies[cell.x][cell.y])
                if not cell.dead:
                    if len(newCells) > 0:
                        if self.members < CELL_LIMIT:
                            for newCell in newCells:
                                self.members += 1
                                cells[newCell.x][newCell.y] = newCell
                if cell.dead: 
                    self.members -= 1
                    cells[cell.x].pop(cell.y)
        self.positions = deepcopy(cells)

    def compute_energies(self, energy_matrix, ticks):
        available_energy = 1 - ENERGY_LOSS

        ENERGY_X_Y = 0.24
        ENERGY_X_NEG_Y = 0.24
        ENERGY_NEG_X_Y = 0.24
        ENERGY_NEG_X_NEG_Y = 0.24

        if ticks == 0:
            self.previous_energy_matrix = deepcopy(energy_matrix)
            return self.previous_energy_matrix
        elif ticks > 0:
            new_energy_matrix = energy_matrix
            if self.members > 0:
                for idx in list(self.previous_energy_matrix):
                    for idy in list(self.previous_energy_matrix[idx]):
                        cell_energy = self.previous_energy_matrix[idx][idy]
                        if idx in self.positions:
                            if idy in self.positions[idx]:
                                self.calculate_replication_possibilities(idx, idy, new_energy_matrix)
                        
                                peer_energies = [0,0,0,0]
                                self_energy = self.positions[idx][idy].energy

                                if idx+1 in new_energy_matrix:
                                    if idy+1 in new_energy_matrix[idx+1]:
                                        peer_energies[0] = self.positions[idx+1][idy+1].energy
                                    if idy-1 in new_energy_matrix[idx+1]:
                                        peer_energies[1] = self.positions[idx+1][idy-1].energy

                                if idx-1 in new_energy_matrix:
                                    if idy+1 in new_energy_matrix[idx-1]:
                                        peer_energies[2] = self.positions[idx-1][idy+1].energy
                                    if idy-1 in new_energy_matrix[idx-1]:
                                        peer_energies[3] = self.positions[idx-1][idy-1].energy


                                if idx+1 in new_energy_matrix:
                                    if idy+1 in new_energy_matrix[idx+1]:
                                        new_energy_matrix[idx+1][idy+1] += cell_energy * ENERGY_X_Y
                                    if idy-1 in new_energy_matrix[idx+1]:
                                        new_energy_matrix[idx+1][idy-1] += cell_energy * ENERGY_X_NEG_Y

                                if idx-1 in new_energy_matrix:
                                    if idy+1 in new_energy_matrix[idx-1]:
                                        new_energy_matrix[idx-1][idy+1] += cell_energy * ENERGY_NEG_X_Y  
                                    if idy-1 in new_energy_matrix[idx-1]:
                                        new_energy_matrix[idx-1][idy-1] += cell_energy * ENERGY_NEG_X_NEG_Y
                self.previous_energy_matrix = deepcopy(new_energy_matrix)
            else: return energy_matrix
            return new_energy_matrix
    
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


class Cell:
    def __init__(self, x, y, id) -> None:
        self.x = x
        self.y = y
        self.id = str(id)
        self.energy = 0
        self.dead = False
        self.ableToReplicate = [0,0,0,0]

    def tick(self, food_energy):
        self.energy = food_energy
        self.energy -= ENERGY_DECAY
        newCell = None
        if self.energy < 0:
            self.die()
        else:
            newCell = self.replicate()
        return newCell

    def die(self):
        self.dead = True
    
    def replicate(self):
        xdir = 1
        ydir = 1
        cells = []
        if not self.ableToReplicate == [0,0,0,0]:
            for i in range(len(self.ableToReplicate)):
                if self.ableToReplicate[i] == 1:
                    if i == 0:
                        xdir = 1
                        ydir = 1
                        cells.append(Cell(self.x+xdir, self.y+ydir, uuid.uuid1()))
                    elif i == 1:
                        xdir = 1
                        ydir = -1
                        cells.append(Cell(self.x+xdir, self.y+ydir, uuid.uuid1()))
                    elif i == 2:
                        xdir = -1
                        ydir = 1
                        cells.append(Cell(self.x+xdir, self.y+ydir, uuid.uuid1()))
                    elif i == 3:
                        xdir = -1
                        ydir = -1
                        cells.append(Cell(self.x+xdir, self.y+ydir, uuid.uuid1()))
        return cells

def generate_energy(org, emod, pos):
    food_energy = collections.defaultdict(dict)
    for idx in list(org.positions):
            for idy in list(org.positions[idx]):
                cell = org.positions[idx][idy]
                if ticks < 70:
                    food_energy[cell.x][cell.y] = 1
                else:
                    food_energy[cell.x][cell.y] = 0
                    if cell.y == 100 and cell.x == 380 and pos == None:
                        food_energy[cell.x][cell.y] = 100*emod
                    elif pos != None:
                        food_energy[pos[0]][pos[1]] = 100*emod
                    if cell.y == 100 and cell.x == 420:
                        food_energy[cell.x][cell.y] = 100*emod
    return food_energy

ticks = 0
org = Organism(ORIGIN)
lastMousePos = [380, 100]
emod = 1
while True:
    energy = generate_energy(org, emod, lastMousePos)
    org.tick(ticks, energy)
    render_cells(org)
    (edelta, pos) = handle_pygame()
    if pos != None: lastMousePos = pos
    emod += edelta
    pygame.display.update()

    ticks += 1