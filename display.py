import pygame
import numpy as np
import uuid
import collections

pygame.init()

white = (255,255,255)
black = (0,0,0)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

gameDisplay = pygame.display.set_mode((800,800))
gameDisplay.fill(black)

ENERGY_DECAY = 0.1
CELL_LIMIT = 800
ORIGIN = [400, 0]

ENERGY_CONSERVED = 0.5

pixAr = pygame.PixelArray(gameDisplay)

def render_cells(org):
    for idx in list(org.positions):
        for idy in list(org.positions[idx]):
            cell = org.positions[idx][idy]
            if cell.x < len(pixAr) and cell.y < len(pixAr[0]) and cell.dead == False:
                pixAr[cell.x][cell.y] = (0, int(cell.energy*255), 0)
            elif cell.dead:
                pixAr[cell.x][cell.y] = black

def handle_pygame():
    energy_mod = 0
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
            if event.mod == pygame.KMOD_NONE:
                pass
            else:
                if event.mod & pygame.KMOD_LSHIFT:
                    energy_mod += 0.1
                if event.mod & pygame.KMOD_RSHIFT:
                    energy_mod -= 0.1
                if event.mod & pygame.KMOD_CTRL:
                    energy_mod -= 1
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    return energy_mod

class Organism:
    def __init__(self, origin) -> None:
        self.origin = origin
        self.originCell = Cell(origin[0], origin[1], uuid.uuid1())
        self.positions = collections.defaultdict(dict)
        self.members = 0
        self.positions[self.originCell.x][self.originCell.y] = self.originCell
        self.energyConserved = ENERGY_CONSERVED
        self.previous_energy_matrix = []

    def tick(self, ticks):
        if ticks < 800:
            food_energy = [1]*self.members
        else:
            food_energy = [0.04]*self.members
            for i in range(10):
                food_energy[(ticks+(100*i))%799] = 1

        energies = collections.defaultdict(dict)#self.compute_energies(food_energy, ticks)
        updated_positions = self.positions
        for idx in list(self.positions):
            for idy in list(self.positions[idx]):
                cell = self.positions[idx][idy]
                energies[cell.x][cell.y] = 0
                newCell = cell.tick(energies[cell.x][cell.y])
                if not cell.dead:
                    if newCell is not None:
                        if self.members < CELL_LIMIT:
                            self.members += 1
                            updated_positions[newCell.x][newCell.y] = newCell
                            cell.ableToReplicate -= 1
                if cell.dead: 
                    self.members -= 1
                    updated_positions[cell.x].pop(cell.y)
        self.positions = updated_positions

    def compute_energies(self, energy_matrix, ticks):
        ENERGY_UP = 0.45
        ENERGY_DOWN = 0.45
        if ticks == 0:
            self.previous_energy_matrix = [energy_matrix[0]]*3
            return self.previous_energy_matrix

        elif ticks > 0:
            new_energy_matrix = [0]*self.members
            if len(self.cells) > 3:
                for cell_index in range(self.members):
                    if cell_index < len(self.previous_energy_matrix)-1:
                        energy_down = self.previous_energy_matrix[cell_index+1]
                        energy_up = self.previous_energy_matrix[cell_index-1]
                    else:
                        energy_down = 0
                        energy_up = 0
                    if cell_index == 0:
                        new_energy_matrix[cell_index] = energy_down * ENERGY_DOWN
                    if cell_index == len(self.cells)-1:
                        new_energy_matrix[cell_index] = energy_up * ENERGY_UP
                    else:
                        new_energy_matrix[cell_index] = (energy_up * ENERGY_UP) + (energy_down * ENERGY_DOWN)
                total_energy = np.add(new_energy_matrix, energy_matrix)
                for e in range(len(total_energy)):
                    if total_energy[e] > 1: total_energy[e] = 1
                    if total_energy[e] < 0: total_energy[e] = 0

                self.previous_energy_matrix = total_energy
            else: return energy_matrix
            return self.previous_energy_matrix


class Cell:
    def __init__(self, x, y, id) -> None:
        self.x = x
        self.y = y
        self.id = str(id)
        self.energy = 0
        self.dead = False
        self.ableToReplicate = 1

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

        if np.random.uniform(-1, 1) < 0:
            xdir *= -1
        else:
            xdir *= 1
        # if np.random.uniform(-1, 1) < 0:
        #     ydir *= -1
        # else:
        #     ydir *= 1
            
        if self.ableToReplicate > 0:
            return Cell(self.x+xdir, self.y+ydir, uuid.uuid1())
        return None

organisms = []
for i in range(10):
    organisms.append(Organism([i*80,0]))
ticks = 0
energy_by_tick = 1
while True:
    for i in range(10):
        organisms[i].tick(ticks)
        render_cells(organisms[i])
    energy_by_tick += handle_pygame()
    pygame.display.update()

    ticks += 1
    energy_by_tick = 0.1