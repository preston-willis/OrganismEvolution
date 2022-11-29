import pygame
import numpy as np
import uuid
pygame.init()

white = (255,255,255)
black = (0,0,0)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

gameDisplay = pygame.display.set_mode((800,800))
gameDisplay.fill(black)

ENERGY_DECAY = 0.00005
CELL_LIMIT = 500
ORIGIN = [400, 0]

ENERGY_CONSERVED = 0.5

pixAr = pygame.PixelArray(gameDisplay)

def render_cells(org):
    for cell in org.cells:
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
        self.cells = [Cell(origin[0], origin[1], uuid.uuid1())]
        self.occupied = [origin]
        self.energyConserved = ENERGY_CONSERVED

    def tick(self, food_energy):
        ind = len(self.cells)-1
        food_from_cell = food_energy
        newCells = []
        for cell in reversed(self.cells):
            if ind < len(self.cells)-1:
                food_from_cell = self.cells[ind + 1].energy*(1-self.energyConserved)
                self.cells[ind + 1].energy *= self.energyConserved

            newCell = cell.tick(food_from_cell)

            if not cell.dead:
                if newCell is not None:
                    if [newCell.x, newCell.y] not in self.occupied and len(self.cells) < CELL_LIMIT:
                        newCells.append(newCell)
                        self.occupied.append([newCell.x, newCell.y])
                        cell.ableToReplicate -= 1
                        cell.adjacents.append(newCell)
            if cell.dead: 
                del self.cells[ind]
                self.occupied.remove([cell.x, cell.y])
                self.cells[ind-1].ableToReplicate = 1
            ind -= 1
        [self.cells.append(i) for i in newCells]

class Cell:
    def __init__(self, x, y, id) -> None:
        self.x = x
        self.y = y
        self.id = str(id)
        self.energy = 0
        self.dead = False
        self.ableToReplicate = 1
        self.adjacents = []

    def tick(self, food_energy):
        self.energy += food_energy
        self.energy -= ENERGY_DECAY
        newCell = None
        if self.energy > 1:
            self.energy = 1
        if self.energy < 0:
            self.energy = 0
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

organism = Organism(ORIGIN)
ticks = 0
energy_by_tick = 1
while True:
    organism.tick(energy_by_tick)
    render_cells(organism)
    print(energy_by_tick)
    energy_by_tick += handle_pygame()
    pygame.display.update()

    ticks += 1
    energy_by_tick = 0.1