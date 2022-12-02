import pygame


class Renderer:
    def __init__(self) -> None:
        pygame.init()
        self.gameDisplay = pygame.display.set_mode((800,800))
        self.gameDisplay.fill((0,0,0))
        self.pixAr = pygame.PixelArray(self.gameDisplay)
        self.clock = pygame.time.Clock()

    def render_cells(self, org, energy_pos):
        for idx in list(org.positions):
            for idy in list(org.positions[idx]):
                cell = org.positions[idx][idy]
                if cell.x < len(self.pixAr) and cell.y < len(self.pixAr[0]) and cell.dead == False:
                    color = int(cell.energy*255)
                    if color > 255: color = 255
                    if color < 0: color = 0
                    self.pixAr[cell.x][cell.y] = (0, color, 0)
                elif cell.dead:
                    self.pixAr[cell.x][cell.y] = (0,0,0)
        self.pixAr[energy_pos[0][0]][energy_pos[0][1]] = (255,0,0)
        self.pixAr[energy_pos[1][0]][energy_pos[1][1]] = (255,0,0)

    def reset(self):
        self.gameDisplay.fill((0,0,0))

    def getFPS(self):
        self.clock.tick(60)
        return self.clock.get_fps()

    def handle_pygame(self):
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
        pygame.display.update()
        return (energy_mod, pos)
