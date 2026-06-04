"""CLI and OpenGL wiring for the organism simulation."""

import argparse
import gc
import multiprocessing
import time

import OpenGL.GL as gl
import OpenGL.GLUT as glut
from OpenGL.GL import *
from OpenGL.GLUT import *

from config import (
    CNN_POPULATION_SIZE,
    CNN_TRAINING_EPOCHS,
    CNN_TRAINING_MAX_TIME,
    OPENGL_CLEAR_COLOR_A,
    OPENGL_CLEAR_COLOR_B,
    OPENGL_CLEAR_COLOR_G,
    OPENGL_CLEAR_COLOR_R,
    RENDERING_BASE_FPS,
    RENDERING_FPS,
    THERMO_CONDUCTANCE,
    TRAIN_HEADLESS,
    WORLD_SIZE,
)
from evolution import (
    CNN_FITNESS_MODES,
    CNNEvolutionDriver,
    cnn_fitness_mode_label,
    set_cnn_fitness_mode,
)
from Grapher import Grapher
from gpu_handler import GPUHandler
from input_handler import InputHandler
from model_io import clear_saved_networks, configure_organism_manager_from_args
from renderer import Renderer
from runtime import get_device
from simulation import Simulation

gpu_handler = GPUHandler()

current_simulation = None
current_renderer = None
current_input_handler = None
current_best_cnn = None
replay_mode = False
main_args = None


def start_cnn_evolution(grapher=None, load_latest=False, fitness_mode=None):
    set_cnn_fitness_mode(fitness_mode)
    import config

    mode = config.CNN_FITNESS_MODE
    print("Starting CNN Evolution Training...")
    print(f"Fitness mode: {cnn_fitness_mode_label(mode)} ({mode})")
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn", force=True)
    evolution_driver = CNNEvolutionDriver(
        WORLD_SIZE,
        epochs=CNN_TRAINING_EPOCHS,
        max_time=CNN_TRAINING_MAX_TIME,
        fitness_mode=mode,
    )
    if load_latest:
        if evolution_driver.ga.load_latest_model():
            parent = evolution_driver.ga.subjects[0]
            for i in range(1, CNN_POPULATION_SIZE):
                evolution_driver.ga.subjects[i].cppn.fc1.weight.data = parent.cppn.fc1.weight.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc1.bias.data = parent.cppn.fc1.bias.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc2.weight.data = parent.cppn.fc2.weight.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc2.bias.data = parent.cppn.fc2.bias.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc3.weight.data = parent.cppn.fc3.weight.data.clone()
                evolution_driver.ga.subjects[i].cppn.fc3.bias.data = parent.cppn.fc3.bias.data.clone()
                n_in = 1 + 1 + 3
                evolution_driver.ga.subjects[i].conv1.weight.data = (
                    evolution_driver.ga.subjects[i].cppn.generate_conv_weights(n_in, 32, 3)
                )
                evolution_driver.ga.subjects[i].conv1.bias.data = evolution_driver.ga.subjects[i].cppn.generate_bias(32)
                evolution_driver.ga.subjects[i]._regenerate_conv2_from_cppn()
            print("Loaded latest model and initialized population from it")
    evolution_driver.grapher = grapher
    return evolution_driver.run_evolution()
    

def main():
    global current_simulation, current_renderer, current_input_handler, current_best_cnn, replay_mode, main_args

    parser = argparse.ArgumentParser(description="Organism evolution simulation")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--fitness-mode", choices=CNN_FITNESS_MODES, default=None)
    args = parser.parse_args()

    if args.train:
        if not args.load:
            clear_saved_networks()
        grapher = Grapher() if ((not TRAIN_HEADLESS) or args.graph) else None
        start_cnn_evolution(grapher, load_latest=args.load, fitness_mode=args.fitness_mode)
        return

    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
    left_margin = 300
    render_size = int(WORLD_SIZE * (32 / WORLD_SIZE * 20))
    window_width = render_size + left_margin + 60
    window_height = render_size + 60
    glut.glutInitWindowSize(window_width, window_height)
    glut.glutCreateWindow(b"Organism Simulation")
    
    glEnable(GL_TEXTURE_2D)
    glClearColor(OPENGL_CLEAR_COLOR_R, OPENGL_CLEAR_COLOR_G, OPENGL_CLEAR_COLOR_B, OPENGL_CLEAR_COLOR_A)
    glViewport(0, 0, window_width, window_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    current_simulation = Simulation()
    configure_organism_manager_from_args(current_simulation.organism_manager, args, current_simulation)
    current_renderer = Renderer(WORLD_SIZE)
    current_renderer.top_margin = 10
    current_renderer.bottom_margin = 10
    current_renderer.right_margin = 50
    current_input_handler = InputHandler(current_renderer)
    current_renderer._setup_opengl()
    current_renderer.opengl_initialized = True
    current_input_handler.print_controls()

    rendering_frequency = max(1, RENDERING_BASE_FPS // RENDERING_FPS)
    last_frame_time = time.time()
    last_sim_data = None
    rendering_frame_counter = 0
    main_args = args
    
    def display():
        if last_sim_data is not None:
            current_renderer.update_texture(current_renderer.last_image)
            logger = current_simulation.logger if hasattr(current_simulation, "logger") else None
            overlay_data = dict(last_sim_data)
            overlay_data["debug_panel_mode"] = current_renderer.debug_panel_mode
            scale = current_simulation.conductance_scale
            current_renderer.render_opengl(
                overlay_data, scale * THERMO_CONDUCTANCE, replay_mode, current_best_cnn, logger
            )
    
    def keyboard(key, x, y):
        global current_simulation, current_harvest_scale
        if key == 27:
            glut.glutLeaveMainLoop()
        elif key == b"q":
            current_renderer.toggle_hidden_channel_0_view()
        elif key == b"m":
            current_renderer.toggle_render_mode()
        elif key == b"n":
            current_renderer.toggle_filters()
        elif key == b"b":
            current_renderer.toggle_org_energy_view()
        elif key == b"v":
            current_renderer.toggle_hidden_channels_view()
        elif key == b"\t":
            current_renderer.toggle_coupling_view()
        elif key == b"h":
            if current_simulation.conductance_scale == 0.0:
                current_simulation.conductance_scale = 1.0
                print(f"Conductance enabled: {THERMO_CONDUCTANCE}")
            else:
                current_simulation.conductance_scale = 0.0
                print("Conductance disabled")
        elif key == b"r":
            env_type = current_simulation.environment.environment_type
            current_simulation = Simulation()
            current_simulation.environment.environment_type = env_type
            if env_type == 2:
                current_simulation.environment.reset_perlin()
            else:
                current_simulation.environment.terrain = current_simulation.environment.generate_terrain()
            configure_organism_manager_from_args(current_simulation.organism_manager, main_args)
            print(f"Simulation reloaded (environment type: {env_type})")
        elif key == b"1":
            import config
            config.ENVIRONMENT_TYPE = 1
            current_simulation.environment.environment_type = 1
            current_simulation.environment.terrain = current_simulation.environment._generate_energy_mask_terrain()
        elif key == b"2":
            import config
            config.ENVIRONMENT_TYPE = 2
            current_simulation.environment.environment_type = 2
            current_simulation.environment.reset_perlin()
        elif key == b"3":
            import config
            config.ENVIRONMENT_TYPE = 3
            current_simulation.environment.time = 0.0
            current_simulation.environment.environment_type = 3
            current_simulation.environment.terrain = current_simulation.environment._generate_perlin_terrain()
    
    def idle():
        nonlocal rendering_frame_counter, last_sim_data, last_frame_time
        if replay_mode and getattr(current_simulation, "replay_simulation", None) is not None:
            last_sim_data = current_simulation.replay_simulation.update_simulation()
        else:
            last_sim_data = current_simulation.update_simulation()
        rendering_frame_counter += 1
        if rendering_frame_counter >= rendering_frequency and last_sim_data is not None:
            if current_renderer.render_mode == "org_top":
                mask = torch.zeros((WORLD_SIZE, WORLD_SIZE), device=get_device())
            else:
                mask = torch.clamp(last_sim_data["energy"], 0, 1)
            current_renderer.last_image = current_renderer.render(
                last_sim_data["terrain"],
                last_sim_data["topology"],
                mask,
                hidden_channels=last_sim_data.get("hidden_channels"),
                coupling_view=current_renderer.coupling_view_enabled,
            )
            glut.glutPostRedisplay()
            rendering_frame_counter = 0
        last_frame_time = time.time()
    
    import torch

    glut.glutDisplayFunc(display)
    glut.glutKeyboardFunc(keyboard)
    glut.glutIdleFunc(idle)
    try:
        glut.glutMainLoop()
    except KeyboardInterrupt:
        print("\nSimulation interrupted")
    finally:
        print("Simulation ended")


if __name__ == "__main__":
    main()
