import glob
import os

import torch

from cnn import EnergyDistributionCNN
from runtime import get_device


def clear_saved_networks():
    files = glob.glob("data/cnn_*.pt")
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    if files:
        print(f"Cleared {len(files)} saved network files")


def load_latest_cnn():
    files = glob.glob("data/cnn_*_gen*_*.pt")
    if not files:
        print("No saved models found in data/ directory")
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    try:
        device = get_device()
        cnn = EnergyDistributionCNN(device)
        state_dict = torch.load(latest_file, map_location=device)
        cnn.load_state_dict(state_dict)
        cnn._zero_hidden_channel_bias()
        print(f"Loaded model: {latest_file}")
        return cnn
    except Exception as e:
        print(f"Couldn't load {latest_file}: {e}")
        return None


def apply_loaded_cnn(organism_manager, loaded_cnn):
    organism_manager.energy_distribution_cnn = loaded_cnn


def configure_organism_manager_from_args(organism_manager, args, simulation=None):
    if args.load:
        loaded_cnn = load_latest_cnn()
        if loaded_cnn is not None:
            apply_loaded_cnn(organism_manager, loaded_cnn)
            print("Using loaded model in simulation")
        else:
            print("Failed to load model, using default CNN")
