import torch
from torchviz import make_dot

from config import CNN_HIDDEN_CHANNELS, WORLD_SIZE
from cnn import EnergyDistributionCNN
from gpu_handler import GPUHandler
from runtime import set_device

gpu_handler = GPUHandler()
set_device(gpu_handler.get_device())
device = gpu_handler.get_device()

model = EnergyDistributionCNN(device)
model.eval()

shareable_energy = torch.randn(WORLD_SIZE, WORLD_SIZE, device=device)
terrain = torch.randn(WORLD_SIZE, WORLD_SIZE, device=device)
spectrum = torch.randn(CNN_HIDDEN_CHANNELS, WORLD_SIZE, WORLD_SIZE, device=device)
rotation_matrix = torch.randn(WORLD_SIZE, WORLD_SIZE, device=device)

proportions, spectrum_out = model(shareable_energy, terrain, spectrum, rotation_matrix)
dot = make_dot((proportions.sum(), spectrum_out.sum()), params=dict(model.named_parameters()))
dot.render("cnn_architecture", format="png")
