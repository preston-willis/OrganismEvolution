import torch
from torchviz import make_dot
from config import WORLD_SIZE
from gpu_handler import GPUHandler
from main import EnergyDistributionCNN

# Initialize device
gpu_handler = GPUHandler()
device = gpu_handler.get_device()

# Create model instance
model = EnergyDistributionCNN(device)
model.eval()

# Create dummy input tensors matching the forward signature
shareable_energy = torch.randn(WORLD_SIZE, WORLD_SIZE, device=device)
terrain = torch.randn(WORLD_SIZE, WORLD_SIZE, device=device)
sharing_rate = torch.randn(WORLD_SIZE, WORLD_SIZE, device=device)
hidden_channels = torch.randn(3, WORLD_SIZE, WORLD_SIZE, device=device)

# Forward pass to get output
output = model(shareable_energy, terrain, sharing_rate, hidden_channels)

# Create visualization
dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# Save to file
output_file = 'cnn_architecture'
dot.render(output_file, format='png', cleanup=True)

print(f"CNN diagram saved to {output_file}.png")
