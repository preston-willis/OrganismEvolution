import glob
import os
import uuid

import torch

import oriented_conv
from config import CNN_HIDDEN_CHANNELS

CNN_OUTPUT_DIM = 12


class BasicCPPN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fc1 = torch.nn.Linear(3, 16, device=device)
        self.fc2 = torch.nn.Linear(16, 16, device=device)
        self.fc3 = torch.nn.Linear(16, 1, device=device)
        self.to(device)

    def forward(self, coords):
        x = torch.tanh(self.fc1(coords))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def generate_conv_weights(self, in_channels, out_channels, kernel_size):
        coords_list = []
        for out_ch in range(out_channels):
            for in_ch in range(in_channels):
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        x_norm = (kx / max(kernel_size - 1, 1)) * 2 - 1 if kernel_size > 1 else 0
                        y_norm = (ky / max(kernel_size - 1, 1)) * 2 - 1 if kernel_size > 1 else 0
                        r = (x_norm**2 + y_norm**2) ** 0.5
                        in_ch_norm = (in_ch / max(in_channels - 1, 1)) * 2 - 1 if in_channels > 1 else 0
                        out_ch_norm = (out_ch / max(out_channels - 1, 1)) * 2 - 1 if out_channels > 1 else 0
                        coords_list.append([r, in_ch_norm, out_ch_norm])
        coords = torch.tensor(coords_list, dtype=torch.float32, device=self.device)
        weights_flat = self.forward(coords).squeeze(-1)
        return weights_flat.view(out_channels, in_channels, kernel_size, kernel_size)

    def generate_bias(self, out_channels):
        coords_list = []
        for out_ch in range(out_channels):
            out_ch_norm = (out_ch / max(out_channels - 1, 1)) * 2 - 1 if out_channels > 1 else 0
            coords_list.append([0.0, 0.0, out_ch_norm])
        coords = torch.tensor(coords_list, dtype=torch.float32, device=self.device)
        return self.forward(coords).squeeze(-1)


class EnergyDistributionCNN(torch.nn.Module):
    _RING_CIJ = [(1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2)]

    @staticmethod
    def _make_bucket_offsets(device):
        offsets = torch.zeros(8, 8, 2, dtype=torch.long, device=device)
        for k in range(8):
            for l in range(8):
                ci, cj = EnergyDistributionCNN._RING_CIJ[(l + k) % 8]
                offsets[k, l, 0] = ci - 1
                offsets[k, l, 1] = cj - 1
        return offsets

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.register_buffer("_bucket_offsets", self._make_bucket_offsets(device), persistent=False)
        ring_ci = []
        ring_cj = []
        for ci, cj in self._RING_CIJ:
            ring_ci.append(ci)
            ring_cj.append(cj)
        self.register_buffer("_ring_ci", torch.tensor(ring_ci, device=device, dtype=torch.long), persistent=False)
        self.register_buffer("_ring_cj", torch.tensor(ring_cj, device=device, dtype=torch.long), persistent=False)
        self.cppn = BasicCPPN(device)
        self.conv1 = torch.nn.Conv2d(
            1 + 1 + CNN_HIDDEN_CHANNELS, 32, kernel_size=3, stride=1, padding=0, device=device
        )
        self.conv2 = torch.nn.Conv2d(32, CNN_OUTPUT_DIM, kernel_size=1, device=device)
        self.conv1.weight.data = self.cppn.generate_conv_weights(1 + 1 + CNN_HIDDEN_CHANNELS, 32, 3)
        self.conv1.bias.data = self.cppn.generate_bias(32)
        self._regenerate_conv2_from_cppn()
        self.to(device)

    def _regenerate_conv2_from_cppn(self):
        self.conv2.weight.data = self.cppn.generate_conv_weights(32, CNN_OUTPUT_DIM, 1)
        self.conv2.bias.data = self.cppn.generate_bias(CNN_OUTPUT_DIM)
        self._zero_hidden_channel_bias()

    def _zero_hidden_channel_bias(self):
        for ch in range(9, CNN_OUTPUT_DIM):
            self.conv2.bias.data[ch] = 0

    def _rotate_proportions_8way(self, proportions, rotation_matrix):
        bucket = (torch.round(rotation_matrix / (torch.pi / 4)) % 8).long()
        ring = torch.stack([proportions[ci, cj] for ci, cj in self._RING_CIJ])
        d_indices = torch.arange(8, device=proportions.device).view(8, 1, 1)
        source_idx = (d_indices - bucket.unsqueeze(0)) % 8
        rotated_ring = torch.gather(ring, 0, source_idx)
        rotated = proportions.clone()
        rotated[self._ring_ci, self._ring_cj] = rotated_ring
        return rotated

    def forward(self, shareable_energy, terrain, spectrum, rotation_matrix):
        world_size = shareable_energy.shape[0]
        input_channels = torch.cat(
            [shareable_energy.unsqueeze(0), terrain.unsqueeze(0), spectrum], dim=0
        )
        x = oriented_conv.conv1_forward(
            input_channels,
            self.conv1.weight,
            self.conv1.bias,
            rotation_matrix,
            self._bucket_offsets,
        )
        x = self.conv2(x.unsqueeze(0)).squeeze(0)
        proportions_flat = torch.nn.functional.softmax(x[:9], dim=0)
        spectrum_out = torch.nn.functional.softmax(x[9 : 9 + CNN_HIDDEN_CHANNELS], dim=0)
        proportions = proportions_flat.view(3, 3, world_size, world_size)
        proportions = self._rotate_proportions_8way(proportions, rotation_matrix)
        return proportions, spectrum_out


def neighbor_direction_weights_from_proportions(proportions, topology):
    h, w = proportions.shape[2], proportions.shape[3]
    device = proportions.device
    empty = topology == 0
    y_grid = torch.arange(h, device=device).view(h, 1)
    x_grid = torch.arange(w, device=device).view(1, w)
    ring_ci = EnergyDistributionCNN._RING_CIJ
    weights = []
    for g in range(8):
        sci, scj = ring_ci[g]
        oci, ocj = ring_ci[(g + 4) % 8]
        sy = (y_grid + sci - 1) % h
        sx = (x_grid + scj - 1) % w
        weights.append(proportions[oci, ocj, sy, sx] * empty[sy, sx].float())
    stacked = torch.stack(weights, dim=0)
    total = stacked.sum(dim=0, keepdim=True)
    return torch.where(total > 0, stacked / total.clamp(min=1e-12), stacked)


class CNNGeneticAlgorithm:
    def __init__(self, pop_size, mut_rate, mut_mag, device):
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.mut_mag = mut_mag
        self.device = device
        self.fittest_index = 0
        self.run_id = str(uuid.uuid1())[:4]
        self.subjects = [EnergyDistributionCNN(device) for _ in range(pop_size)]
        self.fitness_scores = [0.0] * pop_size

    def reset_fitness(self):
        self.fitness_scores = [0.0] * self.pop_size

    def compute_generation(self):
        self.calc_fittest()
        self.crossover(self.subjects[self.fittest_index])
        self.mutate()

    def calc_fittest(self):
        best_fitness = 0
        best_index = 0
        for i, fitness in enumerate(self.fitness_scores):
            if fitness > best_fitness:
                best_fitness = fitness
                best_index = i
        self.fittest_index = best_index

    def crossover(self, parent):
        for i in range(self.pop_size):
            if i != self.fittest_index:
                self.subjects[i].cppn.fc1.weight.data = parent.cppn.fc1.weight.data.clone()
                self.subjects[i].cppn.fc1.bias.data = parent.cppn.fc1.bias.data.clone()
                self.subjects[i].cppn.fc2.weight.data = parent.cppn.fc2.weight.data.clone()
                self.subjects[i].cppn.fc2.bias.data = parent.cppn.fc2.bias.data.clone()
                self.subjects[i].cppn.fc3.weight.data = parent.cppn.fc3.weight.data.clone()
                self.subjects[i].cppn.fc3.bias.data = parent.cppn.fc3.bias.data.clone()
                n_in = 1 + 1 + CNN_HIDDEN_CHANNELS
                self.subjects[i].conv1.weight.data = self.subjects[i].cppn.generate_conv_weights(n_in, 32, 3)
                self.subjects[i].conv1.bias.data = self.subjects[i].cppn.generate_bias(32)
                self.subjects[i]._regenerate_conv2_from_cppn()

    def mutate(self):
        for i in range(self.pop_size):
            if i == self.fittest_index:
                continue
            for layer in [self.subjects[i].cppn.fc1, self.subjects[i].cppn.fc2, self.subjects[i].cppn.fc3]:
                mutation_mask = torch.rand_like(layer.weight) < self.mut_rate
                mutations = (torch.rand_like(layer.weight) - 0.5) * 2 * self.mut_mag
                layer.weight.data[mutation_mask] += mutations[mutation_mask]
                mutation_mask = torch.rand_like(layer.bias) < self.mut_rate
                mutations = (torch.rand_like(layer.bias) - 0.5) * 2 * self.mut_mag
                layer.bias.data[mutation_mask] += mutations[mutation_mask]
            n_in = 1 + 1 + CNN_HIDDEN_CHANNELS
            self.subjects[i].conv1.weight.data = self.subjects[i].cppn.generate_conv_weights(n_in, 32, 3)
            self.subjects[i].conv1.bias.data = self.subjects[i].cppn.generate_bias(32)
            self.subjects[i]._regenerate_conv2_from_cppn()

    def save_model(self, index, generation=None):
        if generation is not None:
            filename = f"data/cnn_{self.run_id}_gen{generation}_{self.fitness_scores[index]:.6f}.pt"
        else:
            filename = f"data/cnn_{self.run_id}_{self.fitness_scores[index]:.6f}.pt"
        os.makedirs("data", exist_ok=True)
        torch.save(self.subjects[index].state_dict(), filename)
        print(f"Saved model: {filename}")

    def load_model(self, filename):
        try:
            state_dict = torch.load(filename, map_location=self.device)
            self.subjects[0].load_state_dict(state_dict)
            self.subjects[0]._zero_hidden_channel_bias()
            print(f"Loaded model: {filename}")
        except Exception as e:
            print(f"Couldn't load {filename}: {e}")

    def load_latest_model(self):
        files = glob.glob("data/cnn_*_gen*_*.pt")
        if not files:
            print("No saved models found in data/ directory")
            return False
        files.sort(key=os.path.getmtime, reverse=True)
        self.load_model(files[0])
        return True
