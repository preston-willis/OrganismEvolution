import torch

from quantum.config import DEVICE_TYPE


def get_device():
    if DEVICE_TYPE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if DEVICE_TYPE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype():
    return torch.complex64
