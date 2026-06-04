"""Runtime device handle shared across modules."""

_device = None


def get_device():
    return _device


def set_device(torch_device):
    global _device
    _device = torch_device
