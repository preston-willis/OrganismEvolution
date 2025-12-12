import torch
from config import DEVICE_TYPE

class GPUHandler:
    """Handles GPU device selection and management"""
    
    def __init__(self):
        self.device = self._setup_device()
        self._print_device_info()
    
    def _setup_device(self):
        """Setup and return the appropriate device"""
        
        if DEVICE_TYPE == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif DEVICE_TYPE == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("⚠️ Using CPU")
        
        return device
    
    def _print_device_info(self):
        """Print device information"""
        print(f"Using device: {self.device}")
        
        try:
            test_tensor = torch.tensor([1, 2, 3], device=self.device)
        except Exception as e:
            self.device = torch.device("cpu")
    
    def get_device(self):
        """Get the current device"""
        return self.device
    
    def clear_cache(self):
        """Clear GPU cache"""
        from config import DEVICE_TYPE
        if self.device.type == DEVICE_TYPE:
            if DEVICE_TYPE == 'mps':
                torch.mps.empty_cache()
            elif DEVICE_TYPE == 'cuda':
                torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
