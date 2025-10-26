import torch

class GPUHandler:
    """Handles GPU device selection and management"""
    
    def __init__(self):
        self.device = self._setup_device()
        self._print_device_info()
    
    def _setup_device(self):
        """Setup and return the appropriate device"""
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✅ Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("✅ Using CUDA GPU")
        else:
            device = torch.device("cpu")
            print("⚠️ Using CPU")
        
        return device
    
    def _print_device_info(self):
        """Print device information"""
        print(f"Using device: {self.device}")
        
        try:
            test_tensor = torch.tensor([1, 2, 3], device=self.device)
            print(f"Test tensor device: {test_tensor.device}")
            print("✅ PyTorch GPU backend is working!")
        except Exception as e:
            print(f"GPU setup error: {e}")
            print("Falling back to CPU")
            self.device = torch.device("cpu")
    
    def get_device(self):
        """Get the current device"""
        return self.device
    
    def clear_cache(self):
        """Clear GPU cache if using MPS"""
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
