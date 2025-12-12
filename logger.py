import psutil
import time

class Logger:
    """Handles logging and performance monitoring"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.tick_times = []
        self.last_fps_time = time.time()
        self.fps_counter = 0
    
    def get_debug_info(self):
        """Get comprehensive debug information"""
        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        
        # GPU memory (if available)
        gpu_memory = "N/A"
        try:
            import torch
            if torch.backends.mps.is_available():
                gpu_memory = f"MPS: {torch.mps.current_allocated_memory() / 1024 / 1024:.1f}MB"
            elif torch.cuda.is_available():
                gpu_memory = f"CUDA: {torch.cuda.memory_allocated() / 1024 / 1024:.1f}MB"
        except:
            gpu_memory = "Unknown"
        
        # Uptime
        uptime = time.time() - self.start_time
        
        return {
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'gpu_memory': gpu_memory,
            'uptime': uptime
        }
    
    def update_fps(self):
        """Update FPS counter - call this every frame"""
        current_time = time.time()
        self.fps_counter += 1
        
        # Calculate FPS every second
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.last_fps_time = current_time
            self.fps_counter = 0
            self.current_fps = fps
            return fps
        return None
    
    def get_fps(self):
        """Get current FPS"""
        return getattr(self, 'current_fps', 0.0)
    
    def log_tick(self, tick, tick_time, debug_info, topology_info=None, energy_levels=None, total_energy=None, total_terrain=None, system_energy=None):
        """Log tick information with FPS"""
        # Get current FPS
        fps = self.get_fps()
        
        print(f"Tick {tick}:")
        print(f"  FPS: {fps:.1f}")
        print(f"  Memory: {debug_info['memory_mb']:.1f}MB")
        print(f"  CPU: {debug_info['cpu_percent']:.1f}%")
        print(f"  GPU: {debug_info['gpu_memory']}")
        if topology_info is not None:
            print(f"  {topology_info}")
        if energy_levels is not None:
            print(f"  Energy: {energy_levels}")
        if total_energy is not None:
            print(f"  Organism Energy: {total_energy:.6f}")
        if total_terrain is not None:
            print(f"  Terrain Energy: {total_terrain:.6f}")
        if system_energy is not None:
            print(f"  System Energy: {system_energy:.6f}")
        print(f"  Uptime: {debug_info['uptime']:.1f}s")
        print()
