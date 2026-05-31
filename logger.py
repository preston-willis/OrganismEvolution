import psutil
import time

class Logger:
    """Handles logging and performance monitoring"""

    def __init__(self, log_path="simulation.log", enable_log=False):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.tick_times = []
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.log_path = log_path
        self.birth_log_path = "simulation_births.log"
        self.log_file = None
        self.birth_log_file = None
        if enable_log:
            self._open_log_file()

    def _open_log_file(self):
        if self.log_file is not None:
            return
        self.log_file = open(self.log_path, "w")
        self.log_file.write(
            "tick\tcells\torg_energy\tpending\tdecay_energy\tterrain_sum\tterrain_center"
            "\tmin_cell_energy\tmax_cell_energy\tsystem_energy\tevent\n"
        )
        self.log_file.flush()
        self.birth_log_file = open(self.birth_log_path, "w")
        self.birth_log_file.write(
            "tick\ty\tx\tparent_dir\tparent_y\tparent_x\tparent_living"
            "\tparent_hidden\tparent_sharing\tchild_sharing\tbirth_energy\tgiver_weight_sum\n"
        )
        self.birth_log_file.flush()

    def _close_log_file(self):
        if self.birth_log_file is not None:
            self.birth_log_file.close()
            self.birth_log_file = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def set_file_logging(self, enabled):
        if enabled:
            self._open_log_file()
        else:
            self._close_log_file()

    def get_debug_info(self):
        """Get comprehensive debug information"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        gpu_memory = "N/A"
        try:
            import torch
            if torch.backends.mps.is_available():
                gpu_memory = f"MPS: {torch.mps.current_allocated_memory() / 1024 / 1024:.1f}MB"
            elif torch.cuda.is_available():
                gpu_memory = f"CUDA: {torch.cuda.memory_allocated() / 1024 / 1024:.1f}MB"
        except:
            gpu_memory = "Unknown"
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

    def log_sim_tick(
        self,
        tick,
        cells,
        org_energy,
        pending,
        decay_energy,
        terrain_sum,
        terrain_center,
        min_cell_energy,
        max_cell_energy,
        system_energy,
        event="",
    ):
        """Append one simulation tick as a tab-separated row."""
        if self.log_file is None:
            return
        line = (
            f"{tick}\t{cells:.0f}\t{org_energy:.6f}\t{pending:.6f}\t{decay_energy:.6f}"
            f"\t{terrain_sum:.6f}\t{terrain_center:.6f}\t{min_cell_energy:.6f}"
            f"\t{max_cell_energy:.6f}\t{system_energy:.6f}\t{event}\n"
        )
        self.log_file.write(line)
        self.log_file.flush()

    def log_births(self, tick, birth_events):
        if self.birth_log_file is None:
            return
        for event in birth_events:
            line = (
                f"{tick}\t{event['y']}\t{event['x']}\t{event['parent_dir']}"
                f"\t{event['parent_y']}\t{event['parent_x']}\t{event['parent_living']}"
                f"\t{event['parent_hidden']:.6f}\t{event['parent_sharing']:.6f}"
                f"\t{event['child_sharing']:.6f}\t{event['birth_energy']:.6f}"
                f"\t{event['giver_weight_sum']:.6f}\n"
            )
            self.birth_log_file.write(line)
        if birth_events:
            self.birth_log_file.flush()

    def log_tick(self, tick, tick_time, debug_info, topology_info=None, energy_levels=None, total_energy=None, total_terrain=None, system_energy=None):
        """Log tick information with FPS to stdout."""
        fps = self.get_fps()
        print(f"Tick {tick}:")
        print(f"  FPS: {fps:.1f}")
        print(f"  Memory: {debug_info['memory_mb']:.1f}MB")
        print(f"  CPU: {debug_info['cpu_percent']:.1f}%")
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
        print(f"  Log file: {self.log_path}")
        print()
