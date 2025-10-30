class InputHandler:
    """Handles keyboard input and controls for OpenGL display"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        self.quit_requested = False
    
    def handle_input(self, key=None):
        """Handle keyboard input - returns False if quit requested"""
        return not self.quit_requested
    
    def print_controls(self):
        """Print control instructions"""
        print(" Organism Simulation - OpenGL GPU Accelerated Rendering")
        print("Controls:")
        print("  'q' - Quit")
        print("  'm' - Toggle render mode (red dots / green dots)")
        print("  'n' - Toggle filters (enabled: organisms, disabled: environment only)")
        print("  'h' - Toggle harvest rate (enabled: 0.2, disabled: 0)")
        # Training is now started via --train CLI flag
        print("  'r' - Toggle replay mode (show best organism from training)")
        print("  'd' - Toggle debug text overlay")
        print("  'v' - Toggle large dots (10x size)")
        print("  'p' - Toggle performance mode (disable expensive rendering)")
        print("  Ctrl+C - Force quit")
        print("  Click on the window to focus for keyboard input")
