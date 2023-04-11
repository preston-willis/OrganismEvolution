import threading

print_lock = threading.Lock()
class RenderThread(threading.Thread):
    def __init__(self, queue, args=(), kwargs=None):
        threading.Thread.__init__(self, kwargs=None)
        self.queue = queue
        self.daemon = True
        self.saved_state = ()

    def run(self):
        from Renderer import Renderer
        self.pg_client = Renderer()
        tick = 0
        while True:
            tick += 1

            if tick % 1 == 0:
                args = self.queue.get()
                self.update_display(args)
                epos = self.saved_state[1]
                self.pg_client.reset()
                self.pg_client.render_cells(self.saved_state[0], epos)
                self.pg_client.handle_pygame()

    def update_display(self, args):
        self.pg_client.reset()
        with print_lock:
            if args:
                self.saved_state = args


