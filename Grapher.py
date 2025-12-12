import matplotlib.pyplot as plt
from queue import Queue


class Grapher:
    def __init__(self):
        plt.ion()
        # Single window layout using GridSpec: bottom row is ONLY total energy
        self.fig = plt.figure(num='Training Metrics', figsize=(12, 10))
        gs = self.fig.add_gridspec(3, 2)
        self.ax_gen = self.fig.add_subplot(gs[0, 0])
        self.ax_tick_perbot_fit = self.fig.add_subplot(gs[0, 1])
        self.ax_org_energy = self.fig.add_subplot(gs[1, 0])
        self.ax_env_energy = self.fig.add_subplot(gs[1, 1])
        self.ax_total_energy = self.fig.add_subplot(gs[2, :])

        self.gen_history = []
        self.tick_x = []
        self.tick_max_fitness = []
        self.tick_bot_fitnesses = []
        self.tick_org_energy = []
        self.tick_env_energy = []
        self.tick_total_energy = []

        # Async queues to avoid GUI calls from worker threads
        self._tick_q: Queue = Queue()
        self._gen_q: Queue = Queue()
        # Per-bot tick series
        self.bot_tick_fitness = {}     # bot_index -> ([ticks], [fitness])
        self.bot_tick_org_energy = {}  # bot_index -> ([ticks], [org_energy])
        self.bot_tick_env_energy = {}  # bot_index -> ([ticks], [env_energy])
        self.bot_tick_total_energy = {}# bot_index -> ([ticks], [total_energy])

    def reset_tick_metrics(self):
        # Clear tick-series between generations
        self.tick_x.clear()
        self.tick_max_fitness.clear()
        self.tick_bot_fitnesses.clear()
        self.tick_org_energy.clear()
        self.tick_env_energy.clear()
        self.tick_total_energy.clear()
        # Clear per-bot tick series as well
        self.bot_tick_fitness.clear()
        self.bot_tick_org_energy.clear()
        self.bot_tick_env_energy.clear()
        self.bot_tick_total_energy.clear()
        # Clear tick axes immediately for visual reset
        self.ax_tick_perbot_fit.clear()
        self.ax_tick_perbot_fit.set_title('Per-bot Fitness (Cell Count, every 10 ticks)')
        self.ax_tick_perbot_fit.set_xlabel('Tick')
        self.ax_tick_perbot_fit.set_ylabel('Cumulative Cell Count')
        
        self.ax_org_energy.clear()
        self.ax_org_energy.set_title('Cell Count (every 10 ticks)')
        self.ax_org_energy.set_xlabel('Tick')
        self.ax_org_energy.set_ylabel('Cell Count')
        
        self.ax_env_energy.clear()
        self.ax_env_energy.set_title('Environment Energy (every 10 ticks)')
        self.ax_env_energy.set_xlabel('Tick')
        self.ax_env_energy.set_ylabel('Energy')
        
        self.ax_total_energy.clear()
        self.ax_total_energy.set_title('Total Energy (every 10 ticks)')
        self.ax_total_energy.set_xlabel('Tick')
        self.ax_total_energy.set_ylabel('Energy')
        try:
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass

    # These enqueue methods are safe to call from any thread
    def enqueue_generation(self, gen_index: int, best_fitness_history, population_fitnesses=None):
        # Only track best_fitness_history for generation plot; ignore per-bot-by-generation
        self._gen_q.put((gen_index, list(best_fitness_history), None))

    def enqueue_tick(self, tick: int, max_bot_fitness: float, bot_fitnesses, org_energy: float, env_energy: float, total_energy: float):
        if tick % 10 != 0:
            return
        self._tick_q.put((tick, max_bot_fitness, list(bot_fitnesses), org_energy, env_energy, total_energy))

    def enqueue_bot_tick(self, bot_index: int, tick: int, fitness: float, org_energy: float, env_energy: float, total_energy: float):
        """Per-bot tick metrics (called from evaluation threads)."""
        if tick % 10 != 0:
            return
        # Use the same queue, mark bot_index with negative sentinel in max_fit slot not used
        self._tick_q.put((tick, None, (bot_index, fitness), org_energy, env_energy, total_energy))

    # This method must be called from the main thread periodically
    def process_queued(self):
        # Process generation updates
        gen_updated = False
        while not self._gen_q.empty():
            gen_index, best_hist, _ = self._gen_q.get()
            self.gen_history = best_hist
            self.ax_gen.clear()
            self.ax_gen.set_title(f'Generation {gen_index}')
            self.ax_gen.set_xlabel('Generation')
            self.ax_gen.set_ylabel('Best Fitness (Cell Count)')
            self.ax_gen.plot(range(len(self.gen_history)), self.gen_history, marker='o')
            gen_updated = True

        # Process tick updates
        updated_tick = False
        while not self._tick_q.empty():
            tick, max_fit, bots_or_tuple, org_e, env_e, total_e = self._tick_q.get()
            # Global-series updates only when max_fit provided
            if max_fit is not None:
                self.tick_x.append(tick)
                self.tick_max_fitness.append(max_fit)
                avg_bot = sum(bots_or_tuple)/len(bots_or_tuple) if bots_or_tuple else 0.0
                self.tick_bot_fitnesses.append(avg_bot)
                self.tick_org_energy.append(org_e)
                self.tick_env_energy.append(env_e)
                self.tick_total_energy.append(total_e)
                updated_tick = True
            else:
                # Per-bot tuple
                bot_index, fitness = bots_or_tuple
                # fitness series
                ticks, vals = self.bot_tick_fitness.get(bot_index, ([], []))
                ticks.append(tick); vals.append(fitness)
                self.bot_tick_fitness[bot_index] = (ticks, vals)
                # org energy series
                t2, v2 = self.bot_tick_org_energy.get(bot_index, ([], []))
                t2.append(tick); v2.append(org_e)
                self.bot_tick_org_energy[bot_index] = (t2, v2)
                # env energy series
                t3, v3 = self.bot_tick_env_energy.get(bot_index, ([], []))
                t3.append(tick); v3.append(env_e)
                self.bot_tick_env_energy[bot_index] = (t3, v3)
                # total energy series
                t4, v4 = self.bot_tick_total_energy.get(bot_index, ([], []))
                t4.append(tick); v4.append(total_e)
                self.bot_tick_total_energy[bot_index] = (t4, v4)
                updated_tick = True

        if updated_tick:
            # Each metric on its own axis in the same window
            # Per-bot fitness (tick-based) colored (top-right)
            self.ax_tick_perbot_fit.clear()
            self.ax_tick_perbot_fit.set_title('Per-bot Fitness (Cell Count, every 10 ticks)')
            self.ax_tick_perbot_fit.set_xlabel('Tick')
            self.ax_tick_perbot_fit.set_ylabel('Cumulative Cell Count')
            for idx, (ticks, vals) in self.bot_tick_fitness.items():
                self.ax_tick_perbot_fit.plot(ticks, vals, label=f'Bot {idx}')
            # if self.bot_tick_fitness:
            #     self.ax_tick_perbot_fit.legend(loc='best')

            # Per-bot organism energy colored
            self.ax_org_energy.clear()
            self.ax_org_energy.set_title('Per-bot Cell Count (every 10 ticks)')
            self.ax_org_energy.set_xlabel('Tick')
            self.ax_org_energy.set_ylabel('Cell Count')
            for idx, (ticks, vals) in self.bot_tick_org_energy.items():
                self.ax_org_energy.plot(ticks, vals, label=f'Bot {idx}')
            # if self.bot_tick_org_energy:
            #     self.ax_org_energy.legend(loc='best')

            # Per-bot environment energy colored
            self.ax_env_energy.clear()
            self.ax_env_energy.set_title('Per-bot Environment Energy (every 10 ticks)')
            self.ax_env_energy.set_xlabel('Tick')
            self.ax_env_energy.set_ylabel('Energy')
            for idx, (ticks, vals) in self.bot_tick_env_energy.items():
                self.ax_env_energy.plot(ticks, vals, label=f'Bot {idx}')
            # if self.bot_tick_env_energy:
            #     self.ax_env_energy.legend(loc='best')

            # Per-bot total energy colored
            self.ax_total_energy.clear()
            self.ax_total_energy.set_title('Per-bot Total Energy (every 10 ticks)')
            self.ax_total_energy.set_xlabel('Tick')
            self.ax_total_energy.set_ylabel('Energy')
            for idx, (ticks, vals) in self.bot_tick_total_energy.items():
                self.ax_total_energy.plot(ticks, vals, label=f'Bot {idx}')
            # if self.bot_tick_total_energy:
            #     self.ax_total_energy.legend(loc='best')

        # Final draw/refresh
        if updated_tick or gen_updated:
            try:
                self.fig.tight_layout()
                self.fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:
                pass



