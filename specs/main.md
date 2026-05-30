# main.py Behavior Specifications

This document specifies the intended behavior of every class and function in `main.py`.
Tests live in `test_main.py` and map to sections below.

---

## Module-level globals

| Name | Spec |
|------|------|
| `device` | PyTorch device from `GPUHandler.get_device()`. Used by all GPU tensors. |
| `current_harvest_rate` | Runtime harvest rate; toggled with `h` key in interactive mode. |
| `current_best_cnn` | Best CNN from training, used in replay mode. |
| `replay_mode` | Whether interactive sim shows replay organism. |

---

## `BasicCPPN`

CPPN that generates CNN kernel weights from 3D coordinates.

### `__init__(device)`
- Creates `fc1(3→16)`, `fc2(16→16)`, `fc3(16→1)` on `device`.
- All layers use tanh in forward except final linear.

### `forward(coords)` → `(N, 1)`
- **Input:** `(N, 3)` tensor `[radial_distance, in_ch_norm, out_ch_norm]`.
- **Output:** `(N, 1)` weight values.
- Applies tanh → tanh → linear.

### `generate_conv_weights(in_channels, out_channels, kernel_size)` → `(O, I, K, K)`
- Builds coordinate grid over kernel positions and channel indices (normalized to `[-1, 1]`).
- Radial distance `r = sqrt(x_norm² + y_norm²)` is the first CPPN input.
- Reshapes flat CPPN output to conv weight shape.

### `generate_bias(out_channels)` → `(O,)`
- One CPPN sample per output channel with coords `[0, 0, out_ch_norm]`.

---

## `EnergyDistributionCNN`

Oriented CNN: local-frame conv1, world-frame proportion rotation, binary sharing/hidden logits.

### `_RING_CIJ`
- 8 compass ring indices in 3×3 grid, clockwise from East.

### `_make_bucket_offsets(device)` → `(8, 8, 2)` int64
- For orientation bucket `k` and local ring index `l`, stores `(dy, dx)` world offset
  to gather neighbor into local slot `(l+k)%8`.

### `__init__(device)`
- Registers `_bucket_offsets` buffer.
- `conv1`: 4→32, kernel 3; `conv2`: 32→11, kernel 1.
- Weights initialized from CPPN; conv2 hidden bias (index 10) zeroed.

### `_regenerate_conv2_from_cppn()`
- Regenerates conv2 weights/bias from CPPN; re-applies hidden bias zero.

### `_zero_hidden_channel_bias()`
- Sets `conv2.bias[10] = 0`.

### `_rotate_proportions_8way(proportions, rotation_matrix)` → `(3, 3, H, W)`
- **Input:** cell-local proportions `(3, 3, H, W)`; per-cell angle snapped to 8 buckets.
- Rotates ring positions from local frame to world frame via `gather` on ring indices.
- Center cell `(1,1)` unchanged in meaning; ring cells permuted by bucket.

### `forward(shareable_energy, terrain, sharing_rate, hidden_channels, rotation_matrix)`
Returns `(proportions, sharing_rate_output, hidden_channels_output)`.

1. Stack 4 input channels: shareable, terrain, sharing_rate, hidden `(1,H,W)`.
2. `oriented_conv.conv1_forward` with rotation_matrix and bucket offsets.
3. conv2 → 11 channels.
4. Channels 0–8: softmax → `(3,3,H,W)` proportions, then `_rotate_proportions_8way`.
5. Channel 9: binary sharing logit (`> 0` → 1.0) — **not written back to sim state here**.
6. Channel 10: binary hidden logit (`> 0` → 1.0) `(1,H,W)`.

**Invariants:**
- Proportions sum to 1.0 per spatial cell (softmax).
- Proportions ≥ 0.

---

## `CNNGeneticAlgorithm`

### `__init__(pop_size, mut_rate, mut_mag, device)`
- Creates `pop_size` `EnergyDistributionCNN` instances.
- `fitness_scores` initialized to 0; `fittest_index` = 0; random 4-char `run_id`.

### `reset_fitness()`
- Sets all fitness scores to 0.0.

### `calc_fittest()`
- Sets `fittest_index` to index with **strictly greatest** fitness (> 0 required to beat initial best_index 0).
- If all zero, index 0 wins.

### `compute_generation()`
- Calls `calc_fittest`, `crossover(parent=fittest)`, `mutate`.

### `crossover(parent)`
- Copies parent CPPN weights to all subjects **except** `fittest_index`.
- Regenerates conv1/conv2 from CPPN for those copies.

### `mutate()`
- For each subject except fittest: mutates CPPN fc1/fc2/fc3 weights and biases where
  `rand < mut_rate` by adding uniform `[-mut_mag, mut_mag]`.
- Regenerates conv kernels from mutated CPPN.

### `save_model(index, generation=None)`
- Saves `subjects[index].state_dict()` to `data/cnn_{run_id}_gen{N}_{fitness}.pt` or without gen suffix.
- Creates `data/` if missing.

### `load_model(filename)`
- Loads state dict into `subjects[0]`; calls `_zero_hidden_channel_bias`.
- Prints error on failure; does not raise.

### `load_latest_model()` → bool
- Finds newest `data/cnn_*_gen*_*.pt` by mtime; loads via `load_model`.
- Returns False if no files.

---

## `_init_worker(device_str)`
- Sets module-global `_worker_device` from string (`cuda*`, config device, or cpu fallback).
- Patches `main.device` in worker process.

## `_release_device_memory(torch_device)`
- Calls `gc.collect()` and device `empty_cache()` for mps/cuda.

## `_evaluate_cnn_worker(args)` → `(total_cell_count, tick_data)`
- **Args:** `(cpu_state_dict, world_size, max_time, bot_index, collect_tick_data)`.
- Builds CNN + `Simulation(enable_debug=False)`, runs up to `max_time` ticks.
- Fitness = sum of cell counts over ticks; early stop if cells = 0.
- If `collect_tick_data`, records every 10th tick metrics tuple.
- Deletes sim/cnn; calls `_release_device_memory`.

---

## `CNNEvaluator`

### `__init__(world_size, max_time, device)`
- Stores dimensions; `pool = None`; optional grapher hooks.

### `_ensure_pool()`
- Creates `multiprocessing.Pool` with `TRAIN_WORKER_COUNT`, `_init_worker`, optional `maxtasksperchild`.

### `close_pool()` / `__del__`
- Closes/terminates pool; sets `pool = None`.

### `evaluate_population(subjects, pop_size)` → `[fitness, ...]`
- CPU-clones state dicts for pickling; `pool.map(_evaluate_cnn_worker, ...)`.
- Optionally feeds grapher from tick data.
- Calls `_release_device_memory(self.device)`.

### `_evaluate_single_cnn(simulation, bot_index=None)` → float
- Runs `max_time` ticks on given simulation (single process).
- Fitness = cumulative cell count; optional grapher enqueue every 10 ticks.

---

## `CNNEvolutionDriver`

### `__init__(world_size, epochs, max_time)`
- Creates `CNNGeneticAlgorithm` and `CNNEvaluator`.

### `evaluate_cnn(cnn, simulation)` → float
- Fresh `Simulation(enable_debug=False)` with given CNN; cumulative cell-count fitness.

### `create_replay_simulation(best_cnn)`
- Sets global replay state; attaches CNN to new `Simulation()`.

### `run_evolution()` → best `EnergyDistributionCNN`
- For each generation: evaluate population, print scores, `compute_generation`, save best model.
- Optionally updates grapher and replay sim.
- Always closes evaluator pool in `finally`.

---

## `Environment`

### `__init__(world_size, noise_scale, quantization_step)`
- Sets `environment_type` from config; generates initial terrain.

### `generate_terrain()` → `(H, W)` float32
- Dispatches to mask / sine / perlin generators by `environment_type`; default sine.

### `_generate_energy_mask_terrain()`
- Zeros grid; sets 3×3 neighborhoods at `ORGANISM_POSITIONS` to `STARTING_POSITION_TERRAIN_BOOST` (clamped 0–1).

### `_generate_sine_terrain()`
- Product of cosines on grid; normalized to [0,1]; `NOISE_POWER` shaping.
- Radial decay: `exp(-dist / (max_dist * TERRAIN_RADIAL_DECAY_SCALE))`.
- Merges 4 cardinal offset peaks; masks by `ENV_NOISE_THRESHOLD`.

### `_generate_perlin_terrain()`
- Multi-octave 3D Perlin slices at `self.time`; normalized and power-shaped.

### `compute_environment(topology_matrix, harvested_energy)`
- Depletes terrain: `terrain -= harvested × topology` (clamped 0–1).
- Type 1: periodic boost at seed positions.
- Type 3: advances time and regenerates perlin terrain.

---

## `OrganismManager`

Grid state: topology, energy, sharing_rate, hidden `(1,H,W)`, rotation, parent_giver_dir, candidates, `destroyed_energy` (scalar inaccessible bucket for decay/death losses).

### `_make_contrib_accum_weight` / `_make_dest_eff_gather_weight`
- Fixed 3×3 conv weights for shifting 9 contribution channels to destinations / gathering dest efficiency.

### `_shift_sum_contributions(contributions)` → `(H, W)`
- Circular conv accumulates 9 directional contribution planes onto receiving cells.

### `_compute_parent_incoming(contributions)` → `(H, W)`
- For cells with `parent_giver_dir == g`, reads inbound energy from parent neighbor
  along channel `(g+4)%8`.

### `_initialize_topology()`
- Places seeds from `ORGANISM_POSITIONS`: topology=1, energy=1, sharing=`ENERGY_SHARING_RATE>0`, hidden=0, rotation=0.

### `compute_topology()`
- No-op if no `new_cell_candidates`.
- `energy_mask = candidates with energy >= reproduction_threshold`.
- From stored `new_cell_contributions`, finds dominant inbound giver direction per birth site.
- **Birth side effects (when total inbound weight > 0):**
  - `parent_giver_dir` = dominant giver direction
  - `rotation_matrix` = snapped angle so local N opposes giver
  - `sharing_rate` = parent `hidden_channels[0]` at birth (0 or 1)
- Sets topology=1; hidden=0 for new cells.
- Fallback sharing if no giver data: `ENERGY_SHARING_RATE > 0`.

### `_apply_harvest_and_decay(terrain)` → `harvested_energy`
- Cells with `energy < DEATH_THRESHOLD` removed (all state cleared, parent=-1).
- Harvest: `min(terrain, sharing_rate * ENERGY_HARVEST_RATE)`.
- Death: cells below `DEATH_THRESHOLD` removed; their energy added to `destroyed_energy` and cleared from `energy_matrix`.
- Decay: `(ENERGY_DENSITY_DECAY_MODIFIER × sharing_rate² × (1 - avg_neighbor_organism_energy) × (1 - local_terrain) + ENERGY_DECAY) × topology`; set `ENERGY_DENSITY_DECAY_MODIFIER = 0` to disable density/terrain-modulated decay; energy removed added to `destroyed_energy`.
- Energy clamped [0,1] on alive cells.

### `_compute_energy_contributions(shareable, proportions)` → `(3,3,H,W)`
- Elementwise `shareable * proportions`.

### `_accumulate_contributions(contributions, shareable)` → `(H,W)`
- `energy - shareable + shift_sum(contributions)`.

### `_compute_source_removed(contributions, dest_efficiency, receiving_mask)` → `(H,W)`
- Scales each source's outbound contributions by destination receive efficiency gathered per direction.

### `_apply_capacity_constraints(new_energy, receiving_mask)`
- Caps incoming energy by remaining capacity `(1 - energy)`.

### `compute_energy(terrain)` → `harvested_energy`
Tick order:
1. Harvest/decay/death.
2. CNN forward; update hidden on alive cells only.
3. Build contributions; compute `full_distributed`.
4. **Reception rules:**
   - Candidates / empty: full neighbor sum.
   - Seed (alive, no parent): only self shareable (no sharing income).
   - Child (has parent): `shareable + parent_incoming` only.
5. Mark candidates where distributed > reproduction threshold.
6. Store contributions for birth if any candidates.
7. Apply capacity + source removal; scale `source_removed` down when it exceeds `actual_received` (parent-only blocks do not drain to `destroyed_energy`); update `energy_matrix` using `(distributed_total - shareable) * dest_efficiency - source_removed`; clamp loss goes to `destroyed_energy`; store candidate receive in `pending_birth_energy`.

`compute_topology` commits `pending_birth_energy` → `energy_matrix` at birth.

**Invariants:**
- Sharing rate unchanged after birth except on death (cleared to 0).
- Energy in [0, 1] after update.

---

## Thermodynamic constraints

Intended bookkeeping (what tests assert):

| Constraint | Meaning |
|------------|---------|
| **Outflow = shareable** | Per cell, `sum(contributions) == shareable_energy` when proportions sum to 1. |
| **Source removal = received** | After scaling, `sum(source_removed) == sum(actual_received)`. |
| **Net sharing delta** | `Δsum(energy) + Δdestroyed == sum(received) - sum(removed)` for sharing-only step (clamp loss only). |
| **Capacity** | No cell exceeds 1.0; `actual_received <= 1 - energy`. |
| **Parent-only income** | Children use `shareable + parent_incoming`, not full neighbor sum. |
| **Seed income** | Seeds use `distributed_total = shareable` → zero sharing income. |
| **Decay / death sink** | Energy removed by decay or death is accumulated in `OrganismManager.destroyed_energy` (inaccessible bucket). |
| **Total energy** | `sum(energy_matrix) + sum(terrain) + sum(pending_birth_energy) + destroyed_energy` conserved each tick. |
| **Harvest** | `harvested = min(terrain, sharing_rate × ENERGY_HARVEST_RATE)`; terrain loses same `harvested` per cell. |

**Birth energy:** incoming share at candidate sites is stored in `pending_birth_energy` during `compute_energy` and committed to `energy_matrix` only when `compute_topology` sets `topology = 1`. Empty cells never hold organism energy before birth.

## `Renderer`

### State toggles
| Method | Effect |
|--------|--------|
| `toggle_render_mode` | Switches `org_top` ↔ `org_energy`. |
| `toggle_filters` | Organism overlay on/off. |
| `toggle_debug_text` | Debug overlay flag. |
| `toggle_org_energy_view` | Alpha energy visualization. |
| `toggle_sharing_rate_view` | Raw vs thresholded sharing display flag. |
| `toggle_hidden_channel_0_view` | Green hidden overlay on/off. |

### `render(environment, topology, mask, ...)` → `(4, H, W)` RGBA
- Background: R=0, G=B=`env_scaled` (cyan terrain).
- With filters + sharing_rate:
  - Red = `1 - sharing` on cells; green = hidden; blue = 0.
  - White where sharing on and hidden off.
- Org energy mode: alpha from energy mask on cells.

### `_setup_opengl()` / `update_texture()` / `render_opengl()` / `render_text()` / `render_debug_info()`
- OpenGL texture upload and overlay; require GL context.
- `update_texture` no-op if not initialized.

---

## `Simulation`

### `__init__(enable_debug=True)`
- Creates `Environment`, `OrganismManager`, `Logger`; tick=0.

### `update_simulation()` → dict
1. `compute_energy`
2. `compute_topology`
3. `compute_environment`
4. Optional debug log / GPU cache clear when `enable_debug`
5. Increments tick; returns terrain, topology, energy, candidates, sharing, hidden.

### `reset_for_replay()`
- New `OrganismManager` on existing terrain; tick=0.

---

## `clear_saved_networks()`
- Deletes all `data/cnn_*.pt` files.

## `load_latest_cnn()` → `EnergyDistributionCNN | None`
- Loads newest generation checkpoint into new CNN instance.

## `start_cnn_evolution(grapher=None, load_latest=False)` → best CNN
- Sets multiprocessing spawn; runs `CNNEvolutionDriver.run_evolution`.
- Optional population seed from latest checkpoint.

## `main()`
- Parses CLI; `--train` runs evolution; otherwise OpenGL interactive loop with keyboard controls.

---

## Test coverage notes

| Area | Unit tested | Integration / manual |
|------|-------------|----------------------|
| CPPN / CNN math | Yes | — |
| Organism energy / birth | Yes | — |
| Thermodynamics | Yes | See `TestThermodynamics` |
| Environment terrain | Yes | Perlin slow path sampled |
| Genetic algorithm | Yes | — |
| Renderer tensor colors | Yes | OpenGL skipped |
| Multiprocessing pool | Partial | `_evaluate_cnn_worker` direct call |
| `main()` / GLUT loop | Spec only | Manual / E2E |
