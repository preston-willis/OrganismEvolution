# Quantum Coevolution

Minimal **closed**, unitary model: two coupled modes (truncated harmonic oscillators) with coevolved internal Hamiltonians \(H_A, H_B\), fixed beam-splitter coupling \(g\), and neuroevolution selecting for **regulated bipartite entanglement** \(f(S) = S(\ln n - S)\) (peak at \(S^* = \tfrac{1}{2}\ln n\)) plus mean score over each rollout from vacuum \|00⟩.

**What this is:** bipartite exchange layer + explicit selection — correlation symbiosis, not dissipative heat/waste or multi-layer consolidation.

**What this is not (yet):** open baths, per-mode entropy minimization, or extra “consolidator” fields. See **Thesis** and **Future extensions** in [quantum coevolution spec.md](quantum%20coevolution%20spec.md).

## Requirements

- Python 3.10+
- PyTorch 2.0+ (MPS on Apple Silicon, or CUDA/CPU)
- scipy
- matplotlib (only if using `--graph` or `--load`)

Run commands from the **repository root** (`OrganismEvolution/`).

## Quick start

```bash
# Neuroevolution (default: 10,000 generations)
python3 -m quantum --train

# Short smoke run
python3 -m quantum --train --generations 10

# Live matplotlib window (|ψ(x,y)|² + A/B drive heatmaps)
python3 -m quantum --train --graph

# Resume training from latest checkpoint in quantum/data/
python3 -m quantum --train --load

# Animate loaded best organism
python3 -m quantum --load

# Evolvability scaling experiment (random H pairs + short coevolution)
python3 -m quantum --basin
```

Training prints a line every generation (`LOG_INTERVAL` in `quantum/config.py`). Each generation starts from the configured initial field (default **vacuum** |00⟩). Checkpoints are saved every generation as `quantum/data/quantum_gen{N}_{fitness}.pt`.

**Reference at n=16:** \(S^* \approx 1.386\) nats, max parabolic fitness \(\approx 1.922\). Logged `entanglement` is raw \(S\); fitness uses \(f(S)\) plus rollout mean.

## Configuration

Edit `quantum/config.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `N` | 16 | Fock-space truncation per organism |
| `POPULATION_SIZE` | 50 | Evolution population |
| `STEPS_PER_EVAL` | 100 | Rollout steps per fitness evaluation |
| `POSITION_X_MAX` | 4.0 | Position grid extent for heatmaps (oscillator units) |
| `INITIAL_STATE` | vacuum | `vacuum` or `random` product state each generation |

## Tests

```bash
python3 -m pytest quantum/ -q
```

Checkpoints are not written during pytest runs.

## Layout

```
quantum/
  physics.py      # Hamiltonians, evolution, fitness
  evolution.py    # Population training loop
  checkpoint.py   # quantum/data/quantum_gen*.pt save/load
  viz.py          # Position-space heatmaps (Fock → x,y)
  grapher.py      # Live training UI
  sim.py          # --load animation
  basin.py        # Scaling experiment
  config.py
  data/           # Saved checkpoints (not used in tests)
```
