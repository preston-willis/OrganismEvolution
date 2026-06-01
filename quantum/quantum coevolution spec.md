# Quantum Coevolution — Full Technical Specification

## Thesis

**Implemented claim (what this codebase tests):** In a **closed**, unitarily evolving bipartite system, two coupled quantum modes with **coevolved internal Hamiltonians** and a **fixed beam-splitter coupling** can be selected (via neuroevolution) to build **regulated entanglement** and **functional differentiation** between A and B when repeatedly rolled out from a product initial state (default vacuum \|00⟩).

**Metaphor (not simulated here):** A longer-term vision is hierarchical “export” of local disorder into partner modes and additional consolidating layers (life-like circulation of excitation and correlation). **This repository implements only the first layer:** A ↔ B exchange in a single joint pure state, with **no** dissipative bath, **no** extra consolidator fields, and **no** per-organism objective to minimize local entropy.

**Refined empirical hypothesis (basin experiment):** Under coupling + selection for regulated entanglement, the **fraction of random Hamiltonian pairs that evolve** to entangled, organized states may **increase with Fock truncation** `n` — testing **evolvability**, not pure Hamiltonian flow from random *states* alone.

**Out of scope for current claims:** Open-system thermodynamics (heat, waste, baths), literal “entropy minimization” fitness, origins-of-life necessity, or multi-layer consolidation.

-----

## System Overview

Two truncated quantum harmonic oscillators (modes A and B) share one **pure** joint state \|ψ_AB⟩. Evolution is **closed and unitary**. A static beam-splitter coupling `H_int` exchanges excitation between modes. Trainable pieces are **internal Hamiltonians** `H_A`, `H_B` (from matrices `M_A`, `M_B`). **Selection** is explicit: fitness rewards **intermediate subsystem entanglement**, not minimal local entropy.

```
Mode A — H_A (coevolved, Hermitian)     \
                                          >  ψ_AB  —  U = exp(-i H_total dt)
Mode B — H_B (coevolved, Hermitian)     /
                    ↑
            H_int = g (a†_A a_B + h.c.)   — fixed coupling only
```

**Each training generation:** same initial product state for the whole population → rollout → rank by fitness → mutate survivors → optional checkpoint. No ψ memory across generations.

-----

## State Representation

### Field (per organism)

A complex vector in Fock space — superposition over energy levels:

```
ψ = [c_0, c_1, c_2, ..., c_{n-1}]    complex vector, length n
```

Where `c_k` is the complex amplitude for k quanta.

- **Magnitude** `|c_k|²` — probability of finding k quanta
- **Phase** `arg(c_k)` — interference information
- **Normalization** `Σ |c_k|² = 1` always

### Joint State

```
ψ_AB = ψ_A ⊗ ψ_B    complex vector, length n²
```

### Parameters

```
n = 16              field truncation (max quanta per organism)
dtype = complex64   sufficient precision, half memory of complex128
device = mps        Apple Silicon GPU
```

-----

## Hamiltonians

### Ladder Operators (STATIC)

```python
i = torch.arange(1, n)
a  = torch.diag(torch.sqrt(i.float()), diagonal=-1)   # ladder (see note)
ad = a.conj().T                                       # adjoint
```

**Note:** Variable names `a` / `ad` in `physics.py` are adjoints of each other; the coupling uses `kron(ad, a) + kron(a, ad)`, which is the standard beam-splitter form \(g(a^\dagger_A a_B + a_A a^\dagger_B)\) for A as the first tensor factor and B as the second.

### Internal Hamiltonians (TRAINED)

Trainable parameters are unconstrained complex matrices M_A, M_B. Project to Hermitian at each step:

```python
H_A = (M_A + M_A.conj().T) / 2
H_B = (M_B + M_B.conj().T) / 2
```

This guarantees:

- Real eigenvalues (physical energy levels)
- Unitary time evolution
- Energy conservation

### Coupling Hamiltonian (STATIC)

Beam splitter interaction — one quantum hops between fields:

```python
g = 0.1   # coupling strength — only free physics parameter
H_int = g * (torch.kron(ad, a) + torch.kron(a, ad))
```

Physical meaning:

- One term transfers a quantum of excitation from B to A; the other from A to B
- Hermitian sum ⇒ energy-conserving exchange (beam splitter / hopping)
- `g` is the only fixed coupling constant in config (`G = 0.1`)

### Total Hamiltonian

```python
I = torch.eye(n, dtype=dtype, device=device)
H_total = torch.kron(H_A, I) + torch.kron(I, H_B) + H_int
```

Dimension: n² × n² = 256 × 256 for n=16.

-----

## Time Evolution

Unitary Schrödinger evolution (`physics.py`):

```python
dt = 0.01
U = matrix_exp_unitary(H_total, dt)   # torch.matrix_exp on CPU; scipy.linalg.expm fallback
psi_new = normalize(U @ psi_AB)         # redundant normalize; guards numerical drift
```

Properties guaranteed by unitarity:

- Norm and global purity preserved (`ρ_AB` remains pure)
- Time reversibility (no intrinsic dissipation)
- Subsystem von Neumann entropy **can increase or decrease** during rollout (entanglement redistribution)

**Not guaranteed / not modeled:** thermodynamic heat, entropy production in a bath, or irreversible “waste export.” In this closed model, what one subsystem “loses” in purity often appears as **correlation** with the partner, not heat leaving the universe.

-----

## Entropy Measures

### Density Matrix

```python
rho_AB = torch.outer(psi_AB, psi_AB.conj())   # n² × n²
```

### Reduced Density Matrix (trace out A)

With `psi_AB.reshape(n, n)` indexing **i** = A level, **j** = B level:

```python
rho_reshaped = rho_AB.reshape(n, n, n, n)   # indices i, j, i', k  (A, B, A, B)
rho_B = torch.einsum('ijik->jk', rho_reshaped)   # trace over A → n × n
```

(`partial_trace` in `physics.py`; logged metric uses the same trace and labels the result `entanglement` / \(S\).)

For any **pure** bipartite \|ψ⟩, \(S(\rho_A) = S(\rho_B)\). Fitness and logs use this bipartite entanglement entropy (natural log, nats).

### Von Neumann Entropy (fitness input)

```python
def von_neumann_entropy(rho):
    eigenvalues = torch.linalg.eigvalsh(rho).real
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -torch.sum(eigenvalues * torch.log(eigenvalues))
```

Interpretation in this project:

- **Relational** — entanglement between A and B (primary)
- **Not** dissipative thermodynamic entropy (no bath, global state stays pure)
- Participation ratio and drive maps (viz) supplement \(S\) for “structure” and symmetry breaking

### Complexity Measures (tracked, not trained)

Logged each generation for the **best organism’s evolved state** `psi_final` (not the initial vacuum). Implemented in `track_complexity`:

```python
def track_complexity(psi_AB, M_A, n):
    H_A = hermitian(M_A)
    rho_AB = torch.outer(psi_AB, psi_AB.conj())
    rho_reduced = partial_trace(rho_AB, n)   # trace out A → ρ_B; S(ρ_A)=S(ρ_B) for pure ψ

    S_ent = von_neumann_entropy(rho_reduced)
    probs = torch.abs(psi_AB) ** 2
    PR = 1.0 / torch.sum(probs**2)

    H_offdiag = H_A - torch.diag(torch.diag(H_A))
    structure = torch.sqrt(torch.sum(torch.abs(H_offdiag) ** 2)) / torch.sqrt(
        torch.sum(torch.abs(H_A) ** 2)
    )

    eigs = torch.linalg.eigvalsh(rho_reduced.cpu()).real
    spread = torch.std(eigs)

    return {
        "entanglement": S_ent.item(),
        "participation_ratio": PR.item(),
        "hamiltonian_structure": structure.item(),
        "eigenvalue_spread": spread.item(),
    }
```

**Note:** Console `entanglement` is raw bipartite \(S\) (nats). **Fitness** uses the parabolic score below, not raw \(S\) alone.

**Recommended reporting metrics (not all implemented in code yet):**

- \(\mathcal{O}_S = f(S) / f(S^*)\) — fraction of ideal regulated entanglement
- \(|S - S^*|\) with \(S^* = \frac{1}{2}\ln n\)
- Functional asymmetry of A/B drive maps on `psi_final` (symmetry breaking)
- \(\Delta S\) over rollout (`S(t)` in grapher)

-----

## Fitness Function

Fitness rewards **regulated entanglement** via a parabolic function of bipartite subsystem entropy \(S\) (from `reduced_entropy_from_psi`, trace out A):

\[
f(S) = S \cdot (\log n - S)
\]

Peak at \(S^* = \frac{1}{2}\log n\) (ideal intermediate entanglement).

### Single step

```python
def evolution_step(M_A, M_B, psi_AB, n, g, dt, device):
    H_total, _ = total_hamiltonian(M_A, M_B, n, g, dtype, device)
    U = matrix_exp_unitary(H_total, dt)
    psi_new = normalize(U @ psi_AB)
    S_after = reduced_entropy_from_psi(psi_new, n)
    return parabolic_entropy(S_after, n).item(), psi_new
```

### Rollout fitness (used for selection)

For `steps_per_eval` evolution steps (default 100):

```python
def rollout_fitness(M_A, M_B, psi_AB, n, steps, g, dt, device):
    step_sum = 0.0
    state = psi_AB.clone()
    for _ in range(steps):
        step_score, state = evolution_step(M_A, M_B, state, n, g, dt, device)
        step_sum += step_score
    S_final = reduced_entropy_from_psi(state, n)
    return parabolic_entropy(S_final, n).item() + step_sum / steps, state
```

**Total fitness** = parabolic score at **final** \(S\) plus **mean** parabolic score over the rollout.

Population evaluation is **batched** (`rollout_fitness_batch`) for speed.

-----

## Configuration (`quantum/config.py`)

| Constant | Default | Role |
|----------|---------|------|
| `N` | 16 | Fock truncation per organism |
| `POPULATION_SIZE` | 50 | Population |
| `MUTATION_RATE` | 0.01 | Gaussian noise on survivors when refilling |
| `N_GENERATIONS` | 10000 | Default training length |
| `STEPS_PER_EVAL` | 100 | Physics steps per fitness evaluation |
| `G` | 0.1 | Beam-splitter coupling |
| `DT` | 0.01 | Unitary step size |
| `DEVICE_TYPE` | `"mps"` | `"mps"`, `"cuda"`, or CPU fallback via `device.py` |
| `LOG_INTERVAL` | 1 | Print every generation |
| `INITIAL_STATE` | `"vacuum"` | `"vacuum"` (\|00⟩) or `"random"` product state each generation |
| `TRAIN_HEADLESS` | `True` | No matplotlib unless `--graph` |
| `SIM_FRAME_MS` | 150 | `--load` animation frame interval |
| `POSITION_X_MAX` | 4.0 | Position grid extent for heatmaps |
| `BASIN_N_VALUES` | [4, 8, 16] | Basin experiment grid |
| `BASIN_N_TRIALS` | 100 | Trials per n |
| `BASIN_N_GENERATIONS` | 1000 | Generations per basin trial |

-----

## Neuroevolution (`quantum/evolution.py`)

### Population

Each organism is `(M_A, M_B)` — complex `n×n` matrices, initialized `randn * 0.1`.

### One generation (`run_generation`)

1. **Fresh initial state** — `initial_product_state(n, …, INITIAL_STATE)` (default vacuum: amplitude 1 at index 0, i.e. \|00⟩).
2. **Batch evaluate** all organisms with `rollout_fitness_batch` on the same `psi_initial`.
3. **Select** top half by fitness; **mutate** copies of survivors to refill population.
4. **Log complexity** from `track_complexity(psi_final, best_M_A, n)` where `psi_final` is the best organism’s state after the rollout.
5. Return `(population, best_fitness, complexity, best_M_A, best_M_B, psi_initial, psi_final)`.

### Training loop (`train`)

```python
for generation in range(start_generation, n_generations):
    population, best_fitness, complexity, best_M_A, best_M_B, psi_initial, psi_final = (
        run_generation(population, n, steps_per_eval, g, dt, device, dtype)
    )
    history.append({"generation": generation, "best_fitness": best_fitness, **complexity})
    save_checkpoint(generation, best_fitness, population, history,
                    best_M_A, best_M_B, n, population_size, steps_per_eval, g, dt)
    # optional: grapher.update_generation(..., psi_final, ...)
```

- **Fresh `--train`:** clears `quantum/data/quantum_gen*.pt`, generation 0..N-1.
- **`--train --load`:** restores `population` and `history`, resumes at `generation + 1`; requires matching `n`. If `POPULATION_SIZE` differs from the checkpoint, the loaded population is **resized** (rank by one rollout, keep top organisms or refill with mutations).
- **pytest:** does not write checkpoints to `quantum/data/`; tests use an isolated temp directory.

### Basin driver (`run_evolution`)

Seeds a small population with one `(M_A, M_B)` pair, runs `run_generation` for `n_generations`, returns final `complexity` dict (or `None` if `n_generations == 0`).

-----

## Command-line interface (`python3 -m quantum`)

Run from repository root `OrganismEvolution/`.

| Flag | Behavior |
|------|----------|
| `--train` | Neuroevolution for `--generations` (default 10000) |
| `--graph` | Live matplotlib during `--train` (overrides `TRAIN_HEADLESS`) |
| `--load` | With `--train`: resume from latest checkpoint. **Alone:** animate saved best organism |
| `--basin` | `basin_size_experiment()` |
| `--generations N` | Override `N_GENERATIONS` |

Examples:

```bash
python3 -m quantum --train
python3 -m quantum --train --graph
python3 -m quantum --train --load
python3 -m quantum --load
python3 -m quantum --basin
```

-----

## Checkpoints (`quantum/checkpoint.py`)

- **Directory:** `quantum/data/`
- **Filename:** `quantum_gen{generation+1}_{best_fitness:.6f}.pt`  
  Example: `quantum_gen27_0.411145.pt` = completed generation 26.
- **Load rule:** highest generation number in filename (not highest fitness, not mtime).

**Saved fields:**

```python
{
    "generation": int,
    "population": [(M_A_cpu, M_B_cpu), ...],
    "best_M_A": Tensor,          # best pre-mutation pair for this gen (matches --graph)
    "best_M_B": Tensor,
    "history": list[dict],
    "n": int,
    "population_size": int,
    "steps_per_eval": int,
    "g": float,
    "dt": float,
}
```

Checkpoints without `best_M_A` / `best_M_B` cannot be used with `--load` (re-save after one training generation on current code).

---

## Visualization (`quantum/viz.py`, `grapher.py`, `sim.py`)

### Fock indexing

`psi_AB.reshape(n, n)` → amplitude at row **i** (A level), column **j** (B level).  
`H_A` acts on rows; `H_B` on columns.

### Position maps (heatmaps)

Coefficients are mapped to position samples \(x, y \in [-\) `POSITION_X_MAX` \(, +\) `POSITION_X_MAX` \(]\) using harmonic-oscillator eigenfunctions \(\phi_k(x)\):

\[
\psi(x,y) = \sum_{i,j} c_{ij}\,\phi_i(x)\,\phi_j(y)
\]

Vacuum \|00⟩ appears as a **Gaussian blob at the center** of the \(x,y\) grid, not a corner pixel.

### Training UI (`--train --graph`)

**Top row**

- **Left:** \(|\psi(x,y)|^2\) at end of rollout (`psi_final`), magma colormap.
- **Right:** A vs B **drive** on the same plane — red = \(\|H_A\psi\|^2\), blue = \(\|\psi H_B\|^2\) (position-mapped), purple where both strong.

**Bottom row**

- Fitness vs generation, entanglement \(S\) vs generation, \(S(t)\) rollout for current best.

### Load animation (`--load`)

Loads `best_M_A`, `best_M_B` from latest checkpoint; **re-rollouts from vacuum** for `STEPS_PER_EVAL` from `config.py` (not the checkpoint’s saved step count). Same top layout + \(S(t)\) below. Does not restore ψ from file.

-----

## Scaling Experiment (`--basin`)

```bash
python3 -m quantum --basin
```

```python
def basin_size_experiment(
    n_values=[4, 8, 16],
    n_trials=100,
    n_generations=1000,
):
    threshold(n) = 0.1 * log(n)
    # For each n, random M_A/M_B, run_evolution (selection + physics) → fraction with
    # final_complexity["entanglement"] > threshold(n)
```

**What this measures:** **Evolvability** — how often random *Hamiltonian pairs*, after `n_generations` of coevolution with a tiny population, reach entanglement above a low threshold. It is **not** the fraction of random *initial states* that organize under a **fixed** \(H\) without selection.

**Thesis-style success criterion (recommended):** replace or supplement the loose threshold with \(\mathcal{O}_S > 0.9\) or \(|S - S^*| < \varepsilon \ln n\), and plot convergence fraction vs `n`.

-----

## Memory and Performance

### Memory per organism pair at n=16

```
ψ_A, ψ_B          2 × 16 × 8 bytes  = 256 bytes
ψ_AB               256 × 8 bytes     = 2KB
H_A, H_B           2 × 256 × 8 bytes = 4KB
H_total            65536 × 8 bytes   = 512KB
M_A, M_B           2 × 256 × 8 bytes = 4KB
Total                                ~ 520KB
```

### Scaling

```
n=16     H_total 256×256      microseconds per step    — development
n=32     H_total 1024×1024    milliseconds per step    — experiments
n=64     H_total 4096×4096    tens of ms per step      — overnight runs
n=128    H_total 16384×16384  seconds per step         — weekend runs
```

### M3 Pro limits

```
Memory ceiling     ~n=180 (18GB config) or ~n=220 (36GB config)
Practical limit    n=128 with careful batching
Recommended start  n=16
```

-----

## What To Measure

| Metric | Tracks | Expected / observed |
|--------|--------|---------------------|
| Fitness | \(f(S) + \) mean step score; max \(\approx \tfrac{1}{4}(\ln n)^2\) | Increases during training if selection works |
| \(\mathcal{O}_S = f(S)/f(S^*)\) | Regulated entanglement (report) | Rises toward 1 if optimizing |
| Raw \(S\) (log) | Bipartite entanglement (nats) | Rises from ~0 at \|00⟩; target \(S^* = \tfrac{1}{2}\ln n\) |
| \(S(t)\) rollout | Dynamics per episode | Often increases during steps |
| Participation ratio | Spread of \|c_ij\|² on Fock grid | Often rises with structure |
| Hamiltonian structure | Off-diagonal fraction of \(H_A\) | Usually high even at gen 0 — weak emergence signal |
| Symmetry breaking | A vs B drive maps / \(\|H_A - H_B\|\) | Visual and metric divergence over generations |
| Basin vs `n` | `--basin` convergence fraction | **Hypothesis:** increases with `n`; must be measured |

-----

## Key Questions (scoped to current sim)

1. Does selection discover \((H_A, H_B)\) that increase \(\mathcal{O}_S\) from vacuum faster than random pairs?
1. Does raw \(S\) approach \(S^* = \frac{1}{2}\ln n\) under continued training?
1. Do A and B **functionally** differentiate (drive maps, effective roles) even when \(S\) is still low?
1. Does `--basin` convergence fraction grow with `n` (evolvability scaling)?
1. Are results stable under sweeps of `G`, `DT`, and `STEPS_PER_EVAL`?
1. (Future) Does fixing the best \((H_A, H_B)\) and varying only initial \(\psi\) show a large **dynamical** basin?

-----

## Implementation layout

```
quantum/
  config.py         # constants above
  device.py         # get_device(), get_dtype()
  physics.py        # Hamiltonians, evolution_step, rollout_fitness, track_complexity
  evolution.py      # train(), run_generation(), evaluate_population()
  checkpoint.py     # save/load quantum/data/quantum_gen*.pt
  viz.py            # position maps, entropy trajectory, drive RGB
  grapher.py        # --train --graph UI
  sim.py            # --load animation
  basin.py          # --basin experiment
  __main__.py       # CLI
  test_quantum.py
  test_e2e_physics.py
  conftest.py       # pytest: isolated checkpoint dir, cleanup stray files
  data/             # training checkpoints (gitignored in practice)
```

## Dependencies

```python
torch          # >= 2.0; MPS/CUDA/CPU via device.py
scipy          # Hermite functions (viz); matrix_exp fallback
numpy          # visualization grids
matplotlib     # --graph and --load only
pytest         # tests (52)
```

## Tests

```bash
python3 -m pytest quantum/ -q
```

-----


## Differences From Prior Work

| System | This project |
|--------|----------------|
| Standard coupled QHO | Coevolved \(H_A, H_B\); selection on entanglement |
| Quantum reservoir computing | Full unitary rollout on joint ψ, not readout-only training |
| Lenia / classical ALife | Complex QM on a finite grid; explicit Hermitian generators |
| England dissipative adaptation | **Closed** unitary; no dissipation or entropy production |
| Variational quantum eigensolver | Population evolution of Hamiltonians, not fixed ansatz |
| Open-system / Lindblad life models | No environment channel in current code |

-----

## Potential Contributions

- Minimal **closed** quantum coevolution model: fixed coupling + learned local Hamiltonians + explicit fitness
- Demonstration that **regulated entanglement** and **joint structure** in \((x,y)\) maps are selectable from vacuum
- Evidence for **functional symmetry breaking** between coupled modes (drive asymmetry)
- Optional scaling study: **evolvability** of organized entanglement vs Fock dimension `n`
- Conceptual bridge toward multi-layer “consolidation” and open-system export (future work; not in this repo)

## Future extensions (not implemented)

- Extra modes C, D as consolidator layer; hierarchical partial traces
- Per-subsystem objectives (e.g. low local \(S\)) competing with coupling export
- Open-system channels (Lindblad) for literal dissipative thermodynamics
- Dynamical basin tests: fixed \((H_A, H_B)\), random initial \(\psi\) only