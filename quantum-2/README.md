# Quantum-2: critical level-spacing evolution

Evolutionary search over **Pauli-string Hamiltonians**, then an automatic **criticality report**: spacing statistics, comparison to Ising/XXZ references, finite-size scaling, and `critical_hamiltonian.json`.

| Regime | Mean r |
|--------|--------|
| Poisson (integrable) | ≈ 0.386 |
| Critical target | ≈ 0.458 |
| Wigner–Dyson (chaotic) | ≈ 0.530 |

## Setup

```bash
pip install -r quantum-2/requirements.txt
```

## Run (default: A→B transfer task + report)

From the **repo root**:

```bash
python3 quantum-2/__main__.py
```

Default **`--task transfer`**: evolve Hamiltonians that move excitation from subsystem **A** to **B** (vacuum init, driven rollout during training). Spacing-only training: `--task spacing`. Both: `--task combined`.

This will:

1. Run the GA (8 qubits, 200 generations by default)
2. Plot spectrum + spacing histogram + dominant terms → `quantum-2/output/spectrum_analysis.png`
3. Check mean **r** at N=4,6,8 (embedding extra qubits as **I**) → `quantum-2/output/finite_size_scaling.png`
4. Compare to reference Ising / XXZ models
5. Write **`quantum-2/output/critical_hamiltonian.json`** with the Hamiltonian, verdict, and all stats

### Quick smoke test

```bash
python3 quantum-2/__main__.py --generations 10
```

### Re-verify a saved genome (no re-training)

```bash
python3 quantum-2/__main__.py --load-genome quantum-2/output/critical_hamiltonian.json
```

### Evolution only (no report)

```bash
python3 quantum-2/__main__.py --evolve-only
```

### Show plots interactively

```bash
python3 quantum-2/__main__.py --show-plots
```

### Time evolution + field animation

Animate \(|\psi(t)\rangle = e^{-iHt}|\psi_0\rangle\) under the critical Hamiltonian. Left: **\(|\psi|^2\)** on a bipartition (first half vs second half of qubits). Right: **entanglement entropy** growing over time.

Full pipeline then animate:

```bash
python3 quantum-2/__main__.py --animate --show-plots
```

Animate an existing genome only (saves `quantum-2/output/field.gif`):

```bash
python3 quantum-2/__main__.py --animate-only
```

Disordered initial state instead of \(|0000\rangle\):

```bash
python3 quantum-2/__main__.py --animate-only --initial random --show-plots
```

AC drive (energy pump): \(H(t) = H_{\text{crit}} + A\cos(\omega t)\sum_i X_i\):

```bash
python3 quantum-2/__main__.py --animate-only --drive --show-plots
python3 quantum-2/__main__.py --animate-only --drive --drive-amp 0.8 --drive-omega 2.0 --initial vacuum --show-plots
```

## Output: `critical_hamiltonian.json`

```json
{
  "critical": true,
  "hamiltonian": "H = +0.42·ZZII + ...",
  "genome": [[coeff, "PauliString"], ...],
  "spacing": { "r_mean": 0.458, "r_std": ..., "frac_ratios_in_poisson_goe_band": ... },
  "finite_size_scaling": [{ "n_qubits": 4, "r_mean": ... }, ...],
  "reference_models": [{ "name": "Transverse Ising (chain)", "r_mean": ... }],
  "verdict": "..."
}
```

Use `"hamiltonian"` and `"genome"` as the certified critical Hamiltonian.

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--out-dir` | `quantum-2/output` | Report directory |
| `--generations` | 200 | GA generations |
| `--n-qubits` | 4 | Qubits during evolution |
| `--scale-sizes` | `4,6,8` | Finite-size check sizes |
| `--load-genome` | — | Skip evolution, verify JSON |
| `--evolve-only` | off | GA only |
| `--show-plots` | off | Open matplotlib windows |
| `--animate` | off | After report, run dynamics animation |
| `--animate-only` | off | Skip GA/report; animate saved Hamiltonian |
| `--steps` | 200 | Rollout steps |
| `--dt` | 0.05 | Time step |
| `--initial` | `vacuum` | `vacuum` or `random` |
| `--task` | `transfer` | `spacing`, `transfer`, or `combined` |
| `--no-train-drive` | off | Turn off drive during transfer training |
| `--save-animation` | — | e.g. `output/field.gif` |
| `--drive` | off | AC transverse field in animation |
| `--drive-amp` | 0.5 | Drive amplitude \(A\) |
| `--drive-omega` | 1.0 | Drive frequency \(\omega\) |

## Project layout

| File | Role |
|------|------|
| `__main__.py` | Evolve + run full pipeline |
| `critical_report.py` | Criticality checks, plots, JSON export |
| `evolution.py` | Genetic algorithm |
| `fitness.py` | Level-spacing ratio and fitness |
| `hamiltonian.py` | Build H from Pauli strings |
| `analyze.py` | Spectrum / spacing plots |
| `dynamics.py` | \(e^{-iHt}\) propagation, entanglement |
| `animate.py` | Field heatmap + entropy animation |
| `known.py` | Reference Ising / XXZ models |
