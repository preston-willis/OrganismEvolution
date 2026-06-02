# Quantum critical edge

Neuro-evolution searches for Hamiltonians at the **edge of chaos**: competing order and chaos pressures stay **balanced** while dynamics sustain that balance over time.

## Setup

- **Mode A (vacuum):** \(|0\rangle\), low entropy  
- **Mode B (disorder):** seeded random state, high entropy  
- **Initial:** \(|0\rangle_A \otimes |\psi_B^{\text{random}}\rangle\)  
- **Genome:** matrix `M` → \(H=\frac{1}{2}(M+M^\dagger)\) on both modes + fixed beam coupling `G`

## Fitness

### Balance skeleton (order × chaos − imbalance)

\[
F_{\text{bal}} = f_{\text{order}}\, f_{\text{chaos}} - \bigl|f_{\text{order}} - f_{\text{chaos}}\bigr|
\]

High when **both** signals are present and similar; penalized when one dominates.

### Static — \(F_{\text{static}}(H)\) (no time evolution)

From local \(H\) eigenvalues:

| Signal | Order \(f_{\text{order}}\) | Chaos \(f_{\text{chaos}}\) |
|--------|---------------------------|----------------------------|
| **Level-spacing ratio** \(r\) | near Poisson \(r\approx0.386\) | near Wigner–Dyson \(r\approx0.530\) |
| **IPR** (eigenstate localization) | high IPR | low IPR |
| **Spectral gap** | large gap / spread | small gap / spread |

Primary proxy: mean \(r_n = \min(\delta_n,\delta_{n+1})/\max(\delta_n,\delta_{n+1})\) over spacings.  
Target midpoint: \(r^* = \frac{1}{2}(r_{\text{Poisson}} + r_{\text{GOE}})\).

### Dynamic — \(F_{\text{dynamic}}\) (rollout)

From bipartite entanglement \(S(t)\) during rollout:

- \(f_{\text{order}} \propto 1 - S/S_{\max}\) (area-law side)  
- \(f_{\text{chaos}} \propto S/S_{\max}\) (volume-law side)  

Same balance formula, time-averaged over steps (+ final term).

### Total

\[
F = F_{\text{static}}(H) + \lambda\, F_{\text{dynamic}}(H)
\]

`DYNAMIC_LAMBDA` in `config.py` (default `0.3`).

## Run

```bash
python3 -m quantum --train
python3 -m quantum --train --graph
python3 -m quantum --load
python3 -m quantum --demo
```

## Config

| Key | Role |
|-----|------|
| `R_POISSON`, `R_GOE` | Integrable vs chaotic spacing limits |
| `DYNAMIC_LAMBDA` | Weight on rollout balance term (static spacing dominates when low) |
| `POPULATION_SIZE` | Default `20` |
| `MUTATION_RATE` | Default `0.1` |
| `ELITISM_FREE_GENERATIONS` | First N generations use tournament refill only (no clone-best elitism) |
| `DISORDER_SEED` | High-entropy wing B |
| `G`, `DT`, `N` | Coupling, step, Fock cutoff |

## Logs

```
Gen     0 | fitness: 0.12 | F_static: 0.05 | F_dynamic: 0.07 | r: 0.45 | r*: 0.46 | f_ord: 0.52 | f_chaos: 0.48 | S: 0.31
```

## Tests

```bash
python3 -m pytest quantum/ -q
```
