# Quantum-2: regulated entanglement + A→B flow

Evolve **Pauli-string Hamiltonians** under **B-pump / A-sink**. Fitness requires:

1. **Regulated entanglement** — mean \(S(A{:}B)\) in band \([S_\text{low}, S_\text{high}]\), low `var(S)` late
2. **A→B energy flow** — high **transfer** (\(p_B - \lambda p_A\)) from **vacuum and random** init (score = **min** over both)

## Run

```bash
python3 quantum-2/__main__.py
python3 quantum-2/__main__.py --animate-only --initial random --show-plots
python3 quantum-2/__main__.py --animate-only --initial vacuum --show-plots
```

## Fitness (simplified)

Per initial state, over steps 150–200:

```
score = mean(transfer)
      - S_BAND_PENALTY × (distance outside [S_LOW, S_HIGH])
      - S_VAR_PENALTY × var(S)
      - FLOW_PENALTY × max(0, FLOW_MIN - mean(transfer))
```

**Genome fitness** = `min(score_vacuum, score_random)` — must work from **any** init.

## Config

| Param | Default | Role |
|-------|---------|------|
| `S_LOW` / `S_HIGH` | 0.45 / 0.65 | entanglement band |
| `FLOW_MIN` | 0.35 | min mean transfer per init |
| `TRAIN_INITIALS` | vacuum, random | ICs used in training |

## Layout

| File | Role |
|------|------|
| `fitness.py` | Dual-IC regulated flow fitness |
| `dynamics.py` | Open-system rollout |
| `evolution.py` | GA |
| `report.py` | JSON + per-init eval |
