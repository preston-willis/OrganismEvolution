"""
Band-resolved Onsager thermodynamics for the organism simulation.

# ---------------------------------------------------------------------------
# State (per grid cell)
# ---------------------------------------------------------------------------
#   e_f     energy in frequency band f  (tensor shape F×H×W)
#   b_f     spectrum amplitude (softmax consumer weights), Σ_f b_f = 1
#   τ       terrain energy in [0, 1]
#   ξ       topology mask (1 = organism, 0 = empty)
#
#   Total cell energy:  e = Σ_f e_f   with  0 ≤ e ≤ 1
#
# ---------------------------------------------------------------------------
# Thermodynamic primitives
# ---------------------------------------------------------------------------
#   Binary mixing entropy (per unit energy e):
#       S_mix(e) = −[ e ln e + (1−e) ln(1−e) ]
#
#   Chemical potential (ideal lattice gas):
#       μ(e) = −ln( e / (1−e) )
#
#   Onsager conductance (per band, per cell):
#       L_f = (1 / T_env) · σ · κ · ξ · conductance_scale
#       where σ = THERMO_CONDUCTANCE, T_env = THERMO_ENV_TEMPERATURE,
#             κ = PUMP_KAPPA_TERRAIN or PUMP_KAPPA_NEIGH by coupling type.
#
# ---------------------------------------------------------------------------
# Core flux equation (single law for all couplings)
# ---------------------------------------------------------------------------
#   Driving force (non-negative):  X = max(0, Δμ)
#
#   Entropy production rate:
#       σ̇ = L · X²
#
#   Maximum energy flux through the link:
#       J_max = T_env · σ̇
#
#   Raw transferable amount:
#       J_raw = min(source, capacity, J_max)
#
#   Consumer gate (spectrum / receiver):
#       χ = 1  if consumer > ε, else 0;  then χ ← clip(consumer, 0, 1)
#
#   Useful transfer (stored or moved):
#       ΔE_useful = η · χ · J_raw        (η = THERMO_ETA)
#
#   Irreversible sink (heat bath / destroyed bucket):
#       ΔE_sink = J_max − ΔE_useful
#
#   If χ = 0: no useful transfer; all of J_max goes to the sink.
#
# ---------------------------------------------------------------------------
# Couplings (each applies band_flux with its own X, source, capacity, χ)
# ---------------------------------------------------------------------------
#
#   1) Terrain harvest (broadband, shared τ per cell)
#       X_f = max(0, μ(e_f) − μ(τ))
#       source = τ,  capacity = (1 − e_f) ξ,  χ = 1
#       e_f ← e_f + ΔE_useful_f
#       τ debit: min(τ, Σ_f (ΔE_useful_f + ΔE_sink_f)) with proportional
#                scaling when Σ_f flux exceeds τ
#
#   2) Neighbor exchange (per band f)
#       μ̄_f = masked mean of μ(e_f) over 8 toroidal neighbors
#       Outflow:  X = max(0, μ̄_f − μ(e_f)),  source = e_f,  χ = b_f
#       Inflow:   X = max(0, μ(e_f) − μ̄_f),  source = ē_f (neighbor mean),
#                 capacity = (1 − e_f) ξ,  χ = b_f
#
#   3) Reproduction (per band f)
#       X_f = max(0, μ(0) − μ(e_f))
#       source = e_f · b_f,  capacity = e_f,  χ = 1 on emit
#       transfer_f deposited on empty neighbors via direction weights
#
# ---------------------------------------------------------------------------
# Capacity constraint
# ---------------------------------------------------------------------------
#   After flux updates: clip e_f to [0,1] on occupied cells; if Σ_f e_f > 1,
#   scale bands proportionally. Lost energy → clamp_loss (destroyed bucket).
#
# ---------------------------------------------------------------------------
# Tick composition
# ---------------------------------------------------------------------------
#   Harvest tick:  terrain_harvest + neighbor_exchange + clamp
#   Repro tick:    reproduction + spread_ring + clamp
#   Full tick:     harvest then repro
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config import (
    CNN_HIDDEN_CHANNELS,
    ENTROPY_EPSILON,
    PUMP_KAPPA_NEIGH,
    PUMP_KAPPA_TERRAIN,
    SPECTRUM_CONSUMER_EPSILON,
    THERMO_CONDUCTANCE,
    THERMO_ENV_TEMPERATURE,
    THERMO_ETA,
)

_NEIGHBOR_OFFSETS = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def mixing_entropy(energy):
    """
    Binary mixing entropy per cell.

    S_mix(e) = −[ e ln e + (1−e) ln(1−e) ]

    e is clamped to (ε, 1−ε) before the logarithm.
    """
    e = torch.clamp(energy, ENTROPY_EPSILON, 1.0 - ENTROPY_EPSILON)
    return -(e * torch.log(e) + (1.0 - e) * torch.log(1.0 - e)).nan_to_num(0.0)


def chemical_potential(energy):
    """
    Chemical potential of a binary lattice-gas energy level.

    μ(e) = −ln( e / (1−e) )

    e is clamped to (ε, 1−ε) before the ratio.
    """
    e = torch.clamp(energy, ENTROPY_EPSILON, 1.0 - ENTROPY_EPSILON)
    return -torch.log((e + ENTROPY_EPSILON) / (1.0 - e + ENTROPY_EPSILON))


def total_energy(energy_bands):
    """
    Total cell energy from band decomposition.

    e = Σ_f e_f   (sum over band dimension 0)
    """
    return energy_bands.sum(dim=0)


def band_conductance(kappa, topology, conductance_scale, n_bands):
    """
    Onsager conductance L_f per band and cell.

    L_f = (1 / T_env) · σ · κ · ξ · conductance_scale

    Returns shape (F, H, W) with ξ = topology.
    """
    if isinstance(conductance_scale, torch.Tensor):
        scale = float(conductance_scale.item())
    else:
        scale = float(conductance_scale)
    base = (1.0 / THERMO_ENV_TEMPERATURE) * THERMO_CONDUCTANCE * scale * kappa
    conductance = torch.tensor(base, device=topology.device, dtype=topology.dtype)
    return conductance.unsqueeze(0).unsqueeze(0).expand(n_bands, -1, -1) * topology.unsqueeze(0)


def band_flux(driving_force, source_energy, capacity, conductance, consumer):
    """
    Core Onsager energy flux (one equation for every coupling).

    X = max(0, driving_force)
    σ̇ = L · X²
    J_max = T_env · σ̇
    J_raw = min(source, capacity, J_max)

    χ = clip(consumer, 0, 1) when consumer > ε, else 0
    ΔE_useful = η · χ · J_raw
    ΔE_sink = J_max − ΔE_useful

    Returns (ΔE_useful, ΔE_sink, σ̇).
    """
    x = torch.clamp(driving_force, min=0.0)
    sigma_dot = conductance * x * x
    max_transfer = THERMO_ENV_TEMPERATURE * sigma_dot
    useful_raw = torch.minimum(
        source_energy,
        torch.minimum(capacity, max_transfer),
    )
    chi = (consumer > SPECTRUM_CONSUMER_EPSILON).float() * torch.clamp(consumer, 0.0, 1.0)
    useful = torch.clamp(THERMO_ETA * chi * useful_raw, 0.0, 1.0)
    sink = torch.clamp(max_transfer - useful, 0.0, 1.0)
    return useful, sink, sigma_dot


def neighbor_mean_8(field, mask, neighbor_kernel):
    """
    Toroidal 8-neighbor masked average.

    f̄(i) = Σ_{j ∈ N₈(i)} mask(j) f(j) / Σ_{j ∈ N₈(i)} mask(j)

    Isolated occupied cells (no masked neighbors) keep f(i).
    """
    masked = field * mask
    padded = F.pad(masked.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular")
    num = F.conv2d(padded, neighbor_kernel).squeeze(0).squeeze(0)
    mask_padded = F.pad(mask.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="circular")
    den = F.conv2d(mask_padded, neighbor_kernel).squeeze(0).squeeze(0)
    has_neighbors = den > ENTROPY_EPSILON
    mean = torch.where(has_neighbors, num / den, torch.zeros_like(field))
    isolated = (~has_neighbors) & (mask > 0)
    return torch.where(isolated, field, mean)


def spread_ring_flux_bands(flux_bands, direction_weights, topology):
    """
    Route emitted band flux to empty 8-neighbors.

    pending_f(x′) += flux_f(x) · w_dir(x, x′) · (1 − ξ(x′))

    direction_weights has shape (8, H, W); flux_bands shape (F, H, W).
    """
    pending = torch.zeros_like(flux_bands)
    h, w = flux_bands.shape[1], flux_bands.shape[2]
    y_grid = torch.arange(h, device=flux_bands.device).view(h, 1)
    x_grid = torch.arange(w, device=flux_bands.device).view(1, w)
    empty = (1.0 - topology).unsqueeze(0)
    for idx, (dy, dx) in enumerate(_NEIGHBOR_OFFSETS):
        ny = (y_grid + dy) % h
        nx = (x_grid + dx) % w
        pending[:, ny, nx] = pending[:, ny, nx] + flux_bands * direction_weights[idx].unsqueeze(0) * empty[:, ny, nx]
    return pending


def terrain_harvest_bands(energy_bands, spectrum, terrain, topology, conductance_scale):
    """
    Broadband terrain → organism harvest (all bands share one τ per cell).

    X_f = max(0, μ(e_f) − μ(τ))
    L_f from PUMP_KAPPA_TERRAIN; capacity_f = (1 − e_f) ξ; χ = 1

    e_f ← e_f + ΔE_useful_f
    τ_debit = min(τ, Σ_f (ΔE_useful_f + ΔE_sink_f)), with uniform scale on flux
              when Σ_f (useful + sink) exceeds τ.
    """
    n_bands = energy_bands.shape[0]
    mu_terrain = chemical_potential(terrain)
    mu_bands = chemical_potential(energy_bands)
    driving = torch.clamp(mu_bands - mu_terrain.unsqueeze(0), min=0.0)
    conductance = band_conductance(PUMP_KAPPA_TERRAIN, topology, conductance_scale, n_bands)
    capacity = (1.0 - energy_bands) * topology.unsqueeze(0)
    consumer = torch.ones_like(driving)
    useful, sink, _ = band_flux(driving, terrain.unsqueeze(0), capacity, conductance, consumer)
    flux = useful + sink
    total_flux = flux.sum(dim=0)
    terrain_debit = torch.minimum(terrain, total_flux)
    scale = torch.where(
        total_flux > ENTROPY_EPSILON,
        terrain_debit / total_flux,
        torch.zeros_like(total_flux),
    )
    scale = scale.unsqueeze(0)
    useful = useful * scale
    sink = sink * scale
    energy_bands = energy_bands + useful
    return energy_bands, terrain_debit, sink


def neighbor_exchange_bands(energy_bands, spectrum, topology, neighbor_kernel, conductance_scale):
    """
    Per-band equilibration with 8-neighbor mean field.

    For each band f:
      μ̄_f = neighbor_mean_8(μ(e_f), ξ)

      Outflow: X = max(0, μ̄_f − μ(e_f)), source = e_f, χ = b_f
               e_f ← e_f − ΔE_useful_out

      Inflow:  X = max(0, μ(e_f) − μ̄_f), source = ē_f, capacity = (1−e_f) ξ, χ = b_f
               e_f ← e_f + ΔE_useful_in

      sink_f = ΔE_sink_out + ΔE_sink_in
    """
    n_bands = energy_bands.shape[0]
    topo = topology
    conductance = band_conductance(PUMP_KAPPA_NEIGH, topology, conductance_scale, n_bands)
    capacity = (1.0 - energy_bands) * topo.unsqueeze(0)
    sink = torch.zeros_like(energy_bands)

    for f in range(energy_bands.shape[0]):
        e_f = energy_bands[f]
        b_f = spectrum[f]
        mu = chemical_potential(e_f)
        mu_bar = neighbor_mean_8(mu, topo, neighbor_kernel)
        drive_out = torch.clamp(mu_bar - mu, min=0.0)
        useful_out, sink_out, _ = band_flux(
            drive_out, e_f, e_f * topo, conductance[f], b_f
        )
        energy_bands[f] = energy_bands[f] - useful_out

        drive_in = torch.clamp(mu - mu_bar, min=0.0)
        e_bar = neighbor_mean_8(e_f, topo, neighbor_kernel)
        useful_in, sink_in, _ = band_flux(
            drive_in, e_bar, capacity[f], conductance[f], b_f
        )
        energy_bands[f] = energy_bands[f] + useful_in
        sink[f] = sink_out + sink_in

    return energy_bands, sink


def reproduction_bands(
    energy_bands,
    spectrum,
    topology,
    direction_weights,
    conductance_scale,
):
    """
    Band-resolved reproduction flux toward empty neighbors.

    X_f = max(0, μ(0) − μ(e_f))
    source_f = e_f · b_f,  capacity_f = e_f,  χ = 1

    e_f ← e_f − transfer_f
    pending = spread_ring_flux_bands(transfer, direction_weights, ξ)
    """
    n_bands = energy_bands.shape[0]
    mu_bands = chemical_potential(energy_bands)
    mu_zero = chemical_potential(torch.zeros((), device=energy_bands.device, dtype=energy_bands.dtype))
    driving = torch.clamp(mu_zero - mu_bands, min=0.0)
    conductance = band_conductance(PUMP_KAPPA_NEIGH, topology, conductance_scale, n_bands)
    source = energy_bands * spectrum
    capacity = energy_bands
    consumer_emit = torch.ones_like(driving)
    transfer, sink, _ = band_flux(driving, source, capacity, conductance, consumer_emit)
    energy_bands = energy_bands - transfer
    pending = spread_ring_flux_bands(transfer, direction_weights, topology)
    return energy_bands, pending, sink


@dataclass
class FluxTickResult:
    energy_bands: torch.Tensor
    terrain_debit: torch.Tensor
    pending_bands: torch.Tensor
    sink_bands: torch.Tensor
    clamp_loss: torch.Tensor


def _clamp_bands_to_capacity(energy_bands, topology):
    """
    Enforce 0 ≤ e_f ≤ 1 on occupied cells and Σ_f e_f ≤ 1 per cell.

    If Σ_f e_f > 1: scale all bands at that cell by 1 / Σ_f e_f.
    clamp_loss = Σ_cells max(0, e_before − e_after)  (→ destroyed bucket).
    """
    before = total_energy(energy_bands)
    topo = topology.unsqueeze(0)
    energy_bands = torch.clamp(energy_bands * topo, 0.0, 1.0)
    total = total_energy(energy_bands)
    over = total > 1.0
    if over.any():
        scale = torch.where(over, 1.0 / total.clamp(min=ENTROPY_EPSILON), torch.ones_like(total))
        energy_bands = energy_bands * scale.unsqueeze(0) * topo
    after = total_energy(energy_bands)
    clamp_loss = torch.clamp(before - after, min=0.0).sum()
    return energy_bands, clamp_loss


@dataclass
class HarvestTickResult:
    energy_bands: torch.Tensor
    terrain_debit: torch.Tensor
    sink_bands: torch.Tensor
    clamp_loss: torch.Tensor


def apply_harvest_flux_tick(
    energy_bands,
    spectrum,
    terrain,
    topology,
    neighbor_kernel,
    conductance_scale,
):
    """
    Harvest-phase tick: terrain coupling + neighbor exchange + capacity clamp.

    Combines terrain_harvest_bands and neighbor_exchange_bands; sink_bands and
    clamp_loss are summed for the destroyed-energy accounting.
    """
    energy_bands, terrain_debit, sink_h = terrain_harvest_bands(
        energy_bands, spectrum, terrain, topology, conductance_scale
    )
    energy_bands, sink_n = neighbor_exchange_bands(
        energy_bands, spectrum, topology, neighbor_kernel, conductance_scale
    )
    energy_bands, clamp_loss = _clamp_bands_to_capacity(energy_bands, topology)
    return HarvestTickResult(
        energy_bands=energy_bands,
        terrain_debit=terrain_debit,
        sink_bands=sink_h + sink_n,
        clamp_loss=clamp_loss,
    )


def apply_repro_flux_tick(
    energy_bands,
    spectrum,
    topology,
    direction_weights,
    conductance_scale,
):
    """
    Reproduction-phase tick: emit toward empty neighbors + capacity clamp.

    Applies reproduction_bands (spread_ring_flux_bands); terrain_debit is zero.
    """
    energy_bands, pending, sink_r = reproduction_bands(
        energy_bands, spectrum, topology, direction_weights, conductance_scale
    )
    energy_bands, clamp_loss = _clamp_bands_to_capacity(energy_bands, topology)
    return FluxTickResult(
        energy_bands=energy_bands,
        terrain_debit=torch.zeros_like(topology),
        pending_bands=pending,
        sink_bands=sink_r,
        clamp_loss=clamp_loss,
    )


def apply_flux_tick(
    energy_bands,
    spectrum,
    terrain,
    topology,
    neighbor_kernel,
    direction_weights,
    conductance_scale,
):
    """
    Full physics tick: harvest phase then reproduction phase.

    Returns combined terrain_debit, pending_bands, sink_bands, and clamp_loss.
    """
    harvest = apply_harvest_flux_tick(
        energy_bands, spectrum, terrain, topology, neighbor_kernel, conductance_scale
    )
    repro = apply_repro_flux_tick(
        harvest.energy_bands,
        spectrum,
        topology,
        direction_weights,
        conductance_scale,
    )
    return FluxTickResult(
        energy_bands=repro.energy_bands,
        terrain_debit=harvest.terrain_debit,
        pending_bands=repro.pending_bands,
        sink_bands=harvest.sink_bands + repro.sink_bands,
        clamp_loss=harvest.clamp_loss + repro.clamp_loss,
    )
