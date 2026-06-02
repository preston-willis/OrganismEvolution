import math

import torch

from quantum import config


def normalize(psi):
    return psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2))


def hermitian(M):
    return (M + M.mH) / 2


def ladder_operators(n, dtype, device):
    i = torch.arange(1, n, device=device).float()
    a = torch.diag(torch.sqrt(i), diagonal=-1).to(dtype=dtype, device=device)
    ad = a.conj().T
    return a, ad


def _hop_operators(n, dtype, device):
    a, ad = ladder_operators(n, dtype, device)
    a = a.contiguous()
    ad = ad.contiguous()
    hop_ab = torch.kron(ad, a)
    hop_ba = torch.kron(a, ad)
    return hop_ab, hop_ba


def coupling_hamiltonian(n, g, dtype, device):
    hop_ab, hop_ba = _hop_operators(n, dtype, device)
    return g * (hop_ab + hop_ba)


def total_hamiltonian(M, n, g, dtype, device):
    H = hermitian(M).contiguous()
    I = torch.eye(n, dtype=dtype, device=device)
    H_int = coupling_hamiltonian(n, g, dtype, device)
    return torch.kron(H, I) + torch.kron(I, H) + H_int, H


def batch_total_hamiltonian(M_batch, n, g, dtype, device):
    H = hermitian(M_batch).contiguous()
    I = torch.eye(n, dtype=dtype, device=device)
    H_int = coupling_hamiltonian(n, g, dtype, device)
    batch_size = M_batch.shape[0]
    return torch.stack(
        [
            torch.kron(H[i], I) + torch.kron(I, H[i]) + H_int
            for i in range(batch_size)
        ]
    )


def matrix_exp_unitary(H, dt):
    H_cpu = (-1j * H * dt).detach().cpu()
    try:
        U = torch.matrix_exp(H_cpu)
    except Exception:
        from scipy.linalg import expm

        U = torch.from_numpy(expm(H_cpu.numpy()))
    return U.to(device=H.device, dtype=H.dtype)


def normalize_batch(psi):
    norms = torch.sqrt(torch.sum(torch.abs(psi) ** 2, dim=-1, keepdim=True))
    return psi / norms


def partial_trace(rho_AB, n):
    rho_reshaped = rho_AB.reshape(n, n, n, n)
    return torch.einsum("ijik->jk", rho_reshaped)


def partial_trace_A(rho_AB, n):
    rho_reshaped = rho_AB.reshape(n, n, n, n)
    return torch.einsum("ijil->jl", rho_reshaped)


def partial_trace_batch(psi_AB, n):
    rho_AB = psi_AB.unsqueeze(-1) * psi_AB.unsqueeze(-2).conj()
    return torch.einsum(
        "bijik->bjk", rho_AB.reshape(psi_AB.shape[0], n, n, n, n)
    )


def von_neumann_entropy(rho):
    eigenvalues = torch.linalg.eigvalsh(rho.cpu()).real
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -torch.sum(eigenvalues * torch.log(eigenvalues))


def von_neumann_entropy_batch(rho):
    eigenvalues = torch.linalg.eigvalsh(rho.cpu()).real
    eigenvalues = eigenvalues.clamp(min=0.0)
    mask = eigenvalues > 1e-10
    log_eigs = torch.log(eigenvalues.clamp(min=1e-10))
    entropy = -torch.sum(
        torch.where(mask, eigenvalues * log_eigs, torch.zeros_like(eigenvalues)),
        dim=-1,
    )
    return entropy.to(rho.device)


def volume_law_entropy_bound(n):
    return math.log(n)


def critical_entropy_target(n):
    return 0.5 * volume_law_entropy_bound(n)


def r_critical_target():
    return 0.5 * (config.R_POISSON + config.R_GOE)


def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def balance_fitness(f_order, f_chaos):
    return f_order * f_chaos - abs(f_order - f_chaos)


def order_signal_from_r(r):
    span = config.R_GOE - config.R_POISSON
    return _clamp01(1.0 - abs(r - config.R_POISSON) / span)


def chaos_signal_from_r(r):
    span = config.R_GOE - config.R_POISSON
    return _clamp01(1.0 - abs(r - config.R_GOE) / span)


def order_signal_from_entanglement(s, s_volume):
    if s_volume < 1e-12:
        return 0.0
    return _clamp01(1.0 - s / s_volume)


def chaos_signal_from_entanglement(s, s_volume):
    if s_volume < 1e-12:
        return 0.0
    return _clamp01(s / s_volume)


def mean_level_spacing_ratio(eigenvalues):
    eigs = eigenvalues.flatten().real
    eigs, _ = torch.sort(eigs)
    spacings = eigs[1:] - eigs[:-1]
    rs = []
    for i in range(spacings.shape[0] - 1):
        d_lo = spacings[i].item()
        d_hi = spacings[i + 1].item()
        if d_lo < 1e-12 or d_hi < 1e-12:
            continue
        rs.append(min(d_lo, d_hi) / max(d_lo, d_hi))
    if not rs:
        if spacings.shape[0] == 1 and spacings[0].item() > 1e-12:
            return 1.0
        return 0.0
    return sum(rs) / len(rs)


def _static_order_chaos_signals(H):
    eigs = torch.linalg.eigvalsh(H.cpu()).real
    r_mean = mean_level_spacing_ratio(eigs)
    f_order = order_signal_from_r(r_mean)
    f_chaos = chaos_signal_from_r(r_mean)

    if H.shape[0] > 1:
        _, evecs = torch.linalg.eigh(H.cpu())
        n_local = H.shape[0]
        ipr_sum = 0.0
        for k in range(n_local):
            v = evecs[:, k]
            ipr_sum += torch.sum(torch.abs(v) ** 4).item()
        mean_ipr = ipr_sum / n_local
        f_order = 0.5 * (f_order + _clamp01(mean_ipr * n_local))
        f_chaos = 0.5 * (f_chaos + _clamp01(1.0 - mean_ipr * n_local))

        spread = (eigs[-1] - eigs[0]).item()
        if spread > 1e-12:
            gap = (eigs[1] - eigs[0]).item()
            f_order = (2.0 * f_order + _clamp01(gap / spread)) / 3.0
            f_chaos = (2.0 * f_chaos + _clamp01(1.0 - gap / spread)) / 3.0

    return f_order, f_chaos, r_mean


def static_hamiltonian_fitness(M, n, dtype, device):
    H = hermitian(M)
    f_order, f_chaos, r_mean = _static_order_chaos_signals(H)
    return balance_fitness(f_order, f_chaos), f_order, f_chaos, r_mean


def dynamic_balance_from_entanglement(s, s_volume):
    f_order = order_signal_from_entanglement(s, s_volume)
    f_chaos = chaos_signal_from_entanglement(s, s_volume)
    return balance_fitness(f_order, f_chaos), f_order, f_chaos


def reduced_entropy_from_psi(psi_AB, n, subsystem="B"):
    rho_AB = torch.outer(psi_AB, psi_AB.conj())
    if subsystem == "A":
        rho = partial_trace_A(rho_AB, n)
    elif subsystem == "B":
        rho = partial_trace(rho_AB, n)
    else:
        raise ValueError(f"unknown subsystem: {subsystem}")
    return von_neumann_entropy(rho)


def reduced_entropy_from_psi_batch(psi_AB, n):
    rho = partial_trace_batch(psi_AB, n)
    return von_neumann_entropy_batch(rho)


def apply_evolution_step(M, psi_AB, n, g, dt, device):
    dtype = psi_AB.dtype
    H_total, _ = total_hamiltonian(M, n, g, dtype, device)
    U = matrix_exp_unitary(H_total, dt)
    return normalize(U @ psi_AB)


def apply_evolution_step_batch(M_batch, psi_AB, n, g, dt, device):
    dtype = psi_AB.dtype
    H_total = batch_total_hamiltonian(M_batch, n, g, dtype, device)
    U = matrix_exp_unitary(H_total, dt)
    return normalize_batch(torch.bmm(U, psi_AB.unsqueeze(-1)).squeeze(-1))


def evolution_step(M, psi_AB, n, g, dt, device):
    psi_new = apply_evolution_step(M, psi_AB, n, g, dt, device)
    return 0.0, psi_new


def rollout_dynamic_fitness(M, psi_AB, n, steps, g, dt, device):
    if steps == 0:
        return 0.0, 0.0, 0.0, psi_AB.clone()
    s_volume = volume_law_entropy_bound(n)
    step_sum = 0.0
    state = psi_AB.clone()
    for _ in range(steps):
        state = apply_evolution_step(M, state, n, g, dt, device)
        s_val = reduced_entropy_from_psi(state, n).item()
        step_sum += dynamic_balance_from_entanglement(s_val, s_volume)[0]
    s_final = reduced_entropy_from_psi(state, n).item()
    f_dyn, f_o, f_c = dynamic_balance_from_entanglement(s_final, s_volume)
    return f_dyn + step_sum / steps, f_o, f_c, state


def rollout_fitness(M, psi_AB, n, steps, g, dt, device):
    dtype = psi_AB.dtype
    f_static, f_o_s, f_c_s, r_mean = static_hamiltonian_fitness(M, n, dtype, device)
    f_dynamic, f_o_d, f_c_d, state = rollout_dynamic_fitness(
        M, psi_AB, n, steps, g, dt, device
    )
    total = f_static + config.DYNAMIC_LAMBDA * f_dynamic
    return total, state


def rollout_fitness_batch(M_batch, psi_AB, n, steps, g, dt, device):
    batch_size = M_batch.shape[0]
    if steps == 0:
        return torch.zeros(batch_size, device=device), psi_AB.clone()
    scores = []
    states = []
    for i in range(batch_size):
        score, state = rollout_fitness(M_batch[i], psi_AB[i], n, steps, g, dt, device)
        scores.append(score)
        states.append(state)
    return torch.tensor(scores, device=device), torch.stack(states)


def track_criticality(psi_AB, M, n, dtype, device):
    s_ent = reduced_entropy_from_psi(psi_AB, n).item()
    s_volume = volume_law_entropy_bound(n)
    f_static, f_o_s, f_c_s, r_mean = static_hamiltonian_fitness(M, n, dtype, device)
    f_dyn, f_o_d, f_c_d = dynamic_balance_from_entanglement(s_ent, s_volume)
    return {
        "entanglement": s_ent,
        "s_star": critical_entropy_target(n),
        "s_area_bound": 0.0,
        "s_volume_bound": s_volume,
        "r_mean": r_mean,
        "r_star": r_critical_target(),
        "f_order_static": f_o_s,
        "f_chaos_static": f_c_s,
        "f_order_dynamic": f_o_d,
        "f_chaos_dynamic": f_c_d,
        "f_static": f_static,
        "f_dynamic": f_dyn,
        "balance_gap": abs(f_o_s - f_c_s),
    }


def vacuum_mode_state(n, dtype, device):
    psi = torch.zeros(n, dtype=dtype, device=device)
    psi[0] = 1.0
    return psi


def high_entropy_mode_state(n, dtype, device, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    psi = torch.randn(n, generator=gen, dtype=dtype).to(device)
    return normalize(psi)


def critical_edge_initial_state(n, dtype, device, disorder_seed):
    psi_vacuum = vacuum_mode_state(n, dtype, device)
    psi_disorder = high_entropy_mode_state(n, dtype, device, disorder_seed)
    return torch.kron(psi_vacuum.contiguous(), psi_disorder.contiguous())


def rabi_excitation_product_state(n, dtype, device):
    psi = torch.zeros(n * n, dtype=dtype, device=device)
    psi[n] = 1.0
    return psi


def zero_hamiltonian(n, dtype, device):
    return torch.zeros(n, n, dtype=dtype, device=device)
