import torch


def normalize(psi):
    return psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2))


def hermitian(M):
    return (M + M.mH) / 2


def ladder_operators(n, dtype, device):
    i = torch.arange(1, n, device=device).float()
    a = torch.diag(torch.sqrt(i), diagonal=-1).to(dtype=dtype, device=device)
    ad = a.conj().T
    return a, ad


def coupling_hamiltonian(n, g, dtype, device):
    a, ad = ladder_operators(n, dtype, device)
    a = a.contiguous()
    ad = ad.contiguous()
    return g * (torch.kron(ad, a) + torch.kron(a, ad))


def total_hamiltonian(M_A, M_B, n, g, dtype, device):
    H_A = hermitian(M_A)
    H_B = hermitian(M_B)
    I = torch.eye(n, dtype=dtype, device=device)
    H_int = coupling_hamiltonian(n, g, dtype, device)
    H_A = H_A.contiguous()
    H_B = H_B.contiguous()
    return torch.kron(H_A, I) + torch.kron(I, H_B) + H_int, H_A


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


def batch_total_hamiltonian(M_A, M_B, n, g, dtype, device):
    H_A = hermitian(M_A).contiguous()
    H_B = hermitian(M_B).contiguous()
    I = torch.eye(n, dtype=dtype, device=device)
    H_int = coupling_hamiltonian(n, g, dtype, device)
    batch_size = M_A.shape[0]
    return torch.stack(
        [
            torch.kron(H_A[i], I) + torch.kron(I, H_B[i]) + H_int
            for i in range(batch_size)
        ]
    )


def partial_trace_batch(psi_AB, n):
    rho_AB = psi_AB.unsqueeze(-1) * psi_AB.unsqueeze(-2).conj()
    return torch.einsum(
        "bijik->bjk", rho_AB.reshape(psi_AB.shape[0], n, n, n, n)
    )


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


def partial_trace(rho_AB, n):
    """Trace out subsystem A; returns reduced state on B."""
    rho_reshaped = rho_AB.reshape(n, n, n, n)
    return torch.einsum("ijik->jk", rho_reshaped)


def partial_trace_A(rho_AB, n):
    """Trace out subsystem B; returns reduced state on A."""
    rho_reshaped = rho_AB.reshape(n, n, n, n)
    return torch.einsum("ijil->jl", rho_reshaped)


def von_neumann_entropy(rho):
    eigenvalues = torch.linalg.eigvalsh(rho.cpu()).real
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -torch.sum(eigenvalues * torch.log(eigenvalues))


def log_n(n, device):
    return torch.log(torch.tensor(float(n), device=device))


def parabolic_entropy(S, n):
    log_n_val = log_n(n, S.device)
    return S * (log_n_val - S)


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


def evolution_step(M_A, M_B, psi_AB, n, g, dt, device):
    dtype = psi_AB.dtype
    H_total, _ = total_hamiltonian(M_A, M_B, n, g, dtype, device)
    U = matrix_exp_unitary(H_total, dt)
    psi_new = normalize(U @ psi_AB)
    S_after = reduced_entropy_from_psi(psi_new, n)
    return parabolic_entropy(S_after, n).item(), psi_new


def evolution_step_batch(M_A, M_B, psi_AB, n, g, dt, device):
    dtype = psi_AB.dtype
    H_total = batch_total_hamiltonian(M_A, M_B, n, g, dtype, device)
    U = matrix_exp_unitary(H_total, dt)
    psi_new = normalize_batch(torch.bmm(U, psi_AB.unsqueeze(-1)).squeeze(-1))
    S_after = reduced_entropy_from_psi_batch(psi_new, n)
    return parabolic_entropy(S_after, n), psi_new


def rollout_fitness(M_A, M_B, psi_AB, n, steps, g, dt, device):
    if steps == 0:
        return 0.0, psi_AB.clone()
    step_sum = 0.0
    state = psi_AB.clone()
    for _ in range(steps):
        step_score, state = evolution_step(M_A, M_B, state, n, g, dt, device)
        step_sum += step_score
    S_final = reduced_entropy_from_psi(state, n)
    return parabolic_entropy(S_final, n).item() + step_sum / steps, state


def rollout_fitness_batch(M_A, M_B, psi_AB, n, steps, g, dt, device):
    batch_size = psi_AB.shape[0]
    if steps == 0:
        return torch.zeros(batch_size, device=device), psi_AB.clone()
    step_sum = torch.zeros(batch_size, device=device)
    psi = psi_AB
    for _ in range(steps):
        step_score, psi = evolution_step_batch(M_A, M_B, psi, n, g, dt, device)
        step_sum = step_sum + step_score
    S_final = reduced_entropy_from_psi_batch(psi, n)
    return parabolic_entropy(S_final, n) + step_sum / steps, psi


def track_complexity(psi_AB, M_A, n):
    H_A = hermitian(M_A)
    rho_AB = torch.outer(psi_AB, psi_AB.conj())
    rho_A = partial_trace(rho_AB, n)

    S_ent = von_neumann_entropy(rho_A)

    probs = torch.abs(psi_AB) ** 2
    PR = 1.0 / torch.sum(probs**2)

    H_offdiag = H_A - torch.diag(torch.diag(H_A))
    structure = torch.sqrt(torch.sum(torch.abs(H_offdiag) ** 2)) / torch.sqrt(
        torch.sum(torch.abs(H_A) ** 2)
    )

    eigs = torch.linalg.eigvalsh(rho_A.cpu()).real
    spread = torch.std(eigs)

    return {
        "entanglement": S_ent.item(),
        "participation_ratio": PR.item(),
        "hamiltonian_structure": structure.item(),
        "eigenvalue_spread": spread.item(),
    }


def vacuum_product_state(n, dtype, device):
    psi = torch.zeros(n * n, dtype=dtype, device=device)
    psi[0] = 1.0
    return psi


def random_product_state(n, dtype, device):
    psi_A = torch.randn(n, dtype=dtype, device=device)
    psi_A = normalize(psi_A)
    psi_B = torch.randn(n, dtype=dtype, device=device)
    psi_B = normalize(psi_B)
    return torch.kron(psi_A.contiguous(), psi_B.contiguous())


def initial_product_state(n, dtype, device, mode):
    if mode == "vacuum":
        return vacuum_product_state(n, dtype, device)
    if mode == "random":
        return random_product_state(n, dtype, device)
    raise ValueError(f"unknown initial state mode: {mode}")
