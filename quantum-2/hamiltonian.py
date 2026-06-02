import numpy as np

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
PAULIS = {"I": I, "X": X, "Y": Y, "Z": Z}


def build_hamiltonian(genome, n_qubits):
    """
    genome: list of (coefficient, pauli_string) tuples
    e.g. [(0.5, 'XZIY'), (-0.3, 'ZZII')]
    """
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for coeff, pstring in genome:
        term = np.array([[1.0]])
        for p in pstring:
            term = np.kron(term, PAULIS[p])
        H += coeff * term
    return H


def sum_pauli_x(n_qubits):
    """Drive operator: uniform transverse field sum_i X_i."""
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for q in range(n_qubits):
        term = np.array([[1.0]])
        for k in range(n_qubits):
            term = np.kron(term, PAULIS["X" if k == q else "I"])
        H += term
    return H
