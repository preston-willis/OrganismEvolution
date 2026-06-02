import matplotlib.pyplot as plt
import numpy as np

from config import R_POISSON, R_WIGNER
from fitness import spacing_ratios
from hamiltonian import build_hamiltonian


def analyze_hamiltonian(genome, n_qubits, show=True, save_path=None):
    H = build_hamiltonian(genome, n_qubits)
    eigenvalues = np.linalg.eigvalsh(H)
    ratios = spacing_ratios(H)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(ratios, bins=20, density=True)
    axes[0].axvline(R_POISSON, color="b", label="Poisson")
    axes[0].axvline(R_WIGNER, color="r", label="Wigner-Dyson")
    axes[0].set_title("Level spacing ratio distribution")
    axes[0].legend()

    axes[1].plot(sorted(eigenvalues), "o-")
    axes[1].set_title("Energy spectrum")

    terms = sorted(genome, key=lambda x: abs(x[0]), reverse=True)
    labels = [t[1] for t in terms[:10]]
    values = [abs(t[0]) for t in terms[:10]]
    axes[2].barh(labels, values)
    axes[2].set_title("Dominant interaction terms")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("\nEvolved Hamiltonian terms:")
    for coeff, pstring in terms:
        print(f"  {coeff:+.4f} × {pstring}")

    return H, eigenvalues, ratios
