# Reference Pauli-string Hamiltonians for novelty checks (4-qubit examples).

KNOWN_CRITICAL = {
    "Transverse Ising (chain)": [
        (1.0, "ZZII"),
        (1.0, "IZZI"),
        (1.0, "IIZZ"),
        (1.0, "XIII"),
    ],
    "XXZ (nearest-neighbor)": [
        (1.0, "XXII"),
        (1.0, "IXXI"),
        (1.0, "IIXX"),
        (1.0, "ZZII"),
        (1.0, "IZZI"),
        (1.0, "IIZZ"),
    ],
}


def dominant_strings(genome, top_k=10):
    terms = sorted(genome, key=lambda x: abs(x[0]), reverse=True)
    return {pstring for _, pstring in terms[:top_k]}


def matches_known(genome, top_k=10):
    found = dominant_strings(genome, top_k)
    hits = []
    for name, terms in KNOWN_CRITICAL.items():
        ref = {p for _, p in terms}
        overlap = len(found & ref) / max(len(ref), 1)
        if overlap >= 0.5:
            hits.append((name, overlap))
    return hits
