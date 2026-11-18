"""Feature engineering : transforme les mesures cellulaires brutes en vecteur
invariante au site (Delta-RSRP par rapport au sol t=0) pour le modèle LightGBM."""

import numpy as np
from typing import List, Dict, Optional

TOP_K = 8


def build_features(
    cells: List[Dict],
    ground_map: Optional[Dict[int, float]],
    top_k: int = TOP_K
) -> np.ndarray:
    """
    Construit un vecteur de features à partir des cellules détectées.

    Paramètres
    ----------
    cells : liste de dict {pci, rsrp, rsrq, ta, type}
    ground_map : dictionnaire {pci: rsrp_sol} mémorisé à t=0 (niveau 0 m)
    top_k : nombre de cellules les plus puissantes à conserver

    Retour
    ------
    np.ndarray shape (1, 2*top_k + 5)
    """
    if not cells:
        return np.zeros(2 * top_k + 5, dtype=np.float32).reshape(1, -1)

    if ground_map is None:
        ground_map = {}

    rsrps = [c["rsrp"] for c in cells]
    pcis = [c["pci"] for c in cells]

    # Delta-RSRP par rapport au sol (rend le modèle indépendant du site)
    deltas = [
        rsrp - ground_map.get(pci, -132.0)
        for pci, rsrp in zip(pcis, rsrps)
    ]

    # Top-K des deltas et des RSRP bruts (padding pour vecteur de taille fixe)
    deltas_sorted = sorted(deltas, reverse=True)[:top_k]
    deltas_padded = deltas_sorted + [0.0] * (top_k - len(deltas_sorted))

    rsrps_sorted = sorted(rsrps, reverse=True)[:top_k]
    rsrps_padded = rsrps_sorted + [-140.0] * (top_k - len(rsrps_sorted))

    # Statistiques globales
    mean_rsrp = float(np.mean(rsrps))
    std_rsrp = float(np.std(rsrps)) if len(rsrps) > 1 else 0.0
    num_cells = float(len(cells))
    mean_rsrq = float(np.mean([c.get("rsrq", -11.0) for c in cells]))

    # Timing Advance de la cellule la plus forte (serving)
    tas = [c.get("ta", -1) for c in cells if c.get("ta", -1) >= 0]
    serving_ta = float(tas[0]) if tas else -1.0

    features = deltas_padded + rsrps_padded + [
        mean_rsrp,
        std_rsrp,
        num_cells,
        mean_rsrq,
        serving_ta
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)