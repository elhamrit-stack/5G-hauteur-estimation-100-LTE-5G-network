"""Estimateur d'altitude Z en temps réel via ADB + LightGBM + Filtre de Kalman.

Usage :
    python src/real_time_estimator.py

Prérequis :
    - Téléphone Android en mode débogage USB, connecté au laptop.
    - Réseau forcé en 4G/LTE uniquement (évite les bascules 3G/2G).
    - L'application NetMonster (ou similaire) ouverte au premier plan pour forcer
      le rafraîchissement du modem chaque seconde.
    - Modèle entraîné : models/height_lgbm.txt (généré par src/train_model.py).

Déroulement :
    1. Verrouille la référence sol (t=0) : tenez le téléphone immobile au sol.
    2. Montez sur le site : l'altitude estimée s'affiche chaque seconde dans le terminal.
"""

import os
import sys
import time
import numpy as np
import lightgbm as lgb

# Permet l'import des modules src/ depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adb_collector import poll_cells_adb, adb_available
from feature_engineering import build_features
from kalman_filter import HeightKalmanFilter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lightGBM.txt")
CALIBRATION_SECONDS = 5  # nombre de lectures pour verrouiller le sol


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle introuvable : {MODEL_PATH}")
        print("Générez-le d'abord avec : python src/train_model.py")
        return

    if not adb_available():
        print("ADB non détecté. Connectez le téléphone en USB et activez le débogage USB.")
        print("Test possible sans téléphone : utilisez le mode simulation (voir train_model.py).")
        return

    booster = lgb.Booster(model_file=MODEL_PATH)
    kf = HeightKalmanFilter(ml_rmse=2.0)

    ground_map = None
    calibration_samples = []
    step = 0

    print("=== Altimètre Cellulaire Passif 4G/5G ===")
    print("Tenez le téléphone IMMOBILE au sol (0 m) pendant quelques secondes.")
    print("Appuyez sur Ctrl+C pour arrêter.\n")

    try:
        while True:
            cells = poll_cells_adb()

            if not cells:
                print(f"t={step:03d}s | Aucun signal cellulaire détecté via ADB")
                time.sleep(1.0)
                step += 1
                continue

            # --- Phase de calibration au sol ---
            if ground_map is None:
                calibration_samples.append({c["pci"]: c["rsrp"] for c in cells})
                if len(calibration_samples) >= CALIBRATION_SECONDS:
                    # Moyenne des RSRP au sol par PCI
                    ground_map = {}
                    all_pcis = set()
                    for sample in calibration_samples:
                        all_pcis.update(sample.keys())
                    for pci in all_pcis:
                        vals = [s[pci] for s in calibration_samples if pci in s]
                        ground_map[pci] = float(np.mean(vals))
                    kf.reset(z=0.0)
                    print(f">>> Référence sol verrouillée ({len(ground_map)} cellules). "
                          f"Vous pouvez monter.\n")
                else:
                    print(f"t={step:03d}s | Calibration sol... ({len(calibration_samples)}/{CALIBRATION_SECONDS})")
                    time.sleep(1.0)
                    step += 1
                    continue

            # --- Estimation ---
            feats = build_features(cells, ground_map)
            z_raw = float(booster.predict(feats)[0])
            z_filt, vz = kf.update(z_raw, dt=1.0)

            top_rsrp = max(c["rsrp"] for c in cells)
            num_cells = len(cells)

            print(
                f"t={step:03d}s | Cells:{num_cells:2d} | TopRSRP:{top_rsrp:6.1f}dBm | "
                f"RawML:{z_raw:6.1f}m | FilteredZ:{z_filt:6.1f}m | vz:{vz:+5.2f}m/s"
            )

            step += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nArrêt de l'estimation.")


if __name__ == "__main__":
    main()