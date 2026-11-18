"""Entraînement du modèle LightGBM pour l'estimation d'altitude Z depuis le RSRP 4G/5G.

Deux modes :
1. Données réelles : CSV dans data/raw/ (colonnes : timestamp, pci, rsrp, rsrq, ta, true_height)
2. Bootstrap synthétique : génère un modèle préalable basé sur la physique 3GPP
   (downtilt, lobes secondaires, transition NLoS→LoS) pour tester immédiatement.
"""

import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import joblib

from feature_engineering import build_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "height_lgbm.txt")
META_PATH = os.path.join(MODEL_DIR, "metadata.pkl")

TOP_K = 8


def _synthetic_climb(rng, n_towers):
    """Génère une montée synthétique 0-60 m avec physique d'antenne simplifiée."""
    pcis = rng.choice(range(10, 500), size=n_towers, replace=False)
    tower_dists = rng.uniform(120, 900, size=n_towers)
    tower_heights = rng.uniform(22, 38, size=n_towers)
    downtilts = np.deg2rad(rng.uniform(10, 20, size=n_towers))

    def simulate_scan(z: float):
        scan = []
        for pci, d_h, z_bs, tilt in zip(pcis, tower_dists, tower_heights, downtilts):
            d_3d = np.hypot(d_h, z - z_bs)
            elev = np.arctan2(z_bs - z, d_h)
            off = elev - tilt
            gain = -min(12.0 * (off / np.deg2rad(15.0)) ** 2, 22.0)
            if z > z_bs:
                gain += 4.0 * np.sin(8.0 * off)
            rooftop = 9.0 / (1.0 + np.exp(-(z - 22.0) / 2.5))
            rsrp = -32.0 - 24.0 * np.log10(d_3d) + gain + rooftop + rng.normal(0, 1.8)
            if rsrp > -125.0:
                rsrq = -8.0 - 0.06 * z + rng.normal(0, 1.2)
                ta = int(d_3d // 78.0)
                scan.append({"pci": int(pci), "rsrp": float(rsrp), "rsrq": float(rsrq), "ta": ta})
        return scan

    return pcis, tower_dists, tower_heights, downtilts, simulate_scan


def train_bootstrap() -> lgb.Booster:
    """Entraîne un modèle sur données synthétiques 3GPP (0 à 60 m)."""
    rng = np.random.default_rng(42)
    X_train, y_train = [], []

    for _ in range(200):
        n_towers = rng.integers(5, 11)
        pcis, tower_dists, tower_heights, downtilts, simulate_scan = _synthetic_climb(rng, n_towers)
        ground_scan = simulate_scan(0.0)
        if not ground_scan:
            continue
        ground_map = {c["pci"]: c["rsrp"] for c in ground_scan}

        for z in np.linspace(0.0, 60.0, 31):
            scan = simulate_scan(z)
            if scan:
                feats = build_features(scan, ground_map, top_k=TOP_K)
                X_train.append(feats[0])
                y_train.append(z)

    model = lgb.LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        random_state=42, verbose=-1
    )
    model.fit(np.array(X_train), np.array(y_train))
    return model.booster_


def train_real(csv_files):
    """Entraîne sur des CSV réels avec validation croisée groupée par fichier (site)."""
    all_X, all_y, all_groups = [], [], []

    for filepath in csv_files:
        df = pd.read_csv(filepath)
        required = {"timestamp", "pci", "rsrp", "true_height"}
        if not required.issubset(df.columns):
            print(f"Skipping {filepath}: missing columns {required - df.columns}")
            continue

        # Vérifier s'il y a une colonne ground_rsrp ou calculer depuis true_height<=1
        ground_df = df[df["true_height"] <= 1.0]
        ground_map = {}
        if "rsrp" in df.columns and not ground_df.empty:
            ground_map = ground_df.groupby("pci")["rsrp"].mean().to_dict()

        climb_id = os.path.basename(filepath).replace(".csv", "")
        X_c, y_c = [], []

        for ts, group in df.groupby("timestamp"):
            cells = group.to_dict("records")
            feats = build_features(cells, ground_map, top_k=TOP_K)
            X_c.append(feats[0])
            y_c.append(float(group["true_height"].iloc[0]))

        if X_c:
            all_X.append(np.array(X_c))
            all_y.append(np.array(y_c))
            all_groups.extend([climb_id] * len(y_c))

    if not all_X:
        raise ValueError("Aucune donnée utilisable dans les CSV fournis.")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)

    gkf = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    oof = np.zeros_like(y)
    params = {
        "objective": "regression", "metric": "rmse",
        "boosting_type": "gbdt", "num_leaves": 31,
        "learning_rate": 0.05, "n_estimators": 250,
        "feature_fraction": 0.8, "verbose": -1, "random_state": 42
    }

    for train_idx, val_idx in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[train_idx], y[train_idx])
        oof[val_idx] = m.predict(X[val_idx])

    rmse = root_mean_squared_error(y, oof)
    print(f"Cross-site raw ML RMSE: {rmse:.2f} m")

    final = lgb.LGBMRegressor(**params)
    final.fit(X, y)
    return final.booster_, rmse


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if csv_files:
        print(f"Entraînement sur {len(csv_files)} fichier(s) CSV réel(s)...")
        booster, rmse = train_real(csv_files)
    else:
        print("Aucun CSV dans data/raw/. Entraînement du modèle bootstrap synthétique...")
        booster = train_bootstrap()
        rmse = None

    booster.save_model(MODEL_PATH)
    joblib.dump({"rmse": rmse, "top_k": TOP_K}, META_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")
    if rmse is not None:
        print(f"RMSE de validation : {rmse:.2f} m")


if __name__ == "__main__":
    main()