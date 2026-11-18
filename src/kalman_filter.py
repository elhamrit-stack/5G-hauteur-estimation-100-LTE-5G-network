"""Filtre de Kalman 1D à 2 états : [altitude, vitesse verticale].

Lisse la prédiction brute du modèle LightGBM et rejette les sauts dus au
multipath / réflexions métalliques.
"""

import numpy as np


class HeightKalmanFilter:
    def __init__(self, ml_rmse: float = 2.0):
        # État : [hauteur (m), vitesse verticale (m/s)]
        self.x = np.array([[0.0], [0.0]])
        self.P = np.array([[0.5, 0.0], [0.0, 0.5]])  # covariance initiale
        self.H = np.array([[1.0, 0.0]])               # on observe seulement la hauteur
        self.R = np.array([[ml_rmse ** 2]])           # bruit de mesure (calibré sur RMSE ML)
        self.Q = np.array([[0.05, 0.0], [0.0, 0.08]]) # bruit de mouvement humain

    def reset(self, z: float = 0.0):
        self.x = np.array([[z], [0.0]])
        self.P = np.array([[0.5, 0.0], [0.0, 0.5]])

    def update(self, z_meas: float, dt: float = 1.0):
        """Prédiction + correction. Retourne (hauteur_filtrée, vitesse_verticale)."""
        # --- Prédiction (modèle cinématique à vitesse constante) ---
        A = np.array([[1.0, dt], [0.0, 1.0]])
        x_pred = A @ self.x
        P_pred = A @ self.P @ A.T + self.Q

        # --- Correction (mesure LightGBM) ---
        y = np.array([[z_meas]]) - (self.H @ x_pred)           # innovation
        S = self.H @ P_pred @ self.H.T + self.R                # covariance innovation
        K = P_pred @ self.H.T @ np.linalg.inv(S)               # gain de Kalman

        self.x = x_pred + (K @ y)
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        z_filt = max(0.0, float(self.x[0, 0]))
        v_z = float(self.x[1, 0])
        return z_filt, v_z