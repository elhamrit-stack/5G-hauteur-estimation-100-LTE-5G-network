# Altimètre Cellulaire Passif 4G/5G — Pipeline d'Estimation d'Altitude

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![4G/5G](https://img.shields.io/badge/R%C3%A9seau-4G%20%2F%205G-green.svg)
![LightGBM](https://img.shields.io/badge/Mod%C3%A8le-LightGBM-orange.svg)
![Kalman](https://img.shields.io/badge/Filtre-Kalman-red.svg)
![License](https://img.shields.io/badge/Licence-MIT-lightgrey.svg)
![Platform](https://img.shields.io/badge/Platform-Android%20%2F%20Linux-yellow.svg)

---

## 1. Problème

Estimer l'altitude verticale (axe Z) d'un technicien ou d'un opérateur en hauteur (pylône, bâtiment, éolienne, nacelle) **uniquement à partir d'une connexion 4G ou 5G**, sans GPS, sans baromètre, sans accéléromètre/IMU, sans capteur dédié et sans aucun accord ou API opérateur.

Le système transforme le téléphone ou les lunettes connectées en récepteur radio passif : il écoute la façon dont la puissance des antennes relais (RSRP), leur qualité (RSRQ) et leur distance approximée (Timing Advance) varient quand l'utilisateur monte ou descend, puis en déduit une hauteur en mètres.

### Pourquoi l'altitude est le point faible du positionnement radio

| Contrainte | Explication |
|------------|-------------|
| **Géométrie défavorable (VDOP élevé)** | Les stations de base sont installées à des hauteurs similaires (toits ou pylônes entre 20 m et 50 m). Tous les émetteurs sont presque dans le même plan horizontal, ce qui rend l'estimation de la composante verticale très mal conditionnée. |
| **Pas d'accès aux API opérateur** | Les méthodes 3GPP standard (UL-AoA, Multi-RTT, Carrier Phase) nécessitent un accès au cœur de réseau opérateur (LMF) et des signaux dédiés (SRS/PRS) non exposés aux applications tierces. |
| **Pas de capteur supplémentaire** | Le technicien ne doit porter qu'un smartphone ou des lunettes connectées. Pas de baromètre, pas d'IMU, pas de GPS, pas de LiDAR. |
| **Consommation et calcul limités** | Le matériel embarqué (lunettes connectées, smartphone entrée de gamme) ne peut pas supporter de réseaux de neurones lourds ni de connexion continue vers un serveur Edge. |
| **Multipath et réflexions métalliques** | Sur un pylône ou une structure métallique, les ondes rebondissent, créant des sauts de puissance (RSRP) de plusieurs dBm qui faussent une estimation naive. |

---

## 2. Comment les articles scientifiques résolvent ces limites

Le pipeline s'appuie sur 5 publications scientifiques et 2 normes 3GPP pour transformer ces contraintes en solutions exploitables :

| Limite technique | Solution apportée par la littérature | Référence associée |
|------------------|--------------------------------------|---------------------|
| **Altitude mal conditionnée par géométrie** | Exploiter le **downtilt** et les **lobes secondaires** des antennes : le RSRP varie de manière non monotone (15–20 dB sur quelques mètres) quand le terminal traverse les faisceaux, créant une signature propre à chaque hauteur. | Caballero et al. (2026) — *Drone Height Estimation from 5G Signals* |
| **Non-linéarité du signal (lignes de toit, seuils)** | Les régressions linéaires échouent (RMSE ~72 m). Les **arbres de décision d'ensemble** modélisent naturellement les discontinuités brutales (passage NLoS→LoS au-dessus des toits). | Caballero et al. (2026) ; Posch et al. (2023) |
| **Pas d'accès opérateur / pas de signalisation montante** | Utiliser les signaux cellulaires comme **signaux d'opportunité (SOP)** : lire passivement les canaux de diffusion (SSB/CSI-RS) déjà présents dans le modem, sans aucune API opérateur ni trafic data vers le réseau. | Shamaei et al. (2018) — *Exploiting LTE Signals for Navigation* |
| **Ambiguïté distance horizontale vs élévation** | Ajouter le **Timing Advance (TA)** au vecteur de features pour distinguer un éloignement horizontal d'une élévation verticale. | Maigualema-Quimbita et al. (2025) — *ICNS 2025* |
| **Calcul trop lourd pour l'embarqué** | **LightGBM** (histoogram-based splitting, Exclusive Feature Bundling) : modèle < 300 Ko, inférence < 1 ms sur processeur mobile. | Ke et al. (2017) — *NeurIPS* |
| **Bruit radio / sauts de mesure** | **Filtre de Kalman 1D** (état : altitude + vitesse verticale) pour lisser la prédiction brute et rejeter les variations physiquement impossibles. | Caballero et al. (2026) — Algorithme 1 |

**Normes physiques de référence :**
- **3GPP TR 36.777** (2017) : modèles de canal air-to-ground, probabilité de visibilité directe (LoS) dépendante de l'altitude.
- **3GPP TR 38.901** (2022) : modèles de propagation et diagrammes de rayonnement d'antennes (downtilt, lobes secondaires).

---

## 3. Architecture cible (100 % 5G/4G Passif)

```text
Modem 4G/5G du téléphone
        │ (lecture passive : RSRP, RSRQ, PCI, TA)
        ▼
Programme de collecte (lecture modem / ADB)
        │ (calcul ΔRSRP par rapport au sol t=0)
        ▼
Modèle LightGBM (< 300 Ko, inférence < 1 ms)
        │ (hauteur brute z*)
        ▼
Filtre de Kalman 1D (état : hauteur + vitesse verticale)
        ▼
Altitude filtrée stable en mètres (sortie finale)
```

### Pourquoi cette architecture ?
- **Zéro capteur supplémentaire** : le modem 4G/5G est déjà présent dans tout smartphone.
- **Zéro trafic data vers l'opérateur** : les métriques radio (RSRP, TA, PCI) sont lues localement via les canaux de diffusion du modem (signaux d'opportunité).
- **Faible calcul** : un arbre de décision d'ensemble (LightGBM) + un filtre de Kalman 2×2 tournent en < 2 ms sur un processeur modeste.
- **Commercialisable immédiatement** : fonctionne sur n'importe quel réseau public 4G/5G, sans négociation B2B avec les opérateurs télécoms.

---

## 4. Prérequis

### Matériel minimum
- **1 smartphone Android** (4G ou 5G) — pas de root nécessaire.
- **1 laptop** avec Python 3.10+.
- **ADB (Android Debug Bridge)** installé sur le laptop pour récupérer les données modem en temps réel via USB.

### Logiciel
- Bibliothèques Python : `numpy`, `pandas`, `scikit-learn`, `lightgbm` (voir `requirements.txt`).
- Modèle pré-entraîné : `models/lightGBM.txt` (fourni en externe, entraîné hors ligne).

### Applications sur le téléphone (pour la collecte initiale)
- **NetMonster** ou **G-NetTrack Lite** : loguent RSRP, RSRQ, PCI, TA toutes les secondes.
- **Phyphox** : enregistre l'altitude de référence (vérité terrain) via le baromètre du téléphone.

### Site de test recommandé
- Un point haut d'au moins **60 m** (immeuble de 6 étages, pylône, passerelle, colline).
- Forcer le téléphone en mode **4G / LTE uniquement** pendant les tests pour éviter les bascules 3G/2G.

### Non requis
- Aucun abonnement data particulier.
- Aucun accord opérateur.
- Aucun baromètre, IMU, GPS, LiDAR ou radar dédié.

---

## 5. Dataset de référence public : AERPAW Dataset 22

Pour accélérer l'entraînement sans effectuer immédiatement des relevés terrain, vous pouvez utiliser le dataset public :

**Android-Based 4G LTE Measurements for Semi-Circular UAV Trajectory Around a Private AERPAW Base Station**  
https://aerpaw.org/dataset/dataset-22-android-based-4g-lte-measurements-for-semi-circular-uav-trajectory-around-a-private-aerpaw-base-station/

### Contenu du dataset
- Mesures 4G LTE collectées depuis un téléphone Android (RSRP, RSRQ, PCI, TA, fréquence, etc.).
- Trajectoire semi-circulaire d'un drone/UAV autour d'une station de base privée AERPAW.
- Données de position (altitude et coordonnées GPS) associées aux mesures radio.
- Environnement contrôlé avec station de base privée connue.

### Intégration dans le pipeline
1. **Téléchargement** : Télécharger les fichiers CSV depuis le portail AERPAW.
2. **Placement** : Déposer les fichiers dans `data/raw/` du projet.
3. **Pré-traitement** : Le script `src/train_model.py` lit automatiquement les CSV avec les colonnes `timestamp`, `pci`, `rsrp`, `rsrq`, `ta`, `true_height`.
4. **Entraînement** : Le modèle LightGBM est entraîné avec une validation croisée groupée par trajectoire (GroupKFold) pour éviter le sur-apprentissage sur une seule portion de vol.
5. **Transfert** : Une fois entraîné sur ce dataset, le modèle peut être affiné avec quelques relevés réels sur votre propre site (pylône, bâtiment) pour s'adapter à la géométrie locale des antennes.

### Pourquoi ce dataset est adapté
- Il contient des variations d'altitude réelles (pas seulement simulées).
- Il couvre une station de base privée avec antenne directionnelle (downtilt), ce qui reproduit le phénomène physique exploité par le modèle (lobes secondaires, transition NLoS→LoS).
- Il permet de valider l'approche 100 % 4G passive sans posséder de station de base 5G privée.

---

## 6. Structure du dépôt

```
passive-cellular-altimeter/
├── data/
│   ├── raw/            # CSV bruts (NetMonster, Phyphox, AERPAW)
│   └── processed/      # Logs alignés avec ΔRSRP et hauteur vérité terrain
├── models/
│   └── lightGBM.txt    # Modèle LightGBM pré-entraîné (< 300 Ko, fourni en externe)
├── paper/              # Références scientifiques (PDF)
├── src/
│   ├── adb_collector.py        # Lecture cellules 4G/5G via ADB
│   ├── feature_engineering.py  # Features ΔRSRP, top-K, stats, TA
│   ├── kalman_filter.py        # Filtre de Kalman 2 états [z, vz]
│   ├── train_model.py          # Entraînement LightGBM (optionnel, externe)
│   └── real_time_estimator.py  # Boucle temps réel ADB → LightGBM → Kalman → Z
├── tests/
│   └── test_replay.py          # Relecture et comparaison vrai vs estimé
├── collect_data.sh             # Script d'aide export CSV via ADB
├── requirements.txt
└── README.md
```

---

## 7. Workflow d'implémentation rapide (3 jours)

### Jour 1 — Collecte terrain (zéro code personnalisé)
1. Installer **NetMonster** et **Phyphox** sur le téléphone Android.
2. Forcer le réseau en **4G uniquement**.
3. Se tenir au sol (0 m) pendant 15 s pour verrouiller la référence.
4. Monter à pied ou en ascenseur sur le site test (jusqu'à 60 m), marquer une pause de 10 s à chaque palier, puis descendre.
5. Répéter sur 2 ou 3 bâtiments différents.
6. Exporter les CSV de NetMonster (radio) et Phyphox (hauteur vérité terrain).

### Jour 2 — Entraînement et validation (laptop, terminal Python)
1. Aligner les CSV radio et hauteur par timestamp.
2. Calculer les features invariantes au site : `ΔRSRP = RSRP(t) − RSRP(t=0)`.
3. Entraîner un `LGBMRegressor` avec une validation croisée **groupée par bâtiment**.
4. Sauvegarder le booster LightGBM (`lightGBM.txt`, < 300 Ko).
5. Tester sur un bâtiment jamais vu lors de l'entraînement ; viser un RMSE filtré < 2,0 m.

### Jour 3 — Test en direct (laptop terminal + ADB)
1. Connecter le téléphone au laptop via ADB (`adb devices`).
2. Lancer NetMonster en premier plan sur le téléphone pour forcer le rafraîchissement modem chaque seconde.
3. Exécuter le script principal sur le laptop : lecture ADB `dumpsys telephony.registry`, extraction des cellules 4G, inférence LightGBM, filtrage Kalman, affichage de l'altitude en temps réel dans le terminal.
4. Monter sur le site et vérifier que l'altitude affichée correspond à la hauteur réelle à chaque palier.

---

## 8. Entrées et sorties du système

### Entrées (lues chaque seconde)
- **RSRP** (dBm) : puissance reçue de la cellule serveuse et des cellules voisines détectables.
- **RSRQ** : qualité du signal (indique NLoS vs LoS).
- **PCI** : identifiant physique de chaque cellule/antenne.
- **Timing Advance (TA)** : distance oblique approximée vers l'antenne serveuse.
- **Référence au sol (t=0)** : vecteur RSRP mémorisé quand le technicien est immobile au niveau 0 m.
- **ΔRSRP** : différence relative `RSRP(t) − RSRP(t=0)` pour rendre le modèle indépendant du site.

### Sorties (produites chaque seconde)
- **Hauteur brute ML** (`z*`) : estimation instantanée en mètres par LightGBM.
- **Hauteur filtrée** (`z`) : valeur lissée par le filtre de Kalman (m).
- **Vitesse verticale** (`vz`) : m/s, utile pour détecter les paliers d'immobilisation.

---

## 9. Précision et taux d'erreur attendus

| Phase d'activité | Erreur typique (RMSE) | 95 % des cas |
|---|---|---|
| Montée / descente (dynamique) | 1,5 m – 3 m | < 3 m |
| Palier / arrêt (statique) | 0,5 m – 1,5 m | < 2 m |
| Zone critique (15 m – 30 m, niveau toits) | jusqu'à 3 m – 4 m temporairement | stabilise après quelques secondes |

**Note** : sans baromètre ni IMU, la précision est métrique (non centimétrique). Elle suffit pour identifier un étage, un palier de travail ou un seuil de sécurité sur un pylône.

---

## 10. Limites et compromis à connaître

- **4G uniquement** : le Timing Advance 4G a une étape de ~78 m, donc peu utile sur 0–60 m ; le modèle s'appuiera principalement sur ΔRSRP et RSRQ.
- **Zone critique 15 m – 30 m** : bascule brutale entre signal masqué par les immeubles et visibilité directe avec les antennes lointaines ; erreur temporaire possible jusqu'à 3–4 m.
- **Dépendance à la densité d'antennes** : en zone très rurale avec une seule antenne visible, la précision se dégrade.
- **Pas de précision centimétrique** : sans baromètre/IMU ni accès au cœur réseau opérateur (UL-AoA, Multi-RTT), l'erreur reste métrique.
- **Généralisation** : un modèle entraîné sur un quartier fonctionne moins bien sur un autre site sans ré-entraînement partiel ou features normalisées par ΔRSRP.

---

## 11. Lancer le prototype

```bash
# Cloner le dépôt
git clone https://github.com/elhamrit-stack/5G-hauteur-estimation-100-LTE-5G-network.git
cd 5G-hauteur-estimation-100-LTE-5G-network

# Installer les dépendances
pip install -r requirements.txt

# Vérifier que le modèle pré-entraîné est présent
ls models/lightGBM.txt

# Connecter le téléphone Android en USB (débogage USB activé, 4G forcé)
adb devices

# Lancer l'estimation d'altitude Z en temps réel
python src/real_time_estimator.py
```

Déroulement :
1. Tenez le téléphone **immobile au sol** (0 m) pendant ~5 secondes pour verrouiller la référence RSRP.
2. Montez sur le pylône / bâtiment / nacelle.
3. L'altitude estimée s'affiche chaque seconde dans le terminal.
