# Altimètre Cellulaire Passif 4G/5G — Pipeline d'Estimation d'Altitude

## Objectif

Estimer l'altitude verticale (axe Z) d'un technicien ou d'un opérateur en hauteur (ex. pylône, bâtiment, éolienne, nacelle) **uniquement à partir d'une connexion 4G ou 5G**, sans GPS, sans baromètre, sans accéléromètre/IMU, sans capteur dédié et sans aucun accord ou API opérateur.

Le système transforme le téléphone ou les lunettes connectées en récepteur radio passif : il écoute la façon dont la puissance des antennes relais (RSRP), leur qualité (RSRQ) et leur distance approximée (Timing Advance) varient quand l'utilisateur monte ou descend, puis en déduit une hauteur en mètres.

---

## Architecture cible (100 % 5G/4G Passif)

```
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

## Fondement scientifique (5 articles de référence)

Chaque brique du pipeline est validée par une publication ou une norme industrielle :

| Brique du pipeline | Référence scientifique | Apport direct |
|---|---|---|
| **Cœur : RSRP + Arbre d'ensemble + Kalman** | Caballero, E., Fakhreddine, A., & Bettstetter, C. (2026) — *Drone Height Estimation from 5G Signals*, University of Klagenfurt. | Démonstre que les vecteurs RSRP multi-cellules traités par un arbre de décision d'ensemble puis lissé par un filtre de Kalman atteignent un RMSE de 1,0 à 1,6 m en exploitant le downtilt et les lobes secondaires des antennes 5G. |
| **Classification par tranches de hauteur** | Posch, F., Fakhreddine, A., Caballero, E., & Bettstetter, C. (2023) — *A Classifier for Aerial Devices in 5G Networks*, IEEE GLOBECOM Workshops. | Méthodologie d'extraction des signatures RSRP pour distinguer utilisateurs au sol et en hauteur via des arbres de décision. |
| **Moteur d'inférence embarqué ultra-léger** | Ke, G., Meng, Q., Finley, T., et al. (2017) — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS. | Réduit la taille mémoire à < 300 Ko et le temps d'inférence à < 1 ms sur mobile/laptop, tout en conservant la précision des arbres d'ensemble. |
| **Écoute 100 % passive (SOP)** | Shamaei, K., Khalife, J., & Kassas, Z. M. (2018) — *Exploiting LTE Signals for Navigation: Theory to Implementation*, IEEE Trans. on Wireless Communications. | Prouve qu'un terminal peut exploiter les signaux cellulaires descendants sans aucune signalisation montante ni API opérateur. |
| **Ajout du Timing Advance (TA) + ML** | Maigualema-Quimbita, D. A., et al. (2025) — *Positioning of UAVs in Urban Environments Using 5G Networks*, IEEE ICNS. | Valide l'ajout du TA/RTT pour lever l'ambiguïté entre éloignement horizontal et élévation verticale. |

**Normes de référence physiques :**
- **3GPP TR 36.777** (2017) : modèles de canal air-to-ground, probabilité de visibilité directe (LoS) dépendante de l'altitude.
- **3GPP TR 38.901** (2022) : modèles de propagation et diagrammes de rayonnement d'antennes (downtilt, lobes secondaires).

---

## Principe physique exploité

Les antennes relais 4G/5G sont inclinées vers le sol (**downtilt** de 10° à 20°) pour couvrir les utilisateurs terrestres. Lorsqu'un technicien prend de l'altitude, trois phénomènes mesurables se produisent :

1. **Sortie du faisceau principal** : le terminal quitte le lobe principal et traverse des creux d'atténuation puis des lobes secondaires. La puissance RSRP varie de manière non monotone (sauts de 15 à 20 dB sur quelques mètres).
2. **Transition NLoS → LoS au-dessus des toits** : dès que le terminal dépasse ~20–25 m, il capte soudainement de nombreuses antennes lointaines auparavant masquées par les immeubles.
3. **Timing Advance (TA)** : l'opérateur mesure le temps aller-retour pour synchroniser le terminal, donnant une distance oblique approximée vers l'antenne serveuse.

Ces signatures radio, uniques à chaque hauteur, sont apprises par le modèle LightGBM.

---

## Entrées et sorties du système

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

## Pipeline d'exécution (4 étages)

1. **Ancrage zéro automatique (t=0)** : détection d'immobilité au sol, mémorisation du vecteur radio de référence `RSRP₀`.
2. **Pré-traitement** : calcul des ΔRSRP, top-K des cellules les plus puissantes, statistiques (moyenne, écart-type, nombre de cellules visibles), et TA.
3. **Inférence ML** : passage du vecteur de features dans le modèle LightGBM pour obtenir `z*` en < 1 ms.
4. **Filtrage Kalman** : modèle cinématique `[hauteur, vitesse verticale]` pour rejeter les sauts dus aux réflexions métalliques (multipath) et livrer `z` stable.

---

## Précision et taux d'erreur attendus

| Phase d'activité | Erreur typique (RMSE) | 95 % des cas |
|---|---|---|
| Montée / descente (dynamique) | 1,5 m – 3 m | < 3 m |
| Palier / arrêt (statique) | 0,5 m – 1,5 m | < 2 m |
| Zone critique (15 m – 30 m, niveau toits) | jusqu'à 3 m – 4 m temporairement | stabilise après quelques secondes |

**Note** : sans baromètre ni IMU, la précision est métrique (non centimétrique). Elle suffit pour identifier un étage, un palier de travail ou un seuil de sécurité sur un pylône.


---

## Matériel et prérequis

### Minimum indispensable
- **1 smartphone Android** (4G ou 5G) — pas de root nécessaire.
- **1 laptop** avec Python 3.10+ et les bibliothèques : `numpy`, `pandas`, `scikit-learn`, `lightgbm`.
- **ADB (Android Debug Bridge)** installé sur le laptop pour récupérer les données modem en temps réel via USB (ou Wi-Fi adb).
- **Applications logger** sur le téléphone pour la collecte initiale : `NetMonster` et `Phyphox` (Play Store).

### Recommandé pour les tests terrain
- Un site avec point haut d'au moins **60 m** (immeuble de 6 étages, pylône, passerelle, colline).
- Forcer le téléphone en mode **4G / LTE uniquement** pendant les tests pour éviter les bascules 3G/2G qui changent les PCI.

### Non requis
- Aucun abonnement data particulier.
- Aucun accord opérateur.
- Aucun baromètre, IMU, GPS, LiDAR ou radar.

---

## Structure suggérée du dépôt

```
passive-cellular-altimeter/
├── data/
│   ├── raw/            # CSV bruts (NetMonster + Phyphox)
│   └── processed/      # Logs alignés avec ΔRSRP et hauteur vérité terrain
├── src/
│   ├── align_logs.py   # Fusion radio + hauteur de référence par timestamp
│   ├── features.py     # Calcul ΔRSRP, top-K, stats, TA
│   ├── train.py        # Entraînement LightGBM (validation GroupKFold par site)
│   ├── kalman.py       # Filtre de Kalman 2 états [z, vz]
│   └── edge_server.py  # API légère pour test en direct (optionnel)
├── models/
│   ├── lightGBM.txt # Modèle LightGBM pré-entraîné (< 300 Ko, fourni en externe)
│   └── metadata.json   # RMSE de validation, features, date d'entraînement
├── tests/
│   └── test_replay.py  # Relecture d'une montée enregistrée et comparaison vrai vs estimé
├── docs/
│   └── protocol_test.md# Procédure de montée de validation (0 m → 60 m)
└── README.md
```

---

## Dataset de référence public : AERPAW Dataset 22

Pour accélérer l'entraînement sans effectuer immédiatement des relevés terrain, vous pouvez utiliser le dataset public :

**Android-Based 4G LTE Measurements for Semi-Circular UAV Trajectory Around a Private AERPAW Base Station**  
https://aerpaw.org/dataset/dataset-22-android-based-4g-lte-measurements-for-semi-circular-uav-trajectory-around-a-private-aerpaw-base-station/

### Contenu du dataset
- Mesures 4G LTE collectées depuis un téléphone Android (RSRP, RSRQ, PCI, TA, fréquence, etc.)
- Trajectoire semi-circulaire d'un drone/UAV autour d'une station de base privée AERPAW
- Données de position (altitude et coordonnées GPS) associées aux mesures radio
- Environnement contrôlé avec station de base privée connue

### Intégration dans le pipeline

1. **Téléchargement** : Télécharger les fichiers CSV depuis le portail AERPAW.
2. **Placement** : Déposer les fichiers dans `data/raw/` du projet.
3. **Pré-traitement** : Le script `src/train_model.py` (fonction `train_real`) lit automatiquement les CSV avec les colonnes `timestamp`, `pci`, `rsrp`, `rsrq`, `ta`, `true_height`.
4. **Entraînement** : Le modèle LightGBM est entraîné avec une validation croisée groupée par trajectoire (GroupKFold) pour éviter le sur-apprentissage sur une seule portion de vol.
5. **Transfert** : Une fois entraîné sur ce dataset, le modèle peut être affiné avec quelques relevés réels sur votre propre site (pylône, bâtiment) pour s'adapter à la géométrie locale des antennes.

### Pourquoi ce dataset est adapté
- Il contient des variations d'altitude réelles (pas seulement simulées).
- Il couvre une station de base privée avec antenne directionnelle (downtilt), ce qui reproduit le phénomène physique exploité par le modèle (lobes secondaires, transition NLoS→LoS).
- Il permet de valider l'approche 100 % 4G passive sans posséder de station de base 5G privée.


---

## Workflow d'implémentation rapide (3 jours)

### Jour 1 — Collecte terrain (zéro code personnalisé)
1. Installer **NetMonster** et **Phyphox** sur le téléphone Android.
2. Forcer le réseau en **4G uniquement**.
3. Se tenir au sol (0 m) pendant 15 s pour verrouiller la référence.
4. Monter à pied ou en ascenseur sur le site test (jusqu'à 60 m), marquer une pause de 10 s à chaque palier, puis descendre.
5. Répéter sur 2 ou 3 bâtiments différents.
6. Exporter les CSV de NetMonster (radio) et Phyphox (hauteur vérité terrain).

### Jour 2 — Entraînement et validation (laptop, terminal Python)
1. Aligner les CSV radio et hauteur par timestamp (`align_logs.py`).
2. Calculer les features invariantes au site : `ΔRSRP = RSRP(t) − RSRP(t=0)`.
3. Entraîner un `LGBMRegressor` avec une validation croisée **groupée par bâtiment** (pour éviter le sur-apprentissage géographique).
4. Sauvegarder le booster LightGBM (`height_lgbm.txt`, < 300 Ko).
5. Tester sur un bâtiment jamais vu lors de l'entraînement ; viser un RMSE filtré < 2,0 m.

### Jour 3 — Test en direct (laptop terminal + ADB)
1. Connecter le téléphone au laptop via ADB (`adb devices`).
2. Lancer NetMonster en premier plan sur le téléphone pour forcer le rafraîchissement modem chaque seconde.
3. Exécuter le script principal sur le laptop : lecture ADB `dumpsys telephony.registry`, extraction des cellules 4G, inférence LightGBM, filtrage Kalman, affichage de l'altitude en temps réel dans le terminal.
4. Monter sur le site et vérifier que l'altitude affichée correspond à la hauteur réelle à chaque palier.

---

## Limites et compromis à connaître

- **4G uniquement** : le Timing Advance 4G a une étape de ~78 m, donc peu utile sur 0–60 m ; le modèle s'appuiera principalement sur ΔRSRP et RSRQ.
- **Zone critique 15 m – 30 m** : bascule brutale entre signal masqué par les immeubles et visibilité directe avec les antennes lointaines ; erreur temporaire possible jusqu'à 3–4 m.
- **Dépendance à la densité d'antennes** : en zone très rurale avec une seule antenne visible, la précision se dégrade.
- **Pas de précision centimétrique** : sans baromètre/IMU ni accès au cœur réseau opérateur (UL-AoA, Multi-RTT), l'erreur reste métrique.
- **Généralisation** : un modèle entraîné sur un quartier fonctionne moins bien sur un autre site sans ré-entraînement partiel ou features normalisées par ΔRSRP.

---

## Ce que ce README n'est pas

Ce document ne contient **aucun code source implémenté**. Il définit la spécification fonctionnelle, l'architecture, les références scientifiques, les entrées/sorties, les performances cibles et le plan de mise en œuvre pour développer le pipeline d'estimation d'altitude 100 % 4G/5G passif.
