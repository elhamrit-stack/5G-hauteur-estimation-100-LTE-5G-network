"""ADB collector: lit les cellules 4G/5G depuis un téléphone Android connecté en USB.

Utilise `adb shell dumpsys telephony.registry` pour extraire passivement
RSRP, RSRQ, PCI et Timing Advance sans application dédiée ni root.
"""

import subprocess
import re
from typing import List, Dict, Optional


def run_adb_command(cmd: list, timeout: float = 2.0) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except Exception:
        return ""


def poll_cells_adb() -> List[Dict]:
    """Retourne la liste des cellules détectées par le modem Android.

    Chaque dictionnaire contient : pci, rsrp (dBm), rsrq (dB), ta, type.
    """
    raw = run_adb_command(["adb", "shell", "dumpsys", "telephony.registry"])
    if not raw:
        return []

    cells: List[Dict] = []

    # --- 4G LTE ---
    # Bloc type : CellInfoLte:{... mPci=123 rsrp=-88 rsrq=-9 ... ta=3 ...}
    lte_blocks = re.findall(r"CellInfoLte:\{[^}]*\}", raw)
    for block in lte_blocks:
        pci_m = re.search(r"mPci=(\d+)", block)
        rsrp_m = re.search(r"rsrp=(-?\d+)", block)
        rsrq_m = re.search(r"rsrq=(-?\d+)", block)
        ta_m = re.search(r"timingAdvance=(\d+)", block) or re.search(r"ta=(\d+)", block)

        if pci_m and rsrp_m:
            pci = int(pci_m.group(1))
            rsrp = int(rsrp_m.group(1))
            if 0 <= pci < 504 and -140 <= rsrp <= -40:
                cells.append({
                    "pci": pci,
                    "rsrp": float(rsrp),
                    "rsrq": float(rsrq_m.group(1)) if rsrq_m else -11.0,
                    "ta": int(ta_m.group(1)) if ta_m else -1,
                    "type": "LTE"
                })

    # --- 5G NR ---
    nr_blocks = re.findall(r"CellInfoNr:\{[^}]*\}", raw)
    for block in nr_blocks:
        pci_m = re.search(r"mPci=(\d+)", block)
        rsrp_m = re.search(r"ssRsrp=(-?\d+)", block) or re.search(r"rsrp=(-?\d+)", block)
        rsrq_m = re.search(r"ssRsrq=(-?\d+)", block) or re.search(r"rsrq=(-?\d+)", block)

        if pci_m and rsrp_m:
            pci = int(pci_m.group(1))
            rsrp = int(rsrp_m.group(1))
            if 0 <= pci < 1008 and -140 <= rsrp <= -40:
                cells.append({
                    "pci": pci,
                    "rsrp": float(rsrp),
                    "rsrq": float(rsrq_m.group(1)) if rsrq_m else -11.0,
                    "ta": -1,
                    "type": "NR"
                })

    # Déduplication par PCI (on garde la mesure la plus forte)
    unique: Dict[int, Dict] = {}
    for c in cells:
        if c["pci"] not in unique or c["rsrp"] > unique[c["pci"]]["rsrp"]:
            unique[c["pci"]] = c
    return list(unique.values())


def adb_available() -> bool:
    out = run_adb_command(["adb", "devices"], timeout=2.0)
    return "device" in out and "List of devices" in out