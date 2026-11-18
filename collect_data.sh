#!/bin/bash
# Script d'aide pour la collecte de données terrain (Jour 1 du workflow README)
# Nécessite : adb installé, téléphone Android en débogage USB, NetMonster ouvert.

OUT_DIR="data/raw"
mkdir -p "$OUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUT_DIR/climb_${TIMESTAMP}.csv"

echo "Collecte des cellules 4G/5G via ADB vers $OUTFILE"
echo "Colonnes : timestamp,pci,rsrp,rsrq,ta,type"
echo "Appuyez sur Ctrl+C pour arrêter."

echo "timestamp,pci,rsrp,rsrq,ta,type" > "$OUTFILE"

while true; do
  TS=$(date +%s)
  RAW=$(adb shell dumpsys telephony.registry 2>/dev/null)

  # Extraction basique des cellules LTE (à adapter selon le constructeur)
  echo "$RAW" | grep -oP 'CellInfoLte:\{[^}]*\}' | while IFS= read -r block; do
    PCI=$(echo "$block" | grep -oP 'mPci=\K\d+')
    RSRP=$(echo "$block" | grep -oP 'rsrp=\K-?\d+')
    RSRQ=$(echo "$block" | grep -oP 'rsrq=\K-?\d+')
    TA=$(echo "$block" | grep -oP 'timingAdvance=\K\d+' || echo "-1")
    if [ -n "$PCI" ] && [ -n "$RSRP" ]; then
      echo "$TS,$PCI,$RSRP,$RSRQ,$TA,LTE"
    fi
  done >> "$OUTFILE"

  sleep 1
done