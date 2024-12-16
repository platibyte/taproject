#!/bin/bash
# Falls tensorflow installiert werden soll, muss das manuell gemacht werden:
# 1.conda env erstellen und ipykernel installieren
# 2. mit pip tensorflow installieren
# 3. die anderen benötigten packages installieren, aber NICHT numpy

# Name der Umgebung, die entfernt und neu erstellt werden soll
ENV_NAME="test"

# Umgebung entfernen
echo "Entferne Umgebung '$ENV_NAME'..."
conda env remove --name "$ENV_NAME" --yes

# Cache bereinigen
echo "Bereinige Conda-Cache..."
conda clean --all --yes

# Neue Umgebung erstellen
echo "Erstelle neue Umgebung '$ENV_NAME'..."
conda create --name "$ENV_NAME" python=3.9 numpy=1.26.4 pandas nltk tqdm matplotlib ipykernel --yes

echo "Umgebung '$ENV_NAME' wurde erfolgreich neu erstellt."
sleep 2
conda activate "$ENV_NAME"
