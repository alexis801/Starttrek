# 🚀 START TREK -> Reinforcement Learning

**START TREK** est un projet de **reinforcement learning** dont l’objectif est d’entraîner un agent autonome à faire atterrir un module lunaire en toute sécurité dans l’environnement Gymnasium (*LunarLander-v3*).

---

## 🎯 Objectif

Développer un agent capable de :

- Contrôler un module spatial
- Apprendre par **essais/erreurs**
- Maximiser une récompense basée sur :
  - un atterrissage stable
  - une vitesse faible
  - une bonne orientation

L’objectif final est d’atteindre un score moyen ≥ 200 sur 100 épisodes.

---

## 🧠 Méthodologie

- Reinforcement Learning
- Boucle : **state → action → reward**
- Deep Q-Learning

---

## 📁 Structure du projet
```
│── train.py # Entraînement
├── eval.py # Évaluation
├── configs/ # Fichier YAML (env, algo, hypers, seeds)
    |── default.yaml # Paramétres par défaut
    |── train.yaml # Overide pour entrainement
    |── eval.yaml # Overide pour eval
├── models/ # Modèles sauvegardés (.pth)
├── stats/  # Graphiques de la durée des épisodes, statistiques sur les causes d'arrêt, vidéos.
    |── Csv/ Résultat des épisodes
    |── Plots/ Graphiques (reward, epsilon ...)
    |── Videos/ #Vidéo des épisodes
|──Utils/
    |── config.py # Chargement et fusion des configs yaml
    |── seed.py # Reproducibilité (random, torch, numpy)
└── README.md
```

---

## ⚙️ Installation (Linux)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

---


## 🚀 Uttilisation

Entrainer l'agent:

```
python3 train.py
```

Évaluer un modèle :

```
python3 eval.py --model models/model.pth --episodes 100
```

Options eval :

```
--model : chemin du modéle
--épisodes : nombred'épisodes
--seed : seed de départ
--render-mode : rgb_array | human
--csv_out : Chemin du fichier Csv de sortie
```
---


## ⚙️ Configuration (YAML)

Chaque expérience est définie via des fichiers YAML dans configs/

```
--default.yaml : paramétres de base
--train.yaml & eval.yaml : overident unniquement ce qui change
```

```
from utils.config import load_merged_config
cfg = load_merged_config("configs/default.yaml", "configs/train.yaml")
```
---


🔁 Reproducibilité

-Les seeds sont fixé par utils/seed.py : 5 seed sont utilisés (Python, NumPy, PyTorch CPU/GPU, cuDNN)