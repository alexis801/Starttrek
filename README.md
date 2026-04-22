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

L’objectif final est d’atteindre un score moyen ≥ 200.

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
├── models/ # Modèles sauvegardés (.pth)
├── stats/ # Graphiques de la durée des épisodes, statistiques sur les causes d'arrêt, vidéos.
└── README.md
```

---

## ⚙️ Configuration (YAML)

Chaque expérience est définie via un fichier YAML pouvant être modifié
