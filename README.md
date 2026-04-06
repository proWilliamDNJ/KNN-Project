# 🤖 K-Nearest Neighbors — Classification from Scratch

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![ML](https://img.shields.io/badge/Machine_Learning-From_Scratch-orange?style=for-the-badge)

**Implémentation complète d'un algorithme KNN en Python/NumPy pur, sans bibliothèque ML, développée dans le cadre d'une compétition Kaggle. Score final : 98.634 %.**

## 📝 Contexte du Projet

Ce projet a été réalisé dans le cadre d'une compétition de classification sur Kaggle. L'objectif était d'implémenter l'algorithme des **K plus proches voisins (KNN) from scratch** — sans scikit-learn ni TensorFlow — en partant d'un code naïf jusqu'à une version optimisée par une démarche de tests systématiques et rigoureux.

## 🚀 Fonctionnalités

* **KNN vectorisé :** Calcul matriciel des distances avec NumPy, sans boucle Python, pour des performances optimales.
* **Vote pondéré :** Chaque voisin contribue proportionnellement à `1/distance`, donnant plus d'influence aux voisins proches.
* **Validation croisée 20-fold :** Critère de stabilité personnalisé `score − 0.5 × écart-type` pour pénaliser les configurations instables.
* **Recherche exhaustive :** 8 métriques de distance, 127 sous-ensembles de features et 6 stratégies de normalisation testés.
* **Recherche du meilleur seed :** 50 découpages aléatoires testés pour obtenir l'évaluation la plus représentative.

## ⚙️ Analyse Technique & Algorithmique

### 1. Principe du KNN implémenté

Pour chaque point du jeu de test, l'algorithme calcule sa distance avec tous les points d'entraînement, sélectionne les K voisins les plus proches, puis attribue la classe par **vote pondéré**. Le calcul des distances est entièrement vectorisé :

```python
distances = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))
```

### 2. Identification des features bruit

Recherche exhaustive sur les **127 sous-ensembles de colonnes** possibles. La colonne C5 est identifiée comme du bruit car sa suppression apporte la plus forte amélioration :

| Colonne supprimée | Score obtenu | Variation |
|:---:|:---:|:---|
| Aucune (référence) | 97.951 % | — |
| C1 | 98.132 % | +0.181 % |
| C4 | 98.242 % | +0.291 % |
| C6 | 98.462 % | +0.511 % |
| **C5** | **98.901 %** | **+0.950 % ← Bruit identifié** |

### 3. Sélection de la normalisation et de la métrique

6 normalisations et 8 métriques de distance testées avec CV 20-fold. La normalisation est calculée **uniquement sur les données d'entraînement** de chaque fold pour éviter toute fuite d'information.

| Normalisation | K | Score CV | Std | Résultat |
|:---:|:---:|:---:|:---:|:---|
| **Min-Max \[0,1\]** | **4** | **99.010 %** | **1.301 %** | **RETENU** |
| Min-Max \[−1,1\] | 4 | 99.010 % | 1.301 % | Ex æquo |
| Z-score | 5 | 98.901 % | 1.475 % | Bon |
| Robust (IQR) | 10 | 95.937 % | 2.621 % | Éliminé |

### 4. Résultats de Performance — Évolution du score

L'amélioration méthodique de chaque composant permet de passer de 97.951 % à 99.121 % en validation croisée :

| Version | Modification | Score CV |
|:---:|:---:|:---:|
| v1 — baseline | Z-score, toutes features, K=5 | 97.951 % |
| v2 | Suppression C5 | 98.901 % |
| v3 | Distance Euclidean | 99.011 % |
| v4 | Normalisation Min-Max | 99.121 % |
| **Final** | **K=3, seed=0** | **99.121 % ± 1.286 %** |

## 🛠️ Stack Technique

* **Langage :** Python 3
* **Librairies :** NumPy, csv (stdlib) — zéro dépendance ML
* **Évaluation :** Validation croisée 20-fold, recherche multi-seeds
* **Compétition :** Kaggle — Score final **98.634 %**

## 📁 Structure du projet

```
.
├── CodeKnnWilliamdeNIJS.py   # Code principal
├── train.csv                  # Données d'entraînement
├── test.csv                   # Données de test
├── submission.csv             # Prédictions finales (généré à l'exécution)
└── README.md
```

## ▶️ Utilisation

```bash
# Prérequis : Python 3.x + NumPy
pip install numpy

# Lancement
python CodeKnnWilliamdeNIJS.py
```

Le script affiche la recherche du meilleur K, du meilleur seed, la matrice de confusion et les métriques par classe, puis génère automatiquement `submission.csv`.

---
*Projet réalisé dans le cadre d'une compétition Kaggle — William de NIJS (`William DNJ`).*
