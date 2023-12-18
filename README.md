# Hardware for signal processing
Anthony Gallien et Maéva Bachelard


# Projet CUDA avec Implémentation LeNet-5

1. **Apprentissage de CUDA :** Acquérir une compréhension approfondie de l'utilisation de CUDA pour l'accélération des calculs parallèles sur GPU.

2. **Étude de la complexité des algorithmes :** Analyser la complexité des algorithmes, en mettant particulièrement l'accent sur les gains d'accélération observés lors de l'exécution sur GPU par rapport à CPU.

3. **Observation des limites du GPU :** Explorer les limites et les contraintes associées à l'utilisation d'un GPU, en particulier dans le contexte des tâches liées à l'apprentissage profond.

4. **Implémentation d'un CNN (Convolutional Neural Network) :**
   - Concevoir et mettre en œuvre "from scratch" la partie d'inférence d'un CNN.
   - Choix du CNN : LeNet-5, une architecture classique largement reconnue pour son efficacité dans la reconnaissance d'images.

5. **Interopérabilité entre Python et CUDA :** Exporter des données depuis un notebook Python et les réimporter dans un projet CUDA, assurant une intégration fluide entre les phases de développement.

6. **Gestion de projet avec Git :** Mettre en place un suivi de projet efficace en utilisant l'outil Git pour le versionning, facilitant la collaboration et le suivi des changements.


# Projet de Calcul Matriciel CPU-GPU

1. **Création de Matrice sur CPU :**
   - La fonction `MatrixInit` initialise une matrice de taille n x p avec des valeurs aléatoires entre -1 et 1.
   - La fonction `MatrixPrint` affiche une matrice de taille n x p.
   
2. **Addition de Deux Matrices sur CPU :**
   - La fonction `MatrixAdd` additionne deux matrices M1 et M2 de même taille n x p sur CPU.

3. **Addition de Deux Matrices sur GPU :**
   - La fonction `cudaMatrixAdd` additionne deux matrices M1 et M2 de même taille n x p sur GPU. Les dimensions des matrices sont à considérer comme les paramètres gridDim et blockDim, plusieurs possibilitées sont utilisées en utilisant les blocks et les threads.

4. **Multiplication de Deux Matrices NxN sur CPU :**
   - La fonction `MatrixMult` multiplie deux matrices M1 et M2 de taille n x n sur CPU.

5. **Multiplication de Deux Matrices NxN sur GPU :**
   - La fonction `cudaMatrixMult` multiplie deux matrices M1 et M2 de taille n x n sur GPU. Les dimensions des matrices sont à considérer comme les paramètres gridDim et blockDim.

6. **Complexité et Temps de Calcul :**
   - Mesurer le temps CPU et GPU à l'aide de `<time.h>` ou `nvprof`.
   - Confronter les résultats avec les caractéristiques du GPU utilisé.
   - Faire varier les tailles et dimensions des grid et block (par exemple, gridDim = 1 et blockDim = (nombre de lignes x nombre de colonnes)) pour explorer l'impact sur les performances.

Ce projet offre une opportunité d'explorer les performances des opérations matricielles sur CPU et GPU, en tenant compte des aspects de complexité, d'accélération théorique et de temps de calcul réel.
