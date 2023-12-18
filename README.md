# Hardware for signal processing
Anthony Gallien et Maéva Bachelard


# Projet et objectifs

1. **Apprentissage de CUDA :** Acquérir une compréhension approfondie de l'utilisation de CUDA pour l'accélération des calculs parallèles sur GPU.

2. **Étude de la complexité des algorithmes :** Analyser la complexité des algorithmes, en mettant particulièrement l'accent sur les gains d'accélération observés lors de l'exécution sur GPU par rapport à CPU.

3. **Observation des limites du GPU :** Explorer les limites et les contraintes associées à l'utilisation d'un GPU, en particulier dans le contexte des tâches liées à l'apprentissage profond.

4. **Implémentation d'un CNN (Convolutional Neural Network) :**
   - Concevoir et mettre en œuvre "from scratch" la partie d'inférence d'un CNN.
   - Choix du CNN : LeNet-5, une architecture classique largement reconnue pour son efficacité dans la reconnaissance d'images.

5. **Interopérabilité entre Python et CUDA :** Exporter des données depuis un notebook Python et les réimporter dans un projet CUDA, assurant une intégration fluide entre les phases de développement.

6. **Gestion de projet avec Git :** Mettre en place un suivi de projet efficace en utilisant l'outil Git pour le versionning, facilitant la collaboration et le suivi des changements.


# Calcul Matriciel CPU-GPU

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


# Premières couches du réseau de neurone LeNet-5

1. **Layer 1 - Génération des données de test :**

Dans le cadre de l'implémentation d'une opération de Convolution 2D, les matrices suivantes doivent être créées et initialisées :

**Matrice `raw_data` de Taille 32x32 :**
   - Une matrice float `raw_data` de taille 32x32 est créée et initialisée avec des valeurs comprises entre 0 et 1, correspondant à nos données d'entrée.

**Matrice `C1_data` de Taille 6x28x28 :**
   - Une matrice float `C1_data` de taille 6x28x28 est initialisée à 0 et prendra les valeurs de sortie de la convolution 2D. C1 correspond aux données après la première Convolution.

**Matrice `S1_data` de Taille 6x14x14 :**
   - Une matrice float `S1_data` de taille 6x14x14 est initialisée à 0 et prendra les valeurs de sortie du sous-échantillonnage. S1 correspond aux données après le premier Sous-échantillonnage.

**Matrice `C1_kernel` de Taille 6x5x5 :**
   - Une matrice float `C1_kernel` de taille 6x5x5 est initialisée avec des valeurs comprises entre 0 et 1, correspondant à nos premiers noyaux de convolution.

2. **Layer 2 - Convolution 2D :**
   - Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

3. **Layer 3 - Sous-échantillonage**
- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

4. **Fonction d'activation**
- On ajoute la fonction tanh en fin de couche commme fonction d'activation.
