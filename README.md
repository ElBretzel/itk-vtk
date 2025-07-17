# Projet itk-vtk

Ce projet implémente un visualisateur de l'evolution d'une tumeur en ITK / VTK

## Auteurs
- Luca Descrambes
- Bulle Mostovoi
- Paul Haas
- Solène Dintinger

## 1) Données
Création d'une fonction simple pour ouvrir nos 2 images.

## 2) Recalage d'images

### Interpolation et transformée

Nous avons étudié plusieurs types de transformations et d'interpolations afin de sélectionner celles qui nous paraissent intéressantes dans notre cas et de les implémenter.

1. **Rigide**
- Conserve les distances et angles entre les points
- Utilise rotation et translation
- **(+)** : Simple et utile quand la forme d'un objet est la même
- **(-)** : Pas utile pour objets qui se déforment, précision faible

2. **Affine**
- Linéaire
- Utilise plus que rotation et translation
- **(+)** : Plus pratique pour différences d'échelles et de taille entre 2 fichiers
- **(-)** : Peut se tromper avec des mauvais paramètres donc fausse le résultat

3. **B-Spline**
- Pour modéliser les déformations plus complexes (mouvements)
- **(+)** : Plus de précision et plus adapté au sujet
- **(-)** : Assez lent et couteux en mémoire

Nous avons choisi de partir sur une interpolation B-Spline car c'est celle qui nous permet d'afficher le plus précisement possible la différence de position de la tumeur entre nos 2 fichiers.

Pour la transformée, le choix a été plus compliqué. Comme nous n'avons que 2 fichiers à notre disposition et que nous utilisons nos ordinateurs personnels, nous avons décidé d'utiliser une transformation rigide, qui est alors plus simple et moins couteuse en temps et en mémoire, tout en restant une option valide. Le manque d'informations de la transformée rigide est en partie compensée par l'utilisation de l'interpolation B-Spline, tout en ayant une bonne qualité visuelle finale.

### Métrique

Pour la métrique du recalage, nous nous sommes intéressé à la méthode **Mutual Information de Mattes**.
- **(+)** : Bien adaptée pour images médicales, robuste à des différences d'intensité
- **(-)** : Sensible au bruit, assez couteuse


### Optimiseur

Pour l'optimiseur du recalage, nous nous sommes intéressé à la méthode de la **descente de gradient à pas régulier**.
- **(+)** : Compatible avec notre métrique, plus simple et stable
- **(-)** : Convergence lente ou divergence selon le pas



### Algorithme choisi
- **Type de transformation** : Rigide (conservation des distances et angles)
- **Métrique de similarité** : Mutual Information de Mattes
- **Optimiseur** : Descente de gradient à pas régulier

### Paramètres choisis
```python
metric.SetNumberOfHistogramBins(50)  # Nombre de bins pour l'histogramme de l'information mutuelle

optimizer.SetNumberOfIterations(200)  # Nombre maximum d'itérations
optimizer.SetMinimumStepLength(0.001) # Pas minimum
optimizer.SetLearningRate(0.75)       # Taux d'apprentissage
optimizer.SetRelaxationFactor(0.8)    # Facteur de relaxation
optimizer.SetGradientMagnitudeTolerance(1e-4)  # Tolérance sur le gradient
```

## 3) Segmentation des tumeurs

Pour la segmentation, nous avons choisi parmi plusieurs implémentations, automatiques ou semi-automatiques (Kmeans, SVM, Watershed, ...), la méthode de **ConfidenceConnected**.

Cette méthode permet de faire une segmentation semi-automatique qui trouve notre région d'intérêt depuis un ou plusieurs points initiaux et qui utilise les voxels proches d'un seed fixé.

- **(+)** : Simple et rapide, efficace pour régions homogènes (comme tumeurs)
- **(-)** : Faire un bon choix de seed, mauvais pour structures complexes


### Algorithme choisi
- **Méthode Principale** : Croissance des régions
- **Post-traitement** : Opérations morphologiques


### Paramètres Clés
```python
segmenter.SetMultiplier(2.0)               # Contrôle la déviation d'intensité
segmenter.SetNumberOfIterations(5)         # Itérations de croissance
segmenter.SetInitialNeighborhoodRadius(3)  # Taille de région initiale
segmenter.SetReplaceValue(1)              # Valeur de sortie pour la région segmentée

# Traitement morphologique :
rayon = 1  # Taille de l'élément structurant
```

## 4) Analyse et visualisation des changements

### Visualisation choisie

Nous avons décidé d'utiliser une visualisation 3D de notre résultat final car cela donne un rendu intéractif. Il est alors plus pratique de visualiser, pour n'importe qui, la position exacte de la tumeur et nous pouvons aussi ajouter des couleurs qui nous donne des informations supplémentaires.

Code couleur :
- **Blanc** : Le fond et les parties identiques dans les 2 images
- **Gris** : Le contour du crane du patient
- **Rouge** : Les données qui sont dans la première image mais pas la seconde
- **Bleu** : Les données qui sont dans la seconde image mais pas la première
```python
#FIXME Couleurs
```

Avantages :
- **Interactif** : On peut déplacer le modèle pour voir tous ces angles
- **En couleur** : Permet de remarquer rapidement le résultat

Limitations :
- **Intérieur de la tumeur** : Puisque c'est un affichage 3D, on ne peut voir que l'extérieur et ne voit pas les différences qui pourraient être à l'intérieur

```python
#FIXME A modifier si on change la visualisation
```