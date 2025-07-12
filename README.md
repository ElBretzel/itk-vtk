# itk-vtk


Ce projet implémente un visualisateur de l'evolution d'une tumeur en ITK / VTK

## 2) Recalage d'images

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

### Algorithme
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
## 4) Analyse et visualisation des changements
```

