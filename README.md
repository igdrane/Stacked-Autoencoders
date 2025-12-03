# Classification de chiffres par autoencodeurs empilés (MATLAB)

Ce dépôt contient le code MATLAB utilisé pour à la **classification d’images de chiffres manuscrits** 
à l’aide d’**autoencodeurs empilés** (stacked autoencoders) et d’une couche softmax.

L’objectif principal est d’étudier expérimentalement :

1. L’effet du **fine-tuning global** sur les performances.
2. L’impact de l’**ajout d’un deuxième autoencodeur** sur le compromis
   performance / coût calculatoire.
3. Jusqu’à quel point on peut **optimiser la taille des couches cachées**
   (nombre de neurones) tout en gardant une précision de classification
   élevée (≈ 99 %).

Les scripts génèrent automatiquement les **matrices de confusion**,
les **courbes d’accuracy / erreur** et le **nombre de paramètres**
pour différentes architectures.

---

## 1. Organisation du dépôt

- `script1_finetuning_2AE.m`  
  Étude de l’effet du fine-tuning et de l’ajout d’un deuxième autoencodeur.
  Quatre cas sont testés :
  - Cas 1 : `AE1 + softmax` (sans fine-tuning)
  - Cas 2 : `AE1 + softmax` (avec fine-tuning)
  - Cas 3 : `AE1 + AE2 + softmax` (sans fine-tuning)
  - Cas 4 : `AE1 + AE2 + softmax` (avec fine-tuning)

  Les résultats (matrices de confusion, récapitulatif) sont enregistrés
  dans un dossier du type :
  - `results_ft_2AE/`

- `script2_opt_H1.m`  
  Optimisation de la taille de la première couche cachée **H1** (encodeur 1),
  avec **H2 fixé à 50** (encodeur 2).  
  Valeurs testées : `H1 = [20 40 60 80 100]`.  
  Pour chaque H1 :
  - Entraînement AE1 et AE2
  - Fine-tuning global (AE1+AE2+softmax)
  - Calcul : accuracy globale, erreur, accuracy par classe, nb de paramètres
  - Sauvegarde des matrices de confusion et des courbes :
    - `opt_accuracy_vs_H1.png`
    - `opt_error_vs_H1.png`
    - `opt_nParams_vs_H1.png`

  Résultats enregistrés dans :
  - `results_opt_H1_AE1AE2_FT/`

- `script3_opt_H2.m`  
  Optimisation de la taille de la deuxième couche cachée **H2** (encodeur 2),
  avec **H1 fixé à 60**.  
  Valeurs testées : `H2 = [10 20 30 40]`.  
  Pour chaque H2 :
  - Entraînement AE1 (H1=60) et AE2 (H2 variable)
  - Fine-tuning global (AE1+AE2+softmax)
  - Calcul : accuracy globale, erreur, accuracy par classe, nb de paramètres
  - Sauvegarde des matrices de confusion et des courbes :
    - `opt_accuracy_vs_H2.png`
    - `opt_error_vs_H2.png`
    - `opt_nParams_vs_H2.png`

  Résultats enregistrés dans :
  - `results_opt_H2_AE1AE2_FT/`
