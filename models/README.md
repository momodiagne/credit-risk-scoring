# Modèles Statistiques & Algorithmes de Scoring

Ce document détaille les choix d'algorithmes retenus pour l'évaluation du risque de crédit ainsi que pour la gestion des données manquantes.

---

## 1. Algorithme Principal de Scoring : LightGBM

### Qu'est-ce que l'algorithme LightGBM ?

LightGBM (Light Gradient Boosting Machine) est un algorithme d'apprentissage supervisé basé sur les arbres de décision et le **Gradient Boosting**. Il est adapté aux tâches de classification et de régression sur des données structurées.

Dans notre projet, l'objectif de LightGBM est de prédire la probabilité qu'un client se retrouve en défaut de paiement (classification binaire : 0 = Solvable, 1 = Défaut).

### Notion de Boosting et d'Arbres séquentiels

Contrairement à un Random Forest où plusieurs arbres sont entraînés indépendamment en parallèle, le **Gradient Boosting** construit les arbres de manière séquentielle :
* Chaque nouvel arbre cherche à corriger les erreurs (les résidus) commises par l'ensemble des arbres précédents.
* À chaque étape, l'algorithme ajuste la prédiction en minimisant une fonction de perte.

### Fonctionnement de l'algorithme étape par étape

1. **Initialisation** :
   Le modèle commence par une prédiction constante de base (par exemple la proportion moyenne de défauts dans le jeu de données).

2. **Calcul des erreurs (Résidus)** :
   Pour chaque individu, le modèle calcule l'écart entre la réalité et la prédiction actuelle.

3. **Construction d'un nouvel arbre** :
   Un nouvel arbre de décision est entraîné pour prédire spécifiquement ces erreurs résiduelles.

4. **Croissance par feuille (Leaf-wise Growth)** :
   Contrairement aux algorithmes de boosting traditionnels qui font grandir les arbres niveau par niveau (level-wise), LightGBM fait grandir l'arbre feuille par feuille en choisissant la feuille qui apporte la plus grande réduction d'erreur.

5. **Mise à jour et Répétition** :
   La nouvelle prédiction est ajoutée aux précédentes avec un facteur de pondération (taux d'apprentissage ou *learning rate*). Les étapes 2 à 5 sont répétées jusqu'à atteindre le nombre d'arbres prédéfini.

### Avantages de LightGBM

* **Rapidité et efficacité mémoire** : Utilise une méthode d'histogrammes pour découper les variables continues, ce qui accélère considérablement l'entraînement.
* **Gestion des relations non linéaires** : Capable de capturer des interactions complexes entre les variables financières (ex: impact de l'âge combiné au taux d'endettement).
* **Robustesse aux classes déséquilibrées** : Très performant lorsque les cas de défaut sont rares par rapport aux clients solvables.

### Limites et Inconvénients

* **Sensibilité au surapprentissage (Overfitting)** : Si le nombre d'arbres est trop élevé ou la profondeur mal réglée, le modèle peut apprendre le bruit des données d'entraînement.
* **Interprétabilité directe complexe** : Un ensemble de plusieurs centaines d'arbres ne peut pas être lu à l'œil nu (nécessite l'utilisation d'outils d'explicabilité comme SHAP).

---

## 2. Algorithme d'Imputation : MissForest (IterativeImputer)

### Qu'est-ce que l'imputation MissForest ?

Dans les dossiers de crédit, la variable `MonthlyIncome` (Revenu Mensuel) présente fréquemment des valeurs manquantes (clients n'ayant pas renseigné leur revenu).

Plutôt que d'utiliser une imputation simple (remplacer par la moyenne ou la médiane), nous utilisons l'algorithme **MissForest** (implémenté via `IterativeImputer` avec un Random Forest).

### Fonctionnement de l'imputation itérative

1. **Initialisation** :
   Les valeurs manquantes du revenu sont temporairement remplacées par la médiane de la colonne.

2. **Modélisation du Revenu** :
   Le revenu devient la variable cible à prédire. Un Random Forest est entraîné en utilisant toutes les autres variables disponibles du client (âge, retards de paiement, nombre de crédits en cours, charges).

3. **Prédiction et Mise à jour** :
   Le modèle prédit la valeur estimée du revenu pour les clients dont le revenu était manquant.

4. **Itération** :
   Le processus est répété jusqu'à ce que les valeurs estimées se stabilisent d'une itération à l'autre.

### Pourquoi avoir choisi MissForest ?

* **Préservation des relations entre variables** : Une imputation simple par la moyenne déforme les distributions et ignore le fait qu'un client ayant un prêt immobilier a statistiquement un revenu différent d'un client jeune sans crédit.
* **Précision renforcée** : MissForest prend en compte l'ensemble du profil financier pour estimer le revenu de façon personnalisée et réaliste.

---

## 3. Explicabilité de la Décision : Méthode SHAP

### Qu'est-ce que la méthode SHAP ?

La méthode SHAP (SHapley Additive exPlanations) est issue de la théorie des jeux de Lloyd Shapley. Elle permet de décomposer la prédiction d'un modèle complexe en attribuant à chaque variable une valeur d'impact (valeur de Shapley).

### Application dans le Scoring Crédit

Pour chaque dossier analysé par le modèle LightGBM :
* SHAP calcule la contribution exacte de chaque critère financier (ex: retards de paiement, âge, taux d'endettement).
* Une valeur SHAP positive indique que le critère augmente le risque de défaut.
* Une valeur SHAP négative indique que le critère réduit le risque de défaut (facteur favorable).

Cette méthode garantit une explication individuelle, transparente et vérifiable pour chaque décision bancaire rendue par l'application.
