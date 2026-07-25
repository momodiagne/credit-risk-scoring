# Imputation des Données Manquantes & Stratégie Métier

Ce document explique en détail la méthodologie d'imputation retenue pour traiter les valeurs manquantes dans le projet de scoring de crédit.

---

## 1. Problématique des Données Manquantes dans le Crédit

Dans les bases de données bancaires, la variable **`MonthlyIncome`** (Revenu Mensuel) présente très souvent un taux élevé d'omissions (environ 20% de valeurs manquantes dans le jeu de données GMSC). 

Plusieurs raisons expliquent ces manques :
* Le client a refusé de communiquer son revenu.
* Le client n'a pas de revenu fixe (travailleur indépendant, intérimaire).
* Omission lors de la saisie du dossier.

En Machine Learning, de nombreux modèles (comme la Régression Logistique ou certaines forêts d'arbres) ne peuvent pas traiter directement les valeurs manquantes (`NaN`). Il est donc indispensable d'imputer ces données.

---

## 2. Pourquoi les méthodes d'imputation classiques sont insuffisantes

### Imputation par la Moyenne ou la Médiane

La méthode la plus simple consiste à remplacer tous les revenus manquants par la moyenne globale ou la médiane des revenus de la base.

**Limites de cette approche** :
* **Destruction des corrélations** : Elle suppose qu'un jeune diplômé de 22 ans et un cadre supérieur de 50 ans sans revenu déclaré ont exactement le même revenu.
* **Réduction de la variance** : Création d'un pic artificiel de données autour de la médiane, ce qui biaise les distributions.
* **Perte d'information** : Le modèle perd la capacité de distinguer le profil réel de l'emprunteur.

---

## 3. L'Approche Retenue : Imputation par MissForest (IterativeImputer)

### Qu'est-ce que l'algorithme MissForest ?

**MissForest** est un algorithme d'imputation itératif basé sur des forêts d'arbres décisionnels (**Random Forest**). 

Au lieu de traiter l'imputation comme un simple calcul statistique global, MissForest traite le remplacement du revenu comme un **problème de prédiction supervisée**.

### Fonctionnement de l'algorithme étape par étape

1. **Initialisation** :
   Les valeurs manquantes du revenu (`MonthlyIncome`) sont temporairement remplacées par la médiane globale de la colonne.

2. **Construction du jeu de données d'apprentissage** :
   * Les clients ayant un revenu connu servent d'ensemble d'entraînement.
   * Les clients ayant un revenu manquant servent d'ensemble de test à prédire.

3. **Entraînement du Random Forest** :
   Un modèle Random Forest est entraîné à prédire le revenu à partir des 9 autres caractéristiques connues du client :
   * Âge de l'emprunteur
   * Taux d'utilisation du crédit revolving
   * Taux d'endettement
   * Nombre de crédits actifs et prêts immobiliers
   * Historique des retards de paiement (30-59j, 60-89j, 90j+)
   * Nombre de personnes à charge

4. **Prédiction et Mise à jour** :
   Le Random Forest prédit un revenu personnalisé pour chaque client dont la valeur était manquante.

5. **Convergence itérative** :
   Les étapes 2 à 4 sont répétées plusieurs fois jusqu'à ce que les revenus imputés se stabilisent.

---

## 4. Prévention de la Fuite de Données (Data Leakage)

Un point critique en ingénierie des données est d'éviter toute **fuite de données (Data Leakage)** :

* **Séparation stricte Entraînement / Production** :
  L'algorithme MissForest (`IterativeImputer`) est ajusté (*fitted*) **uniquement** sur le jeu de données d'entraînement (`X_train`).
* **Sauvegarde de l'imputer** :
  L'objet imputer entraîné est sauvegardé sous forme d'artefact (`models/income_imputer.joblib`).
* **Réutilisation en inférence API** :
  Lorsque le service FastAPI reçoit un nouveau dossier client sans revenu renseigné, il utilise le même objet `income_imputer.joblib` pour estimer le revenu de manière rigoureuse et cohérente avec la phase d'entraînement.

---

## 5. Avantages et Inconvénients de MissForest

### Avantages

* **Haute précision** : Estime un revenu réaliste adapté au profil spécifique de chaque client (âge, charges, crédits).
* **Respect des distributions** : Conserve la variance naturelle des données et les corrélations entre variables.
* **Généralisation optimale** : Permet au modèle de scoring final (LightGBM) de prendre des décisions plus justes.

### Inconvénients

* **Coût en temps de calcul** : L'entraînement de plusieurs Random Forest itératifs est plus long qu'un simple calcul de médiane (résolu dans notre projet en sauvegardant l'imputer pré-entraîné).
