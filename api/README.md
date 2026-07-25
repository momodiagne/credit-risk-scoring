# Service API (Moteur de Calcul)

Ce dossier contient le moteur de calcul de l'application (API REST).

## Rôle principal

Le moteur de calcul agit comme un intermédiaire entre l'interface utilisateur et le système d'évaluation financière :
1. Il reçoit les données saisies par le conseiller (âge, revenus, retards, etc.).
2. Il calcule la probabilité de risque de défaut du client.
3. Il détermine l'avis final (Accordé ou Refusé) ainsi que la liste des facteurs influençant la décision.
4. Il renvoie le résultat instantanément à l'interface graphique.

## Fichiers du dossier

* `main.py` : Point d'entrée du serveur qui gère les demandes de calcul.
* `model_loader.py` : Module chargé de lire les fichiers d'évaluation et d'effectuer la préparation des variables.
* `schemas.py` : Structure des données échangées entre le navigateur et le serveur.
