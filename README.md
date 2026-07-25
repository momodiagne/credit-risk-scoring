# Système d'Aide à la Décision & d'Analyse de Crédit Bancaire

Un outil décisionnel bancaire moderne, transparent et intuitif conçu pour accompagner les conseillers dans l'évaluation de la solvabilité des emprunteurs et la maîtrise du risque de crédit.

---

## À quoi sert ce projet ?

Lorsqu'un client sollicite un prêt bancaire, l'établissement doit analyser sa situation financière pour s'assurer qu'il sera en mesure de rembourser ses mensualités sans se retrouver en situation de surendettement.

Cette application permet aux conseillers bancaires de :
1. **Saisir rapidement les informations financières d'un candidat à l'emprunt** (revenus, charges, retards de paiement, engagements en cours).
2. **Obtenir instantanément un avis décisionnel clair** (*Avis Favorable / Crédit Accordé* ou *Avis Défavorable / Crédit Refusé*).
3. **Comprendre précisément les raisons de la décision** grâce à un découpage transparent des facteurs de risque et des points forts du dossier.

---

## Pourquoi cette approche est-elle innovante ?

Contrairement aux outils traditionnels obscurs (effet "boîte noire"), cette solution met l'accent sur la **transparence** et l'**explicabilité** :

* **Zéro décision arbitraire** : Chaque avis repose sur l'analyse objective de l'historique de paiement, du niveau d'endettement et de la capacité financière.
* **Explication détaillée** : Le conseiller peut visualiser directement les éléments qui jouent en faveur du client (ex: *faible utilisation des crédits*) et les critères de vigilance (ex: *retards de paiement passés*).
* **Conformité & Équité** : Garantit une analyse équitable et conforme aux exigences des régulateurs bancaires.

---

## Comment fonctionne l'interface ? (En 3 étapes)

```
+--------------------------------+       +------------------------------------+
|  1. Saisie du Dossier Client   | ----> |  2. Évaluation de la Solvabilité   |
| (Revenu, Endettement, Retards) |       |  (Score de risque & Avis Favorable)|
+--------------------------------+       +------------------------------------+
                                                           |
                                                           v
                                         +------------------------------------+
                                         |  3. Explication de la Décision     |
                                         | (Facteurs clés & Points d'attention|
                                         +------------------------------------+
```

1. **Panneau latéral de Saisie** : Le conseiller renseigne le profil de l'emprunteur (âge, charges, revenus, historique bancaire).
2. **Carte Décisionnelle** : Un banner lisible affiche immédiatement l'avis final et la jauge de risque globale.
3. **Analyse des Critères & Synthèse** : Des graphiques clairs détaillent les points forts et les axes de préoccupation du dossier.

---

## Comment lancer l'application ?

Pour démarrer l'application et l'ouvrir dans votre navigateur :

1. **Installer les prérequis** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Démarrer les services** :
   ```bash
   # Lancer le moteur de calcul (API)
   uvicorn api.main:app --reload

   # Lancer l'interface utilisateur (Dashboard)
   streamlit run dashboard/app.py
   ```

3. **Consulter l'application** : Ouvrez votre navigateur sur **http://localhost:8501**.

---

## Confidentialité & Éthique
Toutes les données saisies sont traitées en temps réel de manière confidentielle afin de garantir le respect de la vie privée des emprunteurs.