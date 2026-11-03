"""Paramètres métier partagés entre l'entraînement, l'évaluation et l'API."""

# HYPOTHÈSES MÉTIER : GMSC ne contient aucune donnée de coût.
# Valeurs illustratives, exprimées en % du montant prêté.

# Perte si on accorde un crédit à un client qui fait défaut
LOSS_GIVEN_DEFAULT = 0.58

# Marge nette perdue si on refuse un bon client
NET_MARGIN_GOOD_LOAN = 0.08

COST_RATIO_FN_FP = LOSS_GIVEN_DEFAULT / NET_MARGIN_GOOD_LOAN

# Refus si p * perte > (1 - p) * marge  <=>  p > marge / (perte + marge)
DECISION_THRESHOLD = NET_MARGIN_GOOD_LOAN / (LOSS_GIVEN_DEFAULT + NET_MARGIN_GOOD_LOAN)