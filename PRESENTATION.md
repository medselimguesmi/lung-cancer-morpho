# Contenu de la présentation (10 min) — Détection de nodules pulmonaires par morphologie mathématique

Binôme : Ahmed Layouni et Selim Guesmi. Données : LUNA16. Tous les chiffres viennent de `results/*.csv`.
Format demandé : 9 diapositives + démo, ~10 min. Style : sobre, une idée par diapo, figures grandes.

## Diapo 1 — Titre et problème (0:45)
- Titre : « Détection de nodules pulmonaires par morphologie mathématique, scanners LUNA16 »
- Problème : trouver les nodules (petits points denses) dans un scanner 3D contenant vaisseaux et paroi qui leur ressemblent.
- Objectif : chaîne segmentation → détection de candidats → classification, comparée en 3 scénarios (A règles, B ML, C CNN 3D), et mesure de la part de la morphologie.
- Avertissement : résultats expérimentaux, pas un outil de diagnostic.

## Diapo 2 — Données et protocole (0:45)
- LUNA16 (issu de LIDC-IDRI), 25 patients de travail : 15 `dev` (réglage, validation croisée) et 10 `test` (évalués une seule fois).
- 5 484 candidats au total, 33 positifs (0,6 %). Test : 2 228 candidats, 11 positifs, 19 nodules annotés.
- Découpage par patient (jamais par candidat) : pas de fuite entre dev et test.

## Diapo 3 — La chaîne en 12 étapes (1:00)
- Schéma : rééchantillonnage 1 mm → seuil -320 HU → **ouverture** → **fermeture + remplissage** → objets denses → **distance + marqueurs** → **watershed** → **troncature en boule** → **squelette** → **gradient morphologique** → texture/intensité → classification.
- **7 étapes sur 12 sont morphologiques (58 %)** (`results/ablation_chaine.csv`).

## Diapo 4 — Segmentation des poumons (1:00)
- Seuillage -320 HU, `clear_border`, ouverture ball(2), 2 plus grandes composantes, fermeture disk(8) par coupe, remplissage des trous.
- **Dice moyen 0,974, IoU moyen 0,950, pire cas 0,930** (25 patients, `segmentation_scores.csv`). Objectif du plan (≥ 0,9) atteint.
- Image : `results/demo/demo_398977448288.png` (panneau 2) ou notebook 02.

## Diapo 5 — Détection des candidats (1:30)
- Distance euclidienne → marqueurs (maxima locaux, érosion ultime) → watershed → **troncature en boule** (notre ajout : coupe le vaisseau ou la paroi collés) → squelette + filtre de taille 3–30 mm.
- Diagnostic du baseline (13 nodules manqués) : 6 filtre squelette, 5 pas de graine, 2 seuil trop haut.
- Résultat : rappel dev 0,48 → **0,88** (22/25), FP/scan 16 → 216. Test : **0,58** (11/19), 222 FP/scan.
- Honnêteté : le réglage sur 15 scans généralise mal (0,88 → 0,58).
- Image : `results/detection_figures/1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.png`.

## Diapo 6 — Descripteurs et scénarios (1:00)
- 17 descripteurs : forme (sphéricité, solidité, élongation), squelette, gradient morphologique, distance à la paroi, HU, GLCM.
- A : règles par seuils (réglées sur dev) ; B : SVM / Random Forest / XGBoost, validation croisée groupée, **Random Forest retenu** ; C : CNN 3D PyTorch sur patchs 32³ (perte pondérée, suréchantillonnage, augmentation).

## Diapo 7 — Résultats sur le test (1:15)
| Scénario | AUC-ROC | Sens. à 4 FP/scan | Temps |
|---|---|---|---|
| A règles | 0,75 | 0,05 | 0,1 s |
| B morpho + ML | 0,84 | 0,26 | 0,65 s |
| C morpho + CNN 3D | 0,87 | 0,16 | 869 s |
- Images : `results/figures/roc.png`, `results/figures/froc.png`.
- Message : B meilleur compromis (rapide, interprétable, meilleure FROC) ; C meilleure AUC mais 1 300 fois plus lent. Avec 11 positifs, écarts non significatifs. La FROC plafonne à 0,58 : c'est le rappel de la détection.

## Diapo 8 — Ablation : ce que chaque opérateur apporte (1:00)
| Étape retirée | Δ rappel test | FP/scan |
|---|---|---|
| Fermeture | -0,21 | 222 → 193 |
| Watershed | -0,16 | 222 → **18** |
| Troncature en boule | -0,05 | 222 → 126 |
| Ouverture | +0,05 (pas d'effet net) | 231 |
| Squelette | 0 | 222 |
- Descripteurs : sans forme -0,06 d'AUC ; sans gradient -0,003 ; sans squelette +0,05 (le squelette est redondant après la troncature).
- Conclusion : fermeture et watershed sont les opérateurs décisifs pour le rappel ; le watershed échange du rappel contre des FP.

## Diapo 9 — Démo, erreurs, limites (2:00)
- **Démo (≈1 min)** : `python src/demo.py` → scanner, masque, candidats, décision. Exemple : sur le patient …398977448288, 280 candidats, 1 retenu au seuil 0,3 (le nodule, score 0,69). Plan B : image `results/demo/demo_398977448288.png`.
- Erreurs (`results/figures/erreurs.png`) : FP = vaisseaux, hile, paroi ; nodules manqués = petits, collés à la plèvre.
- Limites : 25 patients, 11 positifs en test, un seul jeu de données, pas de validation clinique, CNN peu entraînable.
- Perspectives : plus de patients (LUNA16 complet, 888), validation croisée, détecteur de vaisseaux, CNN pré-entraîné.

## Questions probables
- **Pourquoi disk(8) pour la fermeture ?** Assez grand pour combler les vaisseaux et réintégrer les nodules juxta-pleuraux ; sans fermeture, rappel -0,21 (ablation).
- **Pourquoi le watershed ?** Sépare les nodules collés aux vaisseaux ; sans lui rappel -0,16, mais 18 FP/scan au lieu de 222.
- **Pourquoi la troncature en boule ?** Le filtre squelette rejetait les nodules attachés à un vaisseau ; la boule coupe l'attache avant le filtre.
- **Limites du deep learning ?** 33 positifs : CNN instable, pas mieux que B de façon significative, 869 s d'entraînement CPU.
- **Pourquoi le rappel baisse sur test ?** Réglage sur 15 scans seulement ; pistes : plus de patients.
- **Pas de fuite de données ?** Seuils, règles et hyperparamètres choisis sur dev uniquement, découpage par patient.

## Répartition de la parole (à ajuster)
- Selim : diapos 1–4 (données, segmentation). Ahmed : diapos 5–8 (détection, classification, ablation). Les deux : diapo 9 (démo, conclusion).
