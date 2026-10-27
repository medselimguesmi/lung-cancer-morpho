# lung-cancer-morpho

## Phase 3 : détection des candidats

La détection est implémentée dans `src/detection.py` et s'appuie sur le masque
pulmonaire de la phase 2. Les volumes et masques utilisent l'ordre `(z, y, x)`.
La chaîne seuile les structures denses à l'intérieur des poumons, calcule les
maxima de la transformée de distance, sépare les objets par watershed, puis
filtre les régions par diamètre et descripteurs de squelette. Les fonctions
`morphological_gradient` et `white_tophat` servent à comparer les opérateurs
sur les coupes.

Les paramètres de détection sont dans `config.yaml`. Pour traiter les patients
de développement sélectionnés dans `results/patients.csv` :

```powershell
python src/run_detection.py --split dev
```

Pour un contrôle rapide sur un scan, ajoutez `--limit 1`. Les options `--split
test` et `--split all` existent, mais le jeu `test` doit rester réservé à
l'évaluation finale et ne doit pas servir au réglage des paramètres.

Le script écrit :

- `results/candidates.csv` : un candidat par ligne, avec centre physique
	`(coordX, coordY, coordZ)`, diamètre, volume, intensité et descripteurs du squelette.
- `results/detection_eval.csv` : rappel et faux positifs par scan. Un candidat
	correspond à une annotation au plus si son centre est à l'intérieur du rayon
	annoté; l'appariement est un-à-un.
- `results/detection_figures/` : figures de contrôle avec CT, masque, gradient,
	top-hat, watershed, squelettes retenus et annotations.

Avec les paramètres actuels et les 15 scans `dev` de `results/patients.csv`, le
baseline retrouve 12 des 25 nodules annotés (rappel `0,48`) et produit en moyenne
`16,2` faux positifs par scan. Ces performances restent insuffisantes : les
résultats sont expérimentaux et ne constituent pas un outil de diagnostic.
Les seuils ont été ajustés sur le jeu `dev`; toute amélioration doit être
mesurée sur ce jeu sans consulter le split `test`.