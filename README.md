# AI53 — MLOps Phase 1 : Beans

Pipeline terminé : validation des 1 295 images, préparation déterministe, trois
entraînements contrôlés, sélection sur validation et test final du modèle choisi.

- [Rapport du lab : audit, checkpoints, prédictions et résultats](LAB_REPORT.md)
- [Guide officiel](docs/AI53_Phase1_Lab_Guide.pdf)
- [Comparaison des Runs et contrôle du logging](reports/experiment_comparison.json)
- [Scores du test final](reports/test_metrics.json)
- [Matrice de confusion](reports/test_confusion_matrix.png)

Le CNN et les tests sont ceux du cours. Les anciens TP AI53 sont dans le dossier
parent ; ce projet possède son environnement et son dépôt Git indépendants.

## Résultats

| Run | Seul changement par rapport à A | Macro-F1 validation, époque 4 |
|---|---|---|
| A | Baseline : LR 0,001, dropout 0,30 | 0,625419 |
| B | LR 0,0003 | 0,609117 |
| C | Dropout 0,50 | 0,613860 |

Run sélectionné : **A**, `116957febf994489ac92baa3458f3d34`.
Test final : **accuracy 0,648438 ; macro-F1 0,644477**, sur 128 images.
Une seule graine et un petit jeu de validation limitent la portée de la comparaison.

## Installation et vérification

Depuis ce dossier, avec Git, uv et Python 3.12 :

```bash
uv sync --locked --python 3.12
uv run --locked python scripts/check_environment.py
uv run --locked pytest -q
uv run --locked dvc status
```

Sur cette machine, uv a été installé localement dans le dossier AI53. Si `uv` n'est
pas sur le PATH, exécuter d'abord, depuis `phase1_mlops/` :

```bash
export PATH="$PWD/../.tools/uv-env/bin:$PATH"
export UV_CACHE_DIR="$PWD/../.tools/uv-cache"
```

Si le cache système DVC n'est pas accessible, lui donner un dossier local :

```bash
mkdir -p .dvc/site-cache
uv run --locked dvc config --local core.site_cache_dir "$PWD/.dvc/site-cache"
```

Ce cache est ignoré par Git. Les données volumineuses sont dans le remote DVC
local voisin `../ai53-phase1-dvc-storage`. Sur cette machine ou après restauration
de la sauvegarde complète, les récupérer avec `uv run --locked dvc pull`.
Un clone GitHub seul ne contient ni ce remote, ni les données MLflow ; il contient
le code, les pointeurs, les rapports et l'historique. Pour de nouveaux entraînements
sans sauvegarde, `scripts/download_dataset.py` peut télécharger les archives vérifiées.

## Consulter MLflow et reprendre en Phase 2

Le serveur est arrêté après le lab pour sauvegarder ses fichiers proprement.
Pour le rouvrir, garder ce dossier comme répertoire de travail :

```bash
uv run --locked mlflow server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db
```

Ouvrir <http://127.0.0.1:5000>. Les trois Runs sont dans
`bean-image-classification`. Dans un second terminal :

```bash
uv run --locked python scripts/inspect_runs.py
```

Le modèle retenu est `runs:/116957febf994489ac92baa3458f3d34/bean-cnn`.
Son entrée est un tenseur float32 NCHW `(-1, 3, 96, 96)` : RGB, resize bilinéaire
96×96, conversion en tenseur, normalisation moyenne/écart-type `(0.5, 0.5, 0.5)`.
Il renvoie trois logits dans l'ordre `angular_leaf_spot`, `bean_rust`, `healthy`.
Le checkpoint local `models/model.pt` est celui de C ; le Run sélectionné fait autorité.

Conserver ensemble `mlflow.db`, `mlartifacts/`, ce dépôt et le remote DVC.
Arrêter le serveur avec Ctrl+C avant de copier la base et les artefacts.
Les résultats test sont déjà enregistrés : ils n'ont pas servi à choisir les paramètres.

## Pipeline

```text
data/raw ──> validate ──> prepare ──> train ──> modèles candidats MLflow
    └──────────────────────┘  └───> evaluate <── selection.run_id + modèle MLflow
```

`uv run --locked dvc repro train` construit un candidat sur train/validation.
`uv run --locked dvc repro evaluate` traite la branche du modèle sélectionné.
Les dépendances et paramètres détaillés sont dans [dvc.yaml](dvc.yaml).

La distribution originale reste dans `~/Downloads/AI53_Phase1_Lab`.
`docs/distribution/MANIFEST.sha256` s'applique à cette distribution d'origine,
et non à l'arborescence modifiée du projet. Attribution et licence des images :
[DATASET_SOURCE.md](DATASET_SOURCE.md).
