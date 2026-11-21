# Reprendre le projet en Phase 2

Le dossier AI53 contient une sauvegarde locale dans
`livrables/AI53_Phase1_Complet.zip`. Elle conserve :

- les sources, tests, rapports et l'historique Git du lab ;
- `mlflow.db` et `mlartifacts/`, copiés après arrêt du serveur ;
- le remote DVC voisin avec les données et checkpoints de tous les commits ;
- un manifeste SHA-256 des fichiers de la sauvegarde.

Les environnements Python, caches temporaires et archives ZIP de départ sont
recréables et ne sont pas inclus. Les données brutes/préparées et le checkpoint
de travail se restaurent à partir du remote DVC inclus, sans téléchargement des images.
La sauvegarde est locale ; elle ne fait pas partie du dépôt GitHub.

## Restauration dans un nouveau dossier

1. Extraire l'archive en conservant les deux dossiers voisins.
2. Installer Git, uv et Python 3.12 si nécessaire.
3. Depuis le sous-dossier `phase1_mlops/`, exécuter :

```bash
uv sync --locked --python 3.12
mkdir -p .dvc/site-cache
uv run --locked dvc config --local core.site_cache_dir "$PWD/.dvc/site-cache"
uv run --locked dvc remote modify --local lab-data url "$PWD/../ai53-phase1-dvc-storage"
uv run --locked dvc pull
uv run --locked dvc status
uv run --locked python scripts/check_environment.py
uv run --locked pytest -q
```

La configuration locale de la machine initiale est exclue de la sauvegarde pour
éviter de réutiliser un chemin absolu périmé. Le remote versionné reste relatif.
Le remplacement local par un chemin absolu évite aussi un problème de résolution
du remote relatif dans `dvc push --all-commits` avec cette version de DVC.

## Retrouver le modèle retenu

Depuis le même dossier :

```bash
uv run --locked mlflow server --host 127.0.0.1 --port 5000 --workers 1 --backend-store-uri sqlite:///mlflow.db
```

Ouvrir <http://127.0.0.1:5000>. Expérience : `bean-image-classification`.
Modèle retenu : `runs:/116957febf994489ac92baa3458f3d34/bean-cnn`.
Le checkpoint restauré par DVC est celui de C ; le modèle sélectionné est celui de A.
Les données MLflow contiennent les trois modèles et leurs preuves de validation.

`reports/test_metrics.json` et `reports/test_confusion_matrix.png` conservent
l'évaluation finale déjà réalisée. Il n'est pas nécessaire de la relancer pour
reprendre le modèle et préparer la Phase 2.

Le dépôt de remise est consultable à l'adresse
<https://github.com/SpeedTomy/AI53-Phase1-MLOps>.
La sauvegarde complète contient en plus le stockage DVC et les données historiques
MLflow, qui ne sont pas inclus dans le dépôt Git.
