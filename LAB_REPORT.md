# AI53 — Lab MLOps Phase 1

Travail réalisé le 19 novembre 2026 à partir du starter officiel fourni.
Les mesures sont ajoutées après exécution ; les prédictions sont fixées avant entraînement.

## 0 — Audit du starter

Les fichiers de la distribution et les trois archives ont tous passé la vérification
`shasum -a 256 -c MANIFEST.sha256` avant leur copie. Le starter original est conservé
dans le dossier de téléchargement. Le CNN, les scripts de préparation/évaluation
et les tests du cours sont conservés.

| Fichier / réglage | Problème constaté avant modification | Correction et vérification |
|---|---|---|
| `src/reproducibility.py` | Seul `random` est initialisé ; NumPy, Torch, le générateur et les workers restent incomplets. | Initialiser toutes les sources ; tests de reproductibilité. |
| `src/validate_data.py` | `validate_dataset` lève `NotImplementedError` : aucune barrière qualité effective. | Vérifier structure, classes, effectifs et décodage complet ; tests et exécution sur les 1 295 images. |
| `src/provenance.py` | `build_run_manifest` n'est pas implémenté. | Capturer Git, identité des données, versions, configuration et Run ID ; test et inspection des vrais Runs. |
| `src/train.py` | Les blocs de journalisation MLflow ne sont pas implémentés. | Enregistrer paramètres, Dataset, séries de métriques, artefacts et modèle avec contrat d'entrée/sortie. |
| `dvc.yaml` absent | Pas de dépendances ni d'invalidation automatique des étapes. | Construire quatre étapes et vérifier le DAG puis les recalculs observés. |
| `data/raw.dvc`, `.dvc/` absents | Les archives existent mais les images ne sont pas encore versionnées et sauvegardées par DVC. | Vérifier SHA-256, extraire, `dvc add`, configurer le remote local et `dvc push`. |
| Environnement du dossier AI53 | Le Python courant est 3.14 alors que le starter exige 3.11/3.12. | Utiliser Python 3.12 et une `.venv` propre au lab avec `uv sync --locked`. Ne pas modifier l'environnement des anciens TP. |
| Historique / sélection | Aucun historique Git ni Run de référence ; `selection.run_id` est vide. | Commits par jalon, comparaison sur validation puis gel du Run ID avant un seul test final. |
| Conservation MLflow | `.gitignore` exclut correctement la base et les artefacts, mais Git seul ne les sauvegarde pas. | Conserver la base et les artefacts, arrêter le serveur avant une sauvegarde cohérente pour Phase 2. |

**Checkpoint 1.** Un `model.pt` utilisable ne suffit pas : il ne reconstitue ni le
code exact, ni les données, les versions, les graines et les réglages, ni les scores
ayant motivé la sélection. Il faut pouvoir relier ces éléments au même entraînement.

## Mise en place et périmètre

Le lab a son propre dépôt local dans `phase1_mlops/`. Les commits automatiques sont
attribués à l'identité locale `AI53 Lab <ai53-lab@localhost>` ; aucune identité
personnelle n'est inventée. La publication GitHub reste à vérifier séparément.
Les images, caches, environnements et données MLflow ne doivent pas entrer dans Git.

## 1 — Reproductibilité

`pyproject.toml` décrit les exigences et les plages de versions ; `uv.lock` fige les
versions résolues et les sources ; `.venv/` contient leur installation locale et se
recrée à partir du verrou. Le verrou fourni est conservé.

**Checkpoint 2.** `random_state=42` contrôle les tirages dans un environnement donné.
Il ne garantit pas une identité bit à bit entre architectures CPU/GPU, versions de
bibliothèques ou algorithmes numériques différents. Les réglages cuDNN limitent
le non-déterminisme sans constituer une garantie universelle.

## 2 — Versionnement des données

**Checkpoint 3.** Le remote Git stocke le code, la configuration, l'historique et les
pointeurs DVC. Le remote DVC stocke les octets des données et sorties mises en cache.
`git push` ne transfère pas les images gérées par DVC ; il faut aussi `dvc push`.

## 3 — Barrière qualité

À compléter avec les résultats d'exécution du validateur.

## 4 — Pipeline DVC

`prepare` dépend du rapport qualité **et** des images brutes : le rapport autorise
la suite, mais ses seuls effectifs ne représentent pas le contenu des images.
Les dépendances des étapes incluent leurs imports locaux et les deux fichiers
d'environnement. Les paramètres sont déclarés individuellement.

`train` dépend uniquement des images et manifestes train/validation. `evaluate`
dépend du test préparé et des manifestes train/validation nécessaires à la vérification
d'identité, ainsi que de `selection.run_id`. Il ne dépend ni du checkpoint local ni
des rapports TRAIN : le modèle sélectionné provient de MLflow.

Les fichiers JSON de scores sont déclarés comme métriques DVC. Les rapports sont
`cache: false` pour être conservés dans Git ; images, manifestes et checkpoint sont
dans le cache DVC.

## 5 — Provenance et MLflow

**Checkpoint 4.** Git enregistre le code et la configuration sauvegardés ; la
provenance note le commit et les modifications locales. DVC enregistre l'identité
des données et la relation entre entrées, paramètres et sorties. MLflow enregistre
chaque entraînement avec ses paramètres, métriques, Dataset, artefacts et modèle.

## 6 — Prédictions fixées avant entraînement

Hypothèse du guide : toutes les sorties sont à jour, test compris ; Run sélectionné fixe.

| Changement isolé | VALIDATE | PREPARE | TRAIN | EVALUATE |
|---|---|---|---|---|
| `train.learning_rate` | REUSED | REUSED | RERUN | REUSED |
| `data.image_size` | REUSED | RERUN | RERUN | NEEDS UPDATE |
| Commentaire dans `src/validate_data.py`, rapport inchangé | RERUN | REUSED | REUSED | REUSED |
| `README.md` uniquement | REUSED | REUSED | REUSED | REUSED |

**Checkpoint 5.** Le taux d'apprentissage est une entrée de TRAIN uniquement : ni
le contrôle des images, ni leur préparation, ni le Run déjà sélectionné n'en dépendent.
Un commentaire dans le validateur change son empreinte et relance VALIDATE ; si
le JSON qualité reste identique, les entrées de PREPARE restent identiques, puis celles
des deux branches aussi. Une nouvelle taille change PREPARE et les tenseurs/manifeste
des deux branches ; l'ancien Run doit être revalidé et peut être rejeté.

**Règle de sélection fixée à l'avance :** maximiser la macro-F1 de validation à la
dernière époque ; en cas d'égalité, maximiser l'accuracy de validation ; puis minimiser
la loss de validation ; puis préférer A, B, C dans cet ordre. Le temps est descriptif,
car il dépend de la charge de la machine.

**Prédiction B :** passer de 0,001 à 0,0003 réduit les pas d'Adam. Sur quatre époques,
la convergence peut être plus lente et le score final plus faible ; des courbes plus
stables restent possibles. Ce n'est pas une garantie.

**Protocole C :** revenir au taux de A et passer seulement le dropout de 0,30 à 0,50.
Une régularisation plus forte peut limiter le surapprentissage, mais peut aussi
ralentir l'apprentissage de ce petit réseau sur quatre époques. Graine, données,
batch, augmentation et nombre d'époques sont identiques à A.

## 7 — Comparaison contrôlée

À compléter à partir des trois Runs MLflow terminés, sans utiliser les scores test.

**Checkpoint 6.** La macro-F1 donne le même poids aux trois classes et combine
précision et rappel ; elle permet d'éviter qu'une classe moins bien reconnue soit
masquée par un bon score global. Le test sert une seule fois après la sélection,
afin de ne pas adapter le choix du modèle aux exemples censés estimer sa généralisation.
Une seule graine et 133 images de validation ne suffisent pas à démontrer un avantage
général d'un hyperparamètre.

## 8 — Sélection et test final

À compléter après comparaison de validation et gel du Run ID.

## État de remise

À compléter après vérifications finales. Le rendu du guide est l'URL du dépôt GitHub
et ce rapport ; un dépôt local seul ne satisfait pas la partie publication.
