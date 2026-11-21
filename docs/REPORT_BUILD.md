# Reconstruire le rapport

Le contenu de référence est `LAB_REPORT.md`. Les commentaires `<!-- pagebreak -->`
délimitent les sept pages du PDF et restent invisibles à la lecture sur GitHub.
`scripts/build_report.py` met en page ce même contenu, avec tableaux, liens et figures.

Depuis la racine du dépôt, reconstruire la courbe à partir des historiques enregistrés :

```bash
uv run --locked python scripts/plot_validation.py
```

Le rendu PDF utilise des dépendances documentaires optionnelles, sans modification
du `pyproject.toml`, du `uv.lock` ni de l'environnement d'entraînement :

```bash
uv run --no-project --python 3.12 --with reportlab==5.1.0 --with markdown==3.11.3 python scripts/build_report.py
```

La sortie est `output/pdf/AI53_Phase1_Rapport.pdf`. Le script utilise Arial lorsqu'il
est disponible sur macOS, DejaVu Sans sur Linux, puis les polices PDF standard en
dernier recours. Les sept pages ont été rendues en PNG et inspectées avant remise.
Une reconstruction avec d'autres polices doit faire l'objet du même contrôle visuel.

Ces commandes utilisent les résultats déjà enregistrés. Elles ne lancent aucun
entraînement et ne recalculent pas les prédictions du jeu test.
