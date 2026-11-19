# iBean / Beans dataset downloader

From this folder, run the standalone script with Python 3.11 or 3.12:

```bash
python download_dataset.py
```

It writes the extracted images to `data/raw/`. See [DATASET_SOURCE.md](DATASET_SOURCE.md) for attribution and the license, and [archives.sha256](archives.sha256) for the expected checksums.

Use `--raw-dir "PATH_TO_PROJECT/data/raw"` to extract into your project. If you have the ZIPs already, see [Using local archives](OFFLINE_PRESTAGING.md). When following the lab, use the starter project's copy of the downloader in Milestone 2.
