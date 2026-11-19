# Using local archives

Keep `train.zip`, `validation.zip` and `test.zip` in one folder. Together they take about 180 MB. See [DATASET_SOURCE.md](DATASET_SOURCE.md) for download sources and [archives.sha256](archives.sha256) for the expected checksums.

To extract the images without downloading the ZIPs again, run:

```bash
python download_dataset.py --archive-dir "PATH_TO_ARCHIVES"
```

Replace the quoted path with the folder holding the three ZIPs. The downloader checks each file before extraction. See [README.md](README.md) for the output location.
