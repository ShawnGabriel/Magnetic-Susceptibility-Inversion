# Data directory (not committed)

Large and regenerated files are listed in `.gitignore`:

- `raw/` — optional local geophysical archives
- `processed/` — outputs from notebooks (`synthetic_survey.npz`, `chi_*.npy`, sensitivity folders, etc.)

Run `notebooks/01_synthetic_model.ipynb` then `02_inversion.ipynb` to recreate `processed/` outputs.
