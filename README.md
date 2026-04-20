# EOSC 454 — Synthetic 3D Magnetic Inversion (Red Lake analogue)

This project demonstrates a full **SimPEG** workflow for EOSC 454 (Geophysical Inversion) using a **synthetic** geological analogue of the **Red Lake greenstone belt**, Ontario: a known 3D susceptibility model (mafic lens, felsic intrusion, low-susceptibility shear zone), **forward-modelled TMI** with noise, **ℓ² (Tikhonov) and sparse ℓᵖ (IRLS)** inversion, and a simple **prospectivity** map that highlights the shear-style orogenic-gold target.

No GeologyOntario downloads or shapefiles are required.

## Geological sketch (synthetic)

- **Background greenstone:** χ = 0.01 SI  
- **Mafic/ultramafic lens** (2 × 5 × 1 km): χ = 0.05 SI  
- **Felsic intrusion** (~3 km diameter, ~0.5–2 km depth): χ = 0.0001 SI  
- **Shear / deformation corridor** (1 km wide, dip 70° north, E–W strike): χ = 0.0001 SI — orogenic gold analogue  

## Domain, meshes, and survey

- **Domain:** 20 km × 20 km × 5 km depth, **six** padding layers expanding by **1.5×** (same extent in all notebooks).  
- **Forward mesh** (`01_synthetic_model.ipynb`): **200 m** core cells; total cells **~3.9 × 10⁵** (tensor mesh used for the true model and TMI simulation).  
- **Inversion mesh** (`02_inversion.ipynb`): **350 m** core cells; **~9.5 × 10⁴** active cells — coarser than forward to fit memory and runtime on a typical laptop; this is the main limit on recovered spatial resolution.  
- **Survey:** E–W flight lines, **200 m** line spacing, **200 m** along-line sampling, **60 m** receiver height, inducing field as set in the notebooks, σ = **2 nT** Gaussian noise. Full grid: **10 201** stations.  
- **Decimation:** `station_stride = 2` in `02_inversion.ipynb` keeps every other station → **N = 5101** data; the χ² target misfit uses **ϕ<sub>d</sub> ≈ N** with `chifact = 1.0`.  
- **Sensitivities:** stored on disk (`store_sensitivities="disk"`, float32) under `data/processed/sensitivities_inversion_02/` (configurable in `02_inversion.ipynb`).  
- **Map-view data fit** (three-panel observed / predicted / normalised residual): stations after decimation are **not** a full rectangular grid, so maps use **`scipy.interpolate.griddata`** onto a regular mesh (not a simple `reshape`). Normalised residuals are plotted on a **±3σ** colour scale unless you change `vmin`/`vmax` in the notebook.  
- **Runtime:** smooth inversion is typically **on the order of several to ~10+ minutes** per machine (CPU, RAM, CG iterations, disk I/O).

## Installation

1. `conda env create -f environment.yml`  
2. `conda activate eosc454-env`  
3. (Optional) `python -m ipykernel install --user --name eosc454-env --display-name "Python (eosc454-env)"`

## How to run

1. `notebooks/01_synthetic_model.ipynb` — build forward mesh and true model (`build_true_model` in `src/synthetic_model.py`), forward + noisy TMI; writes `data/processed/chi_true.npy` and `data/processed/synthetic_survey.npz`.  
2. `notebooks/02_inversion.ipynb` — smooth inversion (misfit tracking, Tikhonov / L-curve style plots, three-panel data fit), then **sparse IRLS** initialised from the smooth model; writes `data/processed/chi_smooth_synthetic.npy` and `data/processed/chi_sparse_synthetic.npy`. Uses the coarser inversion mesh, decimated stations, and disk sensitivities as above.  
3. `notebooks/03_prospectivity.ipynb` — depth-integrated χ (0–3 km window in code) and prospectivity map; loads recovered χ in order: **`chi_sparse_synthetic.npy`** → **`susceptibility_sparse.npy`** → **`chi_smooth_synthetic.npy`**. Includes **Great Bear**–style exploration context in markdown.

## Code layout

| Path | Role |
|------|------|
| `src/synthetic_model.py` | Mesh builder, `build_true_model`, plotting |
| `src/forward_sim.py` | Survey grid, `simulate_tmi` |
| `src/inversion_setup.py` | `build_simulation`, optional `build_mesh` / `build_survey` |
| `src/run_inversion.py` | `run_smooth_inversion`, `run_sparse_inversion` |
| `src/prospectivity.py` | Depth integration, `prospectivity_score`, maps |

## References

- SimPEG tutorial: *Sparse Norm Inversion for TMI Data on a Tensor Mesh*  
  <https://docs.simpeg.xyz/content/tutorials/03-magnetics/plot_inv_1a_magnetics_induced.html>  
- Lelièvre, P. G., & Oldenburg, D. W. (2009). *A comprehensive study of including structural orientation information in geophysical inversions*. **Geophysics**.
