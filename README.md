# EOSC 454 — Synthetic 3D Magnetic Inversion (Red Lake analogue)

This project demonstrates a full **SimPEG** workflow for EOSC 454 (Geophysical Inversion) using a **synthetic** geological analogue of the **Red Lake greenstone belt**, Ontario: a known 3D susceptibility model (mafic lens, felsic intrusion, low-susceptibility shear zone), **forward-modelled TMI** with noise, **L2 and sparse IRLS** inversion, and a simple **prospectivity** map that highlights the shear-style orogenic-gold target.

No GeologyOntario downloads or shapefiles are required.

## Geological sketch (synthetic)

- **Background greenstone:** χ = 0.01 SI  
- **Mafic/ultramafic lens** (2 × 5 × 1 km): χ = 0.05 SI  
- **Felsic intrusion** (~3 km diameter, ~0.5–2 km depth): χ = 0.0001 SI  
- **Shear / deformation corridor** (1 km wide, dip 70° north, E–W strike): χ = 0.0001 SI — orogenic gold analogue  

## Mesh and runtime

- **Domain:** 20 km × 20 km × 5 km depth, **200 m** core cells, **six** padding layers expanding by **1.5×** (~**4.6 × 10⁵** cells).  
- **Survey:** E–W lines, **200 m** line spacing, **200 m** along-line sampling, **60 m** flight height, σ = **2 nT** Gaussian noise.  
- Smooth inversion is sized to finish in **roughly a few to ~10 minutes** on a typical laptop (depends on CPU/RAM).

## Installation

1. `conda env create -f environment.yml`  
2. `conda activate eosc454-env`  
3. (Optional) `python -m ipykernel install --user --name eosc454-env --display-name "Python (eosc454-env)"`

## How to run

1. `notebooks/01_synthetic_model.ipynb` — build mesh and true model, forward + noisy TMI; writes `data/processed/chi_true.npy` and `data/processed/synthetic_survey.npz`.  
2. `notebooks/02_inversion.ipynb` — smooth (L2) inversion with misfit curve, sparse (IRLS) inversion, cross-section comparison vs truth; writes `chi_smooth_synthetic.npy` and `chi_sparse_synthetic.npy`. Uses a laptop-safe "lite" setup (coarser mesh + decimated stations + float32 RAM sensitivities).  
3. `notebooks/03_prospectivity.ipynb` — depth-integrated χ and prospectivity map; includes a **Great Bear**-style exploration rationale in markdown.

## Code layout

| Path | Role |
|------|------|
| `src/synthetic_model.py` | Mesh builder, `build_true_model`, plotting |
| `src/forward_sim.py` | Survey grid, `simulate_tmi` |
| `src/inversion_setup.py` | `build_simulation`, optional `build_mesh` / `build_survey` |
| `src/run_inversion.py` | `run_smooth_inversion`, `run_sparse_inversion` |
| `src/prospectivity.py` | Depth integration, prospectivity score, maps |

## References

- SimPEG tutorial: *Sparse Norm Inversion for TMI Data on a Tensor Mesh*  
  <https://docs.simpeg.xyz/content/tutorials/03-magnetics/plot_inv_1a_magnetics_induced.html>  
- Lelièvre, P. G., & Oldenburg, D. W. (2009). *A comprehensive study of including structural orientation information in geophysical inversions*. **Geophysics**.
