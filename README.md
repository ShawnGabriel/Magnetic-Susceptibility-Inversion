# EOSC 454 — 3D Magnetic Inversion and Gold Prospectivity (Red Lake)

This project builds an end-to-end geophysical workflow for EOSC 454 (Geophysical Inversion): prepare aeromagnetic data, run a 3D SimPEG magnetic susceptibility inversion over the Red Lake greenstone belt (Ontario), and convert the recovered susceptibility model into a gold prospectivity map focused on the Great Bear Project area.

## Datasets Used (GeologyOntario)

- **GDS1089** — *Ontario Airborne Geophysical Surveys, Magnetic and Electromagnetic Data, Grid and Profile Data (ASCII and Geosoft Formats) and Vector Data, Saganash Lake Area* (example Geosoft ASCII `.gxf` format source used to match this workflow's magnetic grid ingestion).
- **MRD126-REV1** — *1:250 000 Scale Bedrock Geology of Ontario* (provincial bedrock geology shapefile source for geology overlays).
- **P3227** — *Precambrian Geology, Red Lake* (regional Red Lake bedrock geology reference).

## Installation

1. Create the conda environment:
   - `conda env create -f environment.yml`
2. Activate the environment:
   - `conda activate eosc454-env`

## How to Run

1. Download raw magnetic and geology data from GeologyOntario.
2. Place raw files in `data/raw/` (e.g., `.gxf` magnetic grid and `.shp` geology files).
3. Run `notebooks/01_data_prep.ipynb` to:
   - load/clean the grid,
   - compute residual anomaly (TMI - IGRF),
   - export SimPEG-ready observations to `data/processed/`.
4. Run `notebooks/02_inversion.ipynb` to:
   - build mesh/survey/simulation,
   - run smooth and sparse susceptibility inversions,
   - save recovered models to `data/processed/`.
5. Run `notebooks/03_prospectivity.ipynb` to:
   - build depth-integrated susceptibility maps,
   - overlay bedrock contacts,
   - generate and interpret gold prospectivity scores.

## References

- SimPEG tutorial: *Sparse Norm Inversion for TMI Data on a Tensor Mesh*  
  <https://docs.simpeg.xyz/content/tutorials/03-magnetics/plot_inv_1a_magnetics_induced.html>
- Lelièvre, P. G., & Oldenburg, D. W. (2009). *A comprehensive study of including structural orientation information in geophysical inversions*. **Geophysics**.
