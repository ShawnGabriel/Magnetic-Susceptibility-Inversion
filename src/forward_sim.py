"""Forward modelling of TMI for synthetic magnetic susceptibility models."""

from __future__ import annotations

import numpy as np
from simpeg import data as simpeg_data
from simpeg.potential_fields import magnetics

from inversion_setup import build_magnetic_survey, build_simulation


def simulate_tmi(
    mesh,
    model: np.ndarray,
    survey: magnetics.survey.Survey,
    actind: np.ndarray | None = None,
    noise_std_nt: float = 2.0,
    random_seed: int | None = 42,
    simulation=None,
) -> simpeg_data.Data:
    """Run ``Simulation3DIntegral``, add Gaussian noise, return a ``Data`` object.

    If ``simulation`` is omitted, builds one with ``build_simulation`` using
    ``store_sensitivities="forward_only"`` (all cells active by default). Pass an
    existing ``simulation`` to reuse the operator.

    Parameters
    ----------
    mesh
        Discretize mesh.
    model : np.ndarray
        Susceptibility on all mesh cells, shape ``(mesh.n_cells,)``.
    survey : magnetics.survey.Survey
        Magnetic survey built from receiver locations.
    actind : np.ndarray | None
        Optional boolean mask of active cells; used only if ``simulation`` is
        ``None``; default all ``True``.
    noise_std_nt : float, default=2.0
        Standard deviation of Gaussian noise (nT).
    random_seed : int | None, default=42
        Seed for reproducible noise; ``None`` for non-deterministic draws.
    simulation : Simulation3DIntegral | None
        Optional pre-built forward simulation.

    Returns
    -------
    simpeg.data.Data
        Synthetic observations with ``standard_deviation=noise_std_nt``.
    """
    if simulation is None:
        if actind is None:
            actind = np.ones(mesh.n_cells, dtype=bool)
        simulation = build_simulation(
            mesh, survey, actind, store_sensitivities="forward_only"
        )

    model = np.asarray(model, dtype=float).reshape(-1)
    n_p = int(simulation.chiMap.nP)
    if model.size == mesh.n_cells and n_p == mesh.n_cells:
        m_vec = model
    elif model.size == mesh.n_cells:
        actind_b = np.asarray(simulation.active_cells, dtype=bool)
        m_vec = model[actind_b]
    elif model.size == n_p:
        m_vec = model
    else:
        raise ValueError(
            f"model length {model.size} must be mesh.n_cells ({mesh.n_cells}) or nP ({n_p})."
        )

    d_clean = simulation.dpred(m_vec)
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0.0, float(noise_std_nt), size=d_clean.shape)
    dobs = d_clean + noise
    std = np.full(dobs.shape, float(noise_std_nt), dtype=float)
    return simpeg_data.Data(survey=survey, dobs=dobs, standard_deviation=std)


def build_receiver_grid_ew_lines(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    line_spacing_m: float = 200.0,
    along_line_spacing_m: float = 200.0,
    flight_height_m: float = 60.0,
) -> np.ndarray:
    """E–W flight lines (constant y), spaced ``line_spacing_m`` in y; samples every ``along_line_spacing_m`` in x.

    Parameters
    ----------
    x0, x1, y0, y1 : float
        Survey rectangle corners (m).
    line_spacing_m : float
        Distance between flight lines (northing).
    along_line_spacing_m : float
        Sample spacing along each line (easting).
    flight_height_m : float
        Receiver z (m, positive up).

    Returns
    -------
    np.ndarray
        Receiver locations, shape (N, 3).
    """
    ys = np.arange(y0, y1 + 1e-9, line_spacing_m)
    xs = np.arange(x0, x1 + 1e-9, along_line_spacing_m)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, float(flight_height_m), dtype=float)
    locs = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    return locs.astype(float)
