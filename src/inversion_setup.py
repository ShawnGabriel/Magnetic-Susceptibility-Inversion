"""Set up SimPEG objects for 3D magnetic susceptibility inversion.

This module creates the three core ingredients needed before inversion:
1) an inversion mesh,
2) a magnetic survey object (data geometry + inducing field),
3) a forward simulation object that predicts data from a model.
"""

from __future__ import annotations

from pathlib import Path

import discretize
import numpy as np
from simpeg import maps
from simpeg.potential_fields import magnetics


def build_mesh(
    x: np.ndarray,
    y: np.ndarray,
    core_cell_size: float = 100.0,
    padding_cells: int = 8,
    padding_factor: float = 1.5,
) -> discretize.TensorMesh:
    """Build a 3D TensorMesh around the survey area.

    Parameters
    ----------
    x, y : np.ndarray
        1D survey coordinate vectors.
    core_cell_size : float, default=100.0
        Core cell width in meters (x, y, and z).
    padding_cells : int, default=8
        Number of expanding padding cells added on each side.
    padding_factor : float, default=1.5
        Geometric growth factor for padding cells.

    Returns
    -------
    discretize.TensorMesh
        3D tensor mesh centered on survey area with x/y/z padding.

    Notes
    -----
    The mesh uses:
    - a core region that covers the data footprint,
    - outward padding so boundary effects are reduced,
    - vertical extent that reaches below expected target depths.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x.size < 2 or y.size < 2:
        raise ValueError("x and y must each contain at least 2 points.")

    dx = float(core_cell_size)
    dy = float(core_cell_size)
    dz = float(core_cell_size)

    x_span = float(x.max() - x.min())
    y_span = float(y.max() - y.min())

    # Number of core cells is chosen to at least cover the survey span.
    n_core_x = max(4, int(np.ceil(x_span / dx)))
    n_core_y = max(4, int(np.ceil(y_span / dy)))
    n_core_z = max(12, int(np.ceil(max(x_span, y_span) / (2.0 * dz))))

    # TensorMesh cell-size tuples: (cell_width, n_cells, expansion_factor).
    # Negative factor means cells shrink toward the core from the outside.
    hx = [(dx, padding_cells, -padding_factor), (dx, n_core_x), (dx, padding_cells, padding_factor)]
    hy = [(dy, padding_cells, -padding_factor), (dy, n_core_y), (dy, padding_cells, padding_factor)]
    hz = [(dz, padding_cells, -padding_factor), (dz, n_core_z), (dz, padding_cells, padding_factor)]

    mesh = discretize.TensorMesh([hx, hy, hz], x0="CCC")

    x_center = 0.5 * (x.min() + x.max())
    y_center = 0.5 * (y.min() + y.max())

    # Shift mesh so its center aligns with the survey center in x/y.
    mesh.origin = np.r_[
        x_center - 0.5 * mesh.h[0].sum(),
        y_center - 0.5 * mesh.h[1].sum(),
        -mesh.h[2].sum(),  # top of mesh near z=0
    ]
    return mesh


def build_survey(obs_file_path: str | Path) -> magnetics.survey.Survey:
    """Build a SimPEG magnetic survey from exported observation arrays.

    Parameters
    ----------
    obs_file_path : str | Path
        Path to the `.npz` produced by `export_simpeg_obs`.

    Returns
    -------
    simpeg.potential_fields.magnetics.survey.Survey
        Survey object containing receiver locations and source field.

    Notes
    -----
    The inducing field is set for Red Lake:
    - inclination = 77 degrees
    - declination = 0 degrees
    - strength = 57,000 nT
    """
    obs_file_path = Path(obs_file_path)
    if not obs_file_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_file_path}")

    obs = np.load(obs_file_path)
    rx_locs = np.asarray(obs["receiver_locations"], dtype=float)
    if rx_locs.ndim != 2 or rx_locs.shape[1] != 3:
        raise ValueError("receiver_locations must have shape (N, 3).")

    # Receivers define where magnetic data are measured.
    receivers = magnetics.receivers.Point(rx_locs, components=["tmi"])

    # SourceField stores the Earth's inducing magnetic field parameters.
    source_field = magnetics.sources.UniformBackgroundField(
        receiver_list=[receivers],
        amplitude=57000.0,
        inclination=77.0,
        declination=0.0,
    )

    # Survey combines source + receivers into the measurement geometry.
    survey = magnetics.survey.Survey(source_field)
    return survey


def build_simulation(
    mesh: discretize.TensorMesh,
    survey: magnetics.survey.Survey,
    actind: np.ndarray,
) -> magnetics.simulation.Simulation3DIntegral:
    """Build a 3D integral magnetic forward simulation.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        Inversion mesh.
    survey : simpeg.potential_fields.magnetics.survey.Survey
        Magnetic survey object.
    actind : np.ndarray
        Boolean mask of active cells (True = cell is included in model).

    Returns
    -------
    simpeg.potential_fields.magnetics.simulation.Simulation3DIntegral
        Simulation object that predicts magnetic data from susceptibility.

    Notes
    -----
    `Simulation3DIntegral` computes TMI responses of each active cell and
    assembles them into the linear forward operator used during inversion.
    """
    actind = np.asarray(actind)
    if actind.dtype != bool:
        actind = actind.astype(bool)
    if actind.size != mesh.n_cells:
        raise ValueError("actind must have length mesh.n_cells.")

    # Maps inversion model (active cells only) -> full mesh susceptibility.
    model_map = maps.InjectActiveCells(mesh, actind, valInactive=0.0)

    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        chiMap=model_map,
        active_cells=actind,
        store_sensitivities="ram",
        engine="geoana",
    )
    return simulation
