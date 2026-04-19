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


def _geometric_padding_widths(d_core: float, n_pad: int, factor: float) -> np.ndarray:
    """``n_pad`` widths: first cell touching core is ``d_core * factor``, then ×factor."""
    w0 = float(d_core) * float(factor)
    return np.array([w0 * (factor**i) for i in range(int(n_pad))], dtype=float)


def build_mesh(
    x: np.ndarray,
    y: np.ndarray,
    core_cell_size_xy: float = 500.0,
    core_cell_size_z: float = 250.0,
    depth_core_m: float = 5000.0,
    padding_cells: int = 6,
    padding_factor: float = 1.5,
    max_total_cells: int = 2_000_000,
) -> discretize.TensorMesh:
    """TensorMesh: 500 m x/y core, 250 m z core, geometric padding, survey-aligned extent."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x.size < 2 or y.size < 2:
        raise ValueError("x and y must each contain at least 2 points.")

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    dx0 = float(core_cell_size_xy)
    dy0 = float(core_cell_size_xy)
    dz_core = float(core_cell_size_z)
    n_pad = int(padding_cells)
    if n_pad < 1:
        raise ValueError("padding_cells must be >= 1.")

    n_core_z = max(1, int(np.ceil(float(depth_core_m) / dz_core)))
    pad_z = _geometric_padding_widths(dz_core, n_pad, padding_factor)
    hz = np.concatenate([pad_z[::-1], np.full(n_core_z, dz_core, dtype=float)])

    n_z_total = int(hz.size)
    max_xy_block = max(1, int(max_total_cells // n_z_total))

    dx, dy = dx0, dy0
    n_core_x = max(1, int(np.ceil(x_span / dx)))
    n_core_y = max(1, int(np.ceil(y_span / dy)))
    while (n_core_x + 2 * n_pad) * (n_core_y + 2 * n_pad) > max_xy_block:
        scale = np.sqrt(
            ((n_core_x + 2 * n_pad) * (n_core_y + 2 * n_pad)) / float(max_xy_block)
        )
        dx *= scale * 1.01
        dy *= scale * 1.01
        n_core_x = max(1, int(np.ceil(x_span / dx)))
        n_core_y = max(1, int(np.ceil(y_span / dy)))

    if dx > dx0 * 1.001 or dy > dy0 * 1.001:
        print(
            f"Note: horizontal core cell size increased to dx={dx:.1f} m, dy={dy:.1f} m "
            f"to keep total cells <= {max_total_cells}."
        )

    pad_x = _geometric_padding_widths(dx, n_pad, padding_factor)
    pad_y = _geometric_padding_widths(dy, n_pad, padding_factor)
    hx = np.concatenate([pad_x[::-1], np.full(n_core_x, dx, dtype=float), pad_x])
    hy = np.concatenate([pad_y[::-1], np.full(n_core_y, dy, dtype=float), pad_y])

    mesh = discretize.TensorMesh([hx, hy, hz], x0=np.zeros(3))

    core_width_x = n_core_x * dx
    core_width_y = n_core_y * dy
    x0_mesh = x_center - 0.5 * core_width_x - float(np.sum(pad_x))
    y0_mesh = y_center - 0.5 * core_width_y - float(np.sum(pad_y))
    z0_mesh = -float(np.sum(hz))
    mesh.origin = np.array([x0_mesh, y0_mesh, z0_mesh], dtype=float)

    hx_arr = np.asarray(mesh.h[0], dtype=float)
    hy_arr = np.asarray(mesh.h[1], dtype=float)
    hz_arr = np.asarray(mesh.h[2], dtype=float)
    hx_min, hy_min, hz_min = float(hx_arr.min()), float(hy_arr.min()), float(hz_arr.min())
    extent_x = float(np.sum(hx_arr))
    extent_y = float(np.sum(hy_arr))
    extent_z = float(np.sum(hz_arr))

    print(
        "Mesh shape (nCx, nCy, nCz):",
        tuple(int(s) for s in mesh.shape_cells),
        "| total cells:",
        mesh.n_cells,
    )
    print(
        f"Smallest cell widths — hx: {hx_min:.2f} m, hy: {hy_min:.2f} m, hz: {hz_min:.2f} m "
        f"(global min: {min(hx_min, hy_min, hz_min):.2f} m)"
    )
    print(
        f"Mesh extent — x: {extent_x:.1f} m, y: {extent_y:.1f} m, z: {extent_z:.1f} m "
        f"(survey span x: {x_span:.1f} m, y: {y_span:.1f} m)"
    )
    return mesh


def build_magnetic_survey(receiver_locations: np.ndarray) -> magnetics.survey.Survey:
    """Tutorial-style survey: ``Point`` receivers + ``UniformBackgroundField`` + ``Survey``.

    Inducing field: 57,000 nT, I=77°, D=0° (northern Canada / Red Lake style).
    """
    rx = np.asarray(receiver_locations, dtype=float)
    if rx.ndim != 2 or rx.shape[1] != 3:
        raise ValueError("receiver_locations must have shape (N, 3).")

    receivers = magnetics.receivers.Point(rx, components=["tmi"])
    source_field = magnetics.sources.UniformBackgroundField(
        receiver_list=[receivers],
        amplitude=57000.0,
        inclination=77.0,
        declination=0.0,
    )
    return magnetics.survey.Survey(source_field)


def build_survey(obs_file_path: str | Path) -> magnetics.survey.Survey:
    """Load ``receiver_locations`` from a ``.npz`` (e.g. ``export_simpeg_obs``) and build the survey."""
    obs_file_path = Path(obs_file_path)
    if not obs_file_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_file_path}")

    obs = np.load(obs_file_path)
    rx_locs = np.asarray(obs["receiver_locations"], dtype=float)
    return build_magnetic_survey(rx_locs)


def build_simulation(
    mesh: discretize.TensorMesh,
    survey: magnetics.survey.Survey,
    actind: np.ndarray | None = None,
    store_sensitivities: str = "ram",
    sensitivity_path: str | Path | None = None,
    sensitivity_dtype: np.dtype | str | None = None,
) -> magnetics.simulation.Simulation3DIntegral:
    """``Simulation3DIntegral`` with ``IdentityMap`` when all cells are active (tutorial default).

    Parameters
    ----------
    actind : np.ndarray | None
        Active cells; default all ``True``.
    store_sensitivities : str, default ``"ram"``
        Use ``"forward_only"`` for forward-only ``dpred`` (faster, no stored sensitivities).
        Use ``"ram"`` or ``"disk"`` for inversion. Large meshes should use ``"disk"`` to
        avoid exhausting RAM.
    sensitivity_path : str | Path | None
        Directory for sensitivity blocks when ``store_sensitivities="disk"``. SimPEG default
        is ``./sensitivities``; pass an absolute path if the working directory may vary.
    sensitivity_dtype : np.dtype | str | None
        Optional sensitivity dtype passed to SimPEG (e.g. ``np.float32``) to reduce disk usage.
    """
    if actind is None:
        actind = np.ones(mesh.n_cells, dtype=bool)
    actind = np.asarray(actind, dtype=bool)
    if actind.size != mesh.n_cells:
        raise ValueError("actind must have length mesh.n_cells.")

    if bool(np.all(actind)):
        chi_map = maps.IdentityMap(mesh)
    else:
        chi_map = maps.InjectActiveCells(mesh, actind, value_inactive=0.0)

    kwargs: dict = dict(
        survey=survey,
        chiMap=chi_map,
        active_cells=actind,
        store_sensitivities=store_sensitivities,
        engine="geoana",
    )
    if store_sensitivities == "disk" and sensitivity_path is not None:
        kwargs["sensitivity_path"] = str(Path(sensitivity_path))
    if sensitivity_dtype is not None:
        kwargs["sensitivity_dtype"] = sensitivity_dtype

    return magnetics.simulation.Simulation3DIntegral(mesh, **kwargs)
