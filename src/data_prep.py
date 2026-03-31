"""Utilities for loading and preparing aeromagnetic grid data.

This module is intentionally written with standard Python + NumPy/Pandas tools,
so it does not depend on Geosoft desktop software.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_gxf(filepath: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a Geosoft ASCII GXF grid into arrays and metadata.

    Parameters
    ----------
    filepath : str | Path
        Path to a ``.gxf`` grid file.

    Returns
    -------
    data : np.ndarray
        2D grid of values with shape ``(n_rows, n_cols)``.
    x : np.ndarray
        1D x-coordinate array (length ``n_cols``).
    y : np.ndarray
        1D y-coordinate array (length ``n_rows``).
    metadata : dict[str, Any]
        Parsed metadata dictionary containing keys such as
        ``n_rows``, ``n_cols``, ``x_origin``, ``y_origin``,
        ``x_cell_size``, ``y_cell_size``, and ``nodata`` when present.

    Notes
    -----
    GXF stores data as ASCII with header directives beginning with ``#``.
    This parser reads common directives used by GeologyOntario exports.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GXF file not found: {filepath}")

    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()

    metadata: dict[str, Any] = {}
    data_start_idx: int | None = None

    key_map = {
        "POINTS": ("n_cols", int),
        "ROWS": ("n_rows", int),
        "PTSEPARATION": ("x_cell_size", float),
        "RWSEPARATION": ("y_cell_size", float),
        "XORIGIN": ("x_origin", float),
        "YORIGIN": ("y_origin", float),
        "DUMMY": ("nodata", float),
    }

    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue

        if raw.upper().startswith("#GRID"):
            data_start_idx = i + 1
            break

        if raw.startswith("#"):
            tag = raw[1:].strip().upper()
            if tag in key_map and i + 1 < len(lines):
                out_key, caster = key_map[tag]
                value_line = lines[i + 1].strip()
                try:
                    metadata[out_key] = caster(value_line)
                except ValueError:
                    # Keep parser robust: store the raw text if casting fails.
                    metadata[out_key] = value_line
                i += 2
                continue

        i += 1

    if data_start_idx is None:
        raise ValueError(f"Could not find #GRID section in {filepath}")

    if "n_rows" not in metadata or "n_cols" not in metadata:
        raise ValueError("Missing required GXF dimensions (#ROWS/#POINTS).")

    n_rows = int(metadata["n_rows"])
    n_cols = int(metadata["n_cols"])

    # Read all numeric tokens after #GRID, then reshape to rows x cols.
    token_lines = [ln.strip() for ln in lines[data_start_idx:] if ln.strip()]
    token_lines = [ln for ln in token_lines if not ln.startswith("#")]
    if not token_lines:
        raise ValueError("No grid values found after #GRID.")

    token_series = pd.Series(" ".join(token_lines).split())
    values = pd.to_numeric(token_series, errors="coerce").to_numpy(dtype=float)

    expected = n_rows * n_cols
    if values.size < expected:
        raise ValueError(
            f"GXF has {values.size} values, expected at least {expected} "
            f"({n_rows} rows x {n_cols} cols)."
        )
    if values.size > expected:
        values = values[:expected]

    data = values.reshape((n_rows, n_cols))

    nodata = metadata.get("nodata", None)
    if nodata is not None and np.isfinite(float(nodata)):
        data = data.astype(float, copy=False)
        data[data == float(nodata)] = np.nan

    x0 = float(metadata.get("x_origin", 0.0))
    y0 = float(metadata.get("y_origin", 0.0))
    dx = float(metadata.get("x_cell_size", 1.0))
    dy = float(metadata.get("y_cell_size", 1.0))

    x = x0 + np.arange(n_cols) * dx
    y = y0 + np.arange(n_rows) * dy

    metadata.setdefault("x_origin", x0)
    metadata.setdefault("y_origin", y0)
    metadata.setdefault("x_cell_size", dx)
    metadata.setdefault("y_cell_size", dy)
    metadata.setdefault("nodata", nodata)

    return data, x, y, metadata


def clip_to_study_area(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip a gridded dataset to a bounding box.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays for columns (x) and rows (y).
    data : np.ndarray
        2D grid with shape ``(len(y), len(x))``.
    lon_min, lon_max, lat_min, lat_max : float
        Bounding box limits. These names reflect typical geographic use,
        but the function works for any projected coordinates as long as
        x/y and bounds are in the same CRS.

    Returns
    -------
    x_clip, y_clip, data_clip : tuple[np.ndarray, np.ndarray, np.ndarray]
        Coordinate vectors and clipped 2D grid.
    """
    if data.shape != (len(y), len(x)):
        raise ValueError("Data shape must be (len(y), len(x)).")

    x_mask = (x >= lon_min) & (x <= lon_max)
    y_mask = (y >= lat_min) & (y <= lat_max)

    if not np.any(x_mask) or not np.any(y_mask):
        raise ValueError("No grid cells fall inside the requested bounding box.")

    x_clip = x[x_mask]
    y_clip = y[y_mask]
    data_clip = data[np.ix_(y_mask, x_mask)]
    return x_clip, y_clip, data_clip


def igrf_subtract(tmi_grid: np.ndarray, igrf_value: float = 57000.0) -> np.ndarray:
    """Compute residual magnetic anomaly by removing regional IGRF.

    Parameters
    ----------
    tmi_grid : np.ndarray
        Total Magnetic Intensity (TMI) grid in nT.
    igrf_value : float, default=57000.0
        Regional IGRF field value in nT.

    Returns
    -------
    np.ndarray
        Residual anomaly grid in nT.
    """
    return np.asarray(tmi_grid, dtype=float) - float(igrf_value)


def export_simpeg_obs(
    x: np.ndarray,
    y: np.ndarray,
    tmi_values: np.ndarray,
    flight_height: float = 60.0,
    uncertainty_pct: float = 0.02,
    out_path: str | Path | None = None,
) -> Path:
    """Export gridded magnetic data to a compact SimPEG-ready ``.npz`` file.

    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate vectors for grid columns and rows.
    tmi_values : np.ndarray
        2D magnetic values with shape ``(len(y), len(x))``.
    flight_height : float, default=60.0
        Observation elevation above ground (m). Used as receiver z.
    uncertainty_pct : float, default=0.02
        Relative uncertainty factor. Data uncertainties are computed as
        ``max(uncertainty_pct * peak_amplitude, 5 nT)`` where
        ``peak_amplitude = max(abs(tmi_values))``.
    out_path : str | Path
        Output ``.npz`` path.

    Returns
    -------
    Path
        Path to the written observation file.
    """
    if out_path is None:
        raise ValueError("out_path is required.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(tmi_values, dtype=float)
    if arr.shape != (len(y), len(x)):
        raise ValueError("tmi_values shape must be (len(y), len(x)).")

    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    valid = np.isfinite(arr)

    receiver_locs = np.c_[xx[valid], yy[valid], np.full(valid.sum(), float(flight_height))]
    data = arr[valid]

    peak_amplitude = float(np.nanmax(np.abs(data))) if data.size else 0.0
    floor = 5.0
    sigma = max(float(uncertainty_pct) * peak_amplitude, floor)
    uncertainties = np.full(data.shape, sigma, dtype=float)

    np.savez(
        out_path,
        receiver_locations=receiver_locs,
        data=data,
        uncertainties=uncertainties,
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        flight_height=float(flight_height),
        peak_amplitude=peak_amplitude,
        uncertainty_pct=float(uncertainty_pct),
        uncertainty_floor_nT=floor,
    )
    return out_path
