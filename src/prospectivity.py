"""Prospectivity mapping from recovered magnetic susceptibility models (no external GIS required)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def depth_integrated_susceptibility(
    mesh,
    model: np.ndarray,
    actind: np.ndarray,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate susceptibility over depth and collapse to a 2D map.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        Inversion mesh.
    model : np.ndarray
        Susceptibility model on active cells.
    actind : np.ndarray
        Boolean active-cell mask.
    z_min, z_max : float
        Depth interval (m, positive down). Example: ``0, 500``.

    Returns
    -------
    susc_map : np.ndarray
        2D depth-integrated susceptibility map with shape ``(nCx, nCy)``.
    x : np.ndarray
        Mesh cell-center x coordinates.
    y : np.ndarray
        Mesh cell-center y coordinates.
    """
    actind = np.asarray(actind, dtype=bool)
    model = np.asarray(model, dtype=float).reshape(-1)
    if model.size != int(actind.sum()):
        raise ValueError("model size must match number of active cells.")

    model_full = np.zeros(mesh.n_cells, dtype=float)
    model_full[actind] = model
    model_3d = model_full.reshape(mesh.shape_cells, order="F")

    z_cc = np.asarray(mesh.cell_centers_z)
    hz = np.asarray(mesh.h[2])

    z_top = -abs(float(z_min))
    z_bottom = -abs(float(z_max))
    z_hi = max(z_top, z_bottom)
    z_lo = min(z_top, z_bottom)

    z_mask = (z_cc <= z_hi) & (z_cc >= z_lo)
    if not np.any(z_mask):
        raise ValueError("Requested depth window does not intersect mesh z coordinates.")

    model_sub = model_3d[:, :, z_mask]
    dz_sub = hz[z_mask]
    susc_map = np.sum(model_sub * dz_sub[np.newaxis, np.newaxis, :], axis=2)

    x = np.asarray(mesh.cell_centers_x)
    y = np.asarray(mesh.cell_centers_y)
    return susc_map, x, y


def prospectivity_score(
    susc_map: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    low_susc_threshold: float = 0.001,
) -> np.ndarray:
    """Score cells: low susceptibility plus high-gradient boundaries → higher prospectivity.

    Parameters
    ----------
    susc_map : np.ndarray
        2D susceptibility map.
    x, y : np.ndarray
        1D coordinate arrays corresponding to map axes.
    low_susc_threshold : float, default=0.001
        Threshold below which susceptibility is rewarded.

    Returns
    -------
    np.ndarray
        Prospectivity score map in range ``[0, 1]``.
    """
    susc_map = np.asarray(susc_map, dtype=float)
    if susc_map.shape != (len(x), len(y)):
        raise ValueError("susc_map shape must be (len(x), len(y)).")

    low_component = np.clip((low_susc_threshold - susc_map) / max(low_susc_threshold, 1e-12), 0.0, 1.0)

    dx = float(np.nanmedian(np.diff(np.asarray(x, dtype=float)))) if len(x) > 1 else 1.0
    dy = float(np.nanmedian(np.diff(np.asarray(y, dtype=float)))) if len(y) > 1 else 1.0
    gx, gy = np.gradient(susc_map, dx, dy, edge_order=1)
    grad_mag = np.sqrt(gx**2 + gy**2)

    gmin = float(np.nanmin(grad_mag))
    gmax = float(np.nanmax(grad_mag))
    if np.isclose(gmax, gmin):
        grad_component = np.zeros_like(grad_mag)
    else:
        grad_component = (grad_mag - gmin) / (gmax - gmin)

    score = 0.6 * low_component + 0.4 * grad_component
    score = np.clip(score, 0.0, 1.0)
    return score


def plot_prospectivity(
    score_map: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    deposit_points: list[tuple[float, float]] | None = None,
    title: str = "",
) -> None:
    """Plot a prospectivity map and optional validation points.

    Parameters
    ----------
    score_map : np.ndarray
        2D prospectivity scores in ``[0, 1]`` with shape ``(len(x), len(y))``.
    x, y : np.ndarray
        1D map coordinates.
    deposit_points : list[tuple[float, float]] | None
        Optional points as ``(x, y)`` in the same units as the map.
    title : str, default=""
        Figure title.
    """
    score_map = np.asarray(score_map, dtype=float)
    if score_map.shape != (len(x), len(y)):
        raise ValueError("score_map shape must be (len(x), len(y)).")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        score_map.T,
        origin="lower",
        extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Prospectivity score (0–1)")

    if deposit_points:
        pts = np.asarray(deposit_points, dtype=float)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            marker="*",
            s=220,
            c="cyan",
            edgecolors="black",
            linewidths=0.8,
            label="Validation points",
        )
        ax.legend(loc="best")

    ax.set_title(title if title else "Prospectivity map")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.show()
