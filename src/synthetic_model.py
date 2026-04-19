"""Synthetic geological model (Red Lake greenstone belt analogue) on a 3D mesh."""

from __future__ import annotations

import warnings

import discretize
import matplotlib.pyplot as plt
import numpy as np
from simpeg.utils import BreakingChangeWarning

warnings.filterwarnings("ignore", category=BreakingChangeWarning)

# --- Geological body parameters (metres, z positive up) -------------------------
# Anchored to the mesh "core" region: x,y in [0, 20000] m and z in [0, -3000] m.
CHI_BG = 0.01
CHI_MAFIC = 0.05
CHI_FELSIC = 1.0e-4
CHI_SHEAR = 1.0e-4

# Mafic lens: centred at (10 km, 10 km, -1 km), size 4 km (EW) × 2 km (NS) × 1.5 km thick.
MAFIC_CENTER = np.array([10_000.0, 10_000.0, -1_000.0])
MAFIC_SIZE = np.array([4_000.0, 2_000.0, 1_500.0])  # dx, dy, dz (m)

# Felsic intrusion: centred at (13 km, 13 km, -1.5 km), radius 1.5 km sphere.
FELSIC_CENTER = np.array([13_000.0, 13_000.0, -1_500.0])
FELSIC_RADIUS = 1_500.0

# Shear zone: vertical slab at x=10 km, width=600 m, y: 4–16 km, z: 0 to -3 km.
SHEAR_X_CENTER = 10_000.0
SHEAR_HALF_WIDTH = 300.0  # 600 m total width
SHEAR_Y0, SHEAR_Y1 = 4_000.0, 16_000.0
SHEAR_Z0, SHEAR_Z1 = -3_000.0, 0.0


def _geometric_padding_widths(d_core: float, n_pad: int, factor: float) -> np.ndarray:
    w0 = float(d_core) * float(factor)
    return np.array([w0 * (factor**i) for i in range(int(n_pad))], dtype=float)


def build_synthetic_mesh(
    extent_xy_m: float = 20_000.0,
    depth_m: float = 5_000.0,
    core_cell_m: float = 200.0,
    padding_cells: int = 6,
    padding_factor: float = 1.5,
) -> discretize.TensorMesh:
    """Tensor mesh for a 20 km × 20 km × 5 km depth domain with core cells ``core_cell_m``.

    Origin: south-west corner on surface; x, y increase east/north; z is positive up,
    top of mesh at z = 0. Padding expands outside the core box (x/y), and **below**
    the core in z (no padding above the surface).
    """
    n_core_xy = max(1, int(np.round(extent_xy_m / core_cell_m)))
    n_core_z = max(1, int(np.round(depth_m / core_cell_m)))
    n_pad = int(padding_cells)
    cs = float(core_cell_m)

    pad_xy = _geometric_padding_widths(cs, n_pad, padding_factor)
    pad_z = _geometric_padding_widths(cs, n_pad, padding_factor)
    hx = np.concatenate([pad_xy[::-1], np.full(n_core_xy, cs, dtype=float), pad_xy])
    hy = np.concatenate([pad_xy[::-1], np.full(n_core_xy, cs, dtype=float), pad_xy])
    # z ordering is bottom -> top. Include padding **below** the core, but do not pad above z=0
    # so near-surface cells exist (important for plan slices like k=2).
    hz = np.concatenate([pad_z[::-1], np.full(n_core_z, cs, dtype=float)])

    mesh = discretize.TensorMesh([hx, hy, hz], x0=np.zeros(3))
    # Place the core domain at x,y in [0, extent_xy_m] with padding outside.
    mesh.origin = np.array([-float(np.sum(pad_xy)), -float(np.sum(pad_xy)), -float(np.sum(hz))], dtype=float)
    return mesh


def _shear_indices(cell_centers: np.ndarray) -> np.ndarray:
    """Vertical tabular corridor: ~1 km wide in x, N–S along y, from surface to 3 km depth."""
    cc = np.asarray(cell_centers, dtype=float)
    # Snap target x to the nearest mesh-center column so coarse meshes remain non-empty.
    x_unique = np.unique(cc[:, 0])
    x_center_snap = x_unique[int(np.argmin(np.abs(x_unique - float(SHEAR_X_CENTER))))]
    inside = np.abs(cc[:, 0] - float(x_center_snap)) <= float(SHEAR_HALF_WIDTH)
    inside &= (cc[:, 1] >= float(SHEAR_Y0)) & (cc[:, 1] <= float(SHEAR_Y1))
    inside &= (cc[:, 2] >= float(SHEAR_Z0)) & (cc[:, 2] <= float(SHEAR_Z1))
    return inside


def _index_count(indices) -> int:
    """Count selected cells for bool masks or index arrays returned by model_builder."""
    arr = np.asarray(indices)
    if arr.dtype == bool:
        return int(np.count_nonzero(arr))
    return int(arr.size)


def build_true_model(mesh) -> np.ndarray:
    """Assign susceptibilities for background, mafic lens, felsic blob, and shear zone.

    Uses ``simpeg.utils.model_builder`` blocks/sphere plus a geometric shear slab.
    Overwrites in order: background → mafic → felsic → shear (target overprints mafic).

    Parameters
    ----------
    mesh : discretize.TensorMesh
        Mesh covering the synthetic domain (see ``build_synthetic_mesh``).

    Returns
    -------
    np.ndarray
        Susceptibility (SI) on every cell, shape ``(mesh.n_cells,)``.
    """
    cc = mesh.gridCC
    chi = np.full(mesh.n_cells, CHI_BG, dtype=float)

    # Mafic block extents from centre + size
    half = 0.5 * np.asarray(MAFIC_SIZE, dtype=float)
    p0_m = np.asarray(MAFIC_CENTER, dtype=float) - half
    p1_m = np.asarray(MAFIC_CENTER, dtype=float) + half
    mafic_thickness = abs(float(p1_m[2] - p0_m[2]))
    if mafic_thickness < 1000.0:
        raise ValueError(
            f"Mafic lens vertical extent must be >= 1000 m (got {mafic_thickness:.1f} m)."
        )
    ind_m = (
        (cc[:, 0] >= p0_m[0])
        & (cc[:, 0] <= p1_m[0])
        & (cc[:, 1] >= p0_m[1])
        & (cc[:, 1] <= p1_m[1])
        & (cc[:, 2] >= p0_m[2])
        & (cc[:, 2] <= p1_m[2])
    )
    if _index_count(ind_m) == 0:
        raise ValueError("Mafic lens indices are empty; check mesh/domain overlap.")
    chi[ind_m] = CHI_MAFIC
    print(f"Mafic cells: {_index_count(ind_m)}")

    felsic_thickness = 2.0 * float(FELSIC_RADIUS)
    if felsic_thickness < 1000.0:
        raise ValueError(
            f"Felsic intrusion vertical extent must be >= 1000 m (got {felsic_thickness:.1f} m)."
        )
    ind_f = np.linalg.norm(cc - np.asarray(FELSIC_CENTER, dtype=float), axis=1) <= float(FELSIC_RADIUS)
    if _index_count(ind_f) == 0:
        raise ValueError("Felsic intrusion indices are empty; check mesh/domain overlap.")
    chi[ind_f] = CHI_FELSIC
    print(f"Felsic cells: {_index_count(ind_f)}")

    shear_thickness = abs(float(SHEAR_Z1) - float(SHEAR_Z0))
    if shear_thickness < 1000.0:
        raise ValueError(f"Shear zone vertical extent must be >= 1000 m (got {shear_thickness:.1f} m).")
    ind_s = _shear_indices(cc)
    if _index_count(ind_s) == 0:
        raise ValueError("Shear zone indices are empty; check mesh/domain overlap.")
    chi[ind_s] = CHI_SHEAR
    print(f"Shear cells: {_index_count(ind_s)}")

    # Validation output requested for debugging body placement and susceptibility assignment.
    print(
        "Body cell counts:",
        {
            "mafic": _index_count(ind_m),
            "felsic": _index_count(ind_f),
            "shear": _index_count(ind_s),
        },
    )
    print(
        "Assigned susceptibilities:",
        {
            "background": CHI_BG,
            "mafic": CHI_MAFIC,
            "felsic": CHI_FELSIC,
            "shear": CHI_SHEAR,
        },
    )

    return chi


def plot_true_model(mesh, model: np.ndarray, title_prefix: str = "True model") -> None:
    """EW and NS vertical cross-sections through the domain centre, plus optional depth slice.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        Tensor mesh.
    model : np.ndarray
        Susceptibility array, length ``mesh.n_cells``.
    title_prefix : str
        Prefix for subplot titles.
    """
    m3 = np.asarray(model, dtype=float).reshape(mesh.shape_cells, order="F")
    xc = np.asarray(mesh.cell_centers_x)
    yc = np.asarray(mesh.cell_centers_y)
    zc = np.asarray(mesh.cell_centers_z)
    # Section locations requested for geological-body-centered diagnostics.
    ix = int(np.argmin(np.abs(xc - 10_000.0)))  # NS section through x ~ 10 km
    iy = int(np.argmin(np.abs(yc - 10_000.0)))  # EW section through y ~ 10 km
    # Near-surface plan slice (requested fixed index).
    # Use top-down z ordering so small k is near surface.
    z_top = zc[::-1]
    m3_top = m3[:, :, ::-1]
    k_plan = min(2, int(mesh.shape_cells[2] - 1))
    print(
        f"plot_true_model slices: ix={ix}/{mesh.shape_cells[0]-1}, "
        f"iy={iy}/{mesh.shape_cells[1]-1}, k_plan={k_plan}/{mesh.shape_cells[2]-1}, "
        f"x={xc[ix]:.1f} m, y={yc[iy]:.1f} m, z_plan={z_top[k_plan]:.1f} m"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin, vmax = 0.0, max(float(np.nanmax(model)), CHI_MAFIC * 1.05)

    # Plan view at ~2.5 km depth
    im0 = axes[0].pcolormesh(
        xc, yc, m3_top[:, :, k_plan].T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
    axes[0].set_title(f"{title_prefix}: plan z ≈ {z_top[k_plan]:.0f} m")
    plt.colorbar(im0, ax=axes[0], label="Susceptibility (SI)")

    # EW section (fixed northing)
    X, Z = np.meshgrid(xc, zc, indexing="ij")
    im1 = axes[1].pcolormesh(X, Z, m3[:, iy, :], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Elevation (m)")
    axes[1].set_title(f"{title_prefix}: EW section at y ≈ {yc[iy]:.0f} m")
    axes[1].set_ylim(-3000.0, 0.0)
    plt.colorbar(im1, ax=axes[1], label="Susceptibility (SI)")

    # NS section (fixed easting)
    Y, Z2 = np.meshgrid(yc, zc, indexing="ij")
    im2 = axes[2].pcolormesh(Y, Z2, m3[ix, :, :], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_xlabel("Northing (m)")
    axes[2].set_ylabel("Elevation (m)")
    axes[2].set_title(f"{title_prefix}: NS section at x ≈ {xc[ix]:.0f} m")
    axes[2].set_ylim(-3000.0, 0.0)
    plt.colorbar(im2, ax=axes[2], label="Susceptibility (SI)")

    plt.tight_layout()
    plt.show()


def plot_ns_cross_section_comparison(
    mesh,
    models: list[np.ndarray],
    labels: list[str],
    title: str = "NS section (model centre)",
) -> None:
    """Plot multiple susceptibility models on the same north–south vertical section."""
    xc = np.asarray(mesh.cell_centers_x)
    yc = np.asarray(mesh.cell_centers_y)
    zc = np.asarray(mesh.cell_centers_z)
    ix = int(np.argmin(np.abs(xc - 0.5 * float(xc.min() + xc.max()))))
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    vmax = max(float(np.nanmax(m)) for m in models)
    vmin = 0.0
    for ax, m, lab in zip(axes[0], models, labels):
        m3 = np.asarray(m, dtype=float).reshape(mesh.shape_cells, order="F")
        Y, Z = np.meshgrid(yc, zc, indexing="ij")
        im = ax.pcolormesh(Y, Z, m3[ix, :, :], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Northing (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(lab)
        plt.colorbar(im, ax=ax, label="Susceptibility (SI)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
