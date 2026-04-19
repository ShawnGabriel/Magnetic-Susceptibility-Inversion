"""Run magnetic susceptibility inversions with SimPEG.

This module provides two inversion stages:
1) a smooth (L2) Tikhonov inversion,
2) a sparse (IRLS) inversion initialized from the smooth result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from simpeg import (
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    optimization,
    regularization,
)


def _processed_dir() -> Path:
    """Return `data/processed` path relative to repository root."""
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_iterations_directive(folder: str):
    """Create iteration-saving directive across SimPEG versions."""
    if hasattr(directives, "SaveIterationsGeoH5"):
        # Newer versions may provide geoh5-based iteration writers.
        return directives.SaveIterationsGeoH5(save_objective_function=True)
    if hasattr(directives, "SaveOutputEveryIteration"):
        return directives.SaveOutputEveryIteration(save_txt=False)
    if hasattr(directives, "SaveOutputDictEveryIteration"):
        return directives.SaveOutputDictEveryIteration()
    if hasattr(directives, "SaveIterationsDirectory"):
        return directives.SaveIterationsDirectory(folder=folder)
    return None


def run_smooth_inversion(
    simulation,
    data_obj,
    mesh,
    actind: np.ndarray,
    starting_susceptibility: float = 1e-4,
) -> np.ndarray:
    """Run a smooth (L2) Tikhonov magnetic inversion.

    Parameters
    ----------
    simulation : simpeg.potential_fields.magnetics.simulation.Simulation3DIntegral
        Forward simulation object.
    data_obj : simpeg.data.Data
        SimPEG data container with observed data and uncertainties.
    mesh : discretize.TensorMesh
        Inversion mesh.
    actind : np.ndarray
        Boolean active-cell mask with length `mesh.n_cells`.
    starting_susceptibility : float, default=1e-4
        Starting susceptibility model value for all active cells (SI).

    Returns
    -------
    np.ndarray
        Recovered smooth susceptibility model on active cells.
    """
    actind = np.asarray(actind, dtype=bool)
    n_active = int(actind.sum())
    if n_active == 0:
        raise ValueError("actind contains no active cells.")

    # m0 is the initial model used by the optimizer.
    m0 = np.full(n_active, float(starting_susceptibility), dtype=float)

    # Weighted least-squares data misfit: enforces fit to observed TMI.
    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=simulation)

    # Smoothness regularization (smallness + first derivatives).
    reg = regularization.WeightedLeastSquares(mesh, active_cells=actind)
    reg.alpha_s = 1.0   # model smallness weight
    reg.alpha_x = 1.0   # x-gradient smoothing weight
    reg.alpha_y = 1.0   # y-gradient smoothing weight
    reg.alpha_z = 1.0   # z-gradient smoothing weight

    # ProjectedGNCG handles bound-constrained Gauss-Newton-CG optimization.
    opt = optimization.ProjectedGNCG(
        maxIter=12,       # max outer inversion iterations (laptop-safe)
        lower=0.0,        # susceptibility cannot be negative in this setup
        upper=1.0,        # conservative physical upper bound
        maxIterCG=15,     # max CG steps per GN update
        tolCG=1e-3,       # CG convergence tolerance
    )

    # Combines misfit + regularization into one objective.
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    # Manual initial beta avoids BetaEstimate_ByEig (heavy eigenvalue solve) on large meshes.
    phi_d0 = float(inv_prob.dmisfit(m0))
    phi_m0 = float(inv_prob.reg(m0))
    inv_prob.beta = phi_d0 / max(phi_m0, 1e-30)

    target = directives.TargetMisfit(chifact=1.0)
    save_dir = _save_iterations_directive(folder="smooth_iterations")

    directive_list = [beta_sched, target]
    if save_dir is not None:
        directive_list.append(save_dir)

    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_smooth = inv.run(m0)

    out_path = _processed_dir() / "susceptibility_smooth.npy"
    np.save(out_path, m_smooth)
    return m_smooth


def run_sparse_inversion(
    simulation,
    data_obj,
    mesh,
    actind: np.ndarray,
    m0: np.ndarray,
    p_s: float = 0,
    p_x: float = 1,
    p_y: float = 1,
    p_z: float = 1,
) -> np.ndarray:
    """Run sparse-norm (IRLS) magnetic inversion starting from smooth model.

    Parameters
    ----------
    simulation : simpeg.potential_fields.magnetics.simulation.Simulation3DIntegral
        Forward simulation object.
    data_obj : simpeg.data.Data
        SimPEG data container with observed data and uncertainties.
    mesh : discretize.TensorMesh
        Inversion mesh.
    actind : np.ndarray
        Boolean active-cell mask with length `mesh.n_cells`.
    m0 : np.ndarray
        Starting model (typically smooth inversion output), active cells only.
    p_s, p_x, p_y, p_z : float
        Sparse norm exponents for smallness and x/y/z gradients.
        Lower values (<2) promote blockier, sharper models.

    Returns
    -------
    np.ndarray
        Recovered sparse susceptibility model on active cells.
    """
    actind = np.asarray(actind, dtype=bool)
    n_active = int(actind.sum())
    m0 = np.asarray(m0, dtype=float).reshape(-1)
    if m0.size != n_active:
        raise ValueError("m0 size must equal number of active cells.")

    dmis = data_misfit.L2DataMisfit(data=data_obj, simulation=simulation)

    # Sparse regularization for sharper geological boundaries.
    reg = regularization.Sparse(mesh, active_cells=actind)
    # SimPEG expects one norm value per regularization function (smallness, x, y, z).
    reg.norms = [p_s, p_x, p_y, p_z]
    reg.alpha_s = 1.0
    reg.alpha_x = 1.0
    reg.alpha_y = 1.0
    reg.alpha_z = 1.0

    opt = optimization.ProjectedGNCG(
        maxIter=12,       # laptop-safe outer iterations for IRLS
        lower=0.0,
        upper=1.0,
        maxIterCG=15,
        tolCG=1e-3,
    )

    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    phi_d0 = float(inv_prob.dmisfit(m0))
    phi_m0 = float(inv_prob.reg(m0))
    inv_prob.beta = phi_d0 / max(phi_m0, 1e-30)

    beta_sched = directives.BetaSchedule(coolingFactor=2.0, coolingRate=1)
    target = directives.TargetMisfit(chifact=1.0)

    # IRLS updates model weights to approximate sparse norms each iteration.
    if hasattr(directives, "UpdateIRLS"):
        irls = directives.UpdateIRLS(
            f_min_change=1e-4,
            max_irls_iterations=8,
            irls_cooling_factor=1.5,
        )
    elif hasattr(directives, "Update_IRLS"):
        irls = directives.Update_IRLS(
            f_min_change=1e-4, max_irls_iterations=8, coolEpsFact=1.5
        )
    else:
        irls = None

    save_dir = _save_iterations_directive(folder="sparse_iterations")

    directive_list = [target]
    if irls is not None:
        directive_list.append(irls)
    else:
        directive_list.insert(0, directives.BetaSchedule(coolingFactor=2.0, coolingRate=1))
    if save_dir is not None:
        directive_list.append(save_dir)

    inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)
    m_sparse = inv.run(m0)

    out_path = _processed_dir() / "susceptibility_sparse.npy"
    np.save(out_path, m_sparse)
    return m_sparse
