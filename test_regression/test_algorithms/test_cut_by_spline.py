from pathlib import Path

import meshlib.mrmeshpy as mm
import pytest
from constants import test_files_path
from module_helper import *

input_folder = Path(test_files_path) / "algorithms" / "cut_by_spline"


def _make_spline(mesh):
    # Control points form an already-closed contour (last point repeats the first).
    control_points = mm.loadPoints(input_folder / "control_points_closed.xyz").points.vec

    # Project the control points onto the mesh and gather surface normals there.
    contour_points = mm.Contour3f()
    contour_normals = mm.Contour3f()
    for point in control_points:
        mtp = mm.findProjection(point, mesh).mtp
        contour_points.append(mesh.triPoint(mtp))
        contour_normals.append(mesh.normal(mtp))

    settings = mm.SplineSettings()
    settings.normals = contour_normals
    settings.controlStability = 2
    settings.iterations = 1
    settings.normalsAffectShape = True
    settings.samplingStep = 0.1
    return mm.makeSpline(contour_points, settings)


@pytest.mark.smoke
def test_cut_by_spline_without_projection_self_intersects():
    """
    The spline returned by makeSpline does not lie exactly on the mesh surface,
    so cutting the mesh directly by it fails with a self-intersection error.
    """
    mesh = mm.loadMesh(input_folder / "toothex.ctm")
    spline = _make_spline(mesh)
    with pytest.raises(RuntimeError, match="self intersections"):
        mm.cutMeshByContour(mesh, spline.contour)


@pytest.mark.smoke
def test_cut_by_spline_with_projection():
    """
    Projecting the spline onto the mesh surface with projectSpline before cutting
    removes the self-intersections, so the cut succeeds.
    """
    mesh = mm.loadMesh(input_folder / "toothex.ctm")
    spline = _make_spline(mesh)
    projected_contour = mm.projectSpline(mesh, spline)
    left_faces = mm.cutMeshByContour(mesh, projected_contour)
    assert left_faces.count() > 0
