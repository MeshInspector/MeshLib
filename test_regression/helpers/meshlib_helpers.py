from module_helper import *
import meshlib.mrmeshpy as mrmesh
import pathlib


def relative_hausdorff(mesh1: mrmesh.Mesh or pathlib.Path or mrmesh.Path or str,
                       mesh2: mrmesh.Mesh or pathlib.Path or mrmesh.Path or str):
    """
    Calculate Hausdorff distance between two meshes, normalized on smallest bounding box diagonal.
    1.0 means that the meshes are equal, 0.0 means that they are completely different.
    The value is in range [0.0, 1.0]

    :param mesh1: first mesh or path to it
    :param mesh2: second mesh or path to it
    """
    if isinstance(mesh1, str) or isinstance(mesh1, pathlib.Path):
        mesh1 = mrmesh.loadMesh(str(mesh1))
    elif isinstance(mesh1, mrmesh.Path):
        mesh1 = mrmesh.loadMesh(mesh1)
    if isinstance(mesh2, str) or isinstance(mesh2, pathlib.Path):
        mesh2 = mrmesh.loadMesh(str(mesh2))
    elif isinstance(mesh2, mrmesh.Path):
        mesh2 = mrmesh.loadMesh(mesh2)
    distance = mrmesh.findMaxDistanceSq(mesh1, mesh2) ** 0.5
    diagonal = min(mesh1.getBoundingBox().diagonal(), mesh2.getBoundingBox().diagonal())
    val = 1.0 - (distance / diagonal)
    val = 0.0 if val < 0.0 else val  # there are some specific cases when metric can be below zero,
    # but exact values have no practical meaning, any value beyond zero means "completely different"
    return val
