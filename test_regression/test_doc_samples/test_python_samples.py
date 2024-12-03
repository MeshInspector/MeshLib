import pytest
from module_helper import *

from constants import test_files_path
from pathlib import Path
from helpers.meshlib_helpers import compare_mesh, compare_meshes_similarity, compare_points_similarity
from pytest_check import check

import shutil
import runpy

SAMPLES_FOLDER = Path("../examples/python-examples")


def run_code_sample(code_path: str, args: list):
    """
    Function to run code sample with CLI args
    :param code_path: path to code sample
    :param args: arguments to pass to code sample
    """
    # Preserving original args
    original_sys_argv = sys.argv.copy()
    # setting new args for code to execute
    sys.argv = [code_path] + args
    try:
        runpy.run_path(code_path)
    finally:
        # restoring original args
        sys.argv = original_sys_argv

@pytest.mark.parametrize("doc_case",
                         [
                             pytest.param({'sample': "FreeFormDeformation.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': ['deformed_mesh.stl']}, id="FreeFormDeformation.dox.py",
                                          marks=pytest.mark.bindingsV3),
                             pytest.param({'sample': "LaplacianExample.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': ['deformed_mesh.stl']}, id="LaplacianExample.dox.py",
                                          marks=pytest.mark.bindingsV3),
                             pytest.param({'sample': "MeshBoolean.dox.py", 'input_files': [],
                                           'output_files': ['out_boolean.stl']}, id="MeshBoolean.dox.py"),
                             pytest.param({'sample': "MeshDecimate.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': ['decimatedMesh.stl']}, id="MeshDecimate.dox.py"),
                             pytest.param({'sample': "MeshExtrude.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': ['extrudedMesh.stl']}, id="MeshExtrude.dox.py"),
                             pytest.param({'sample': "MeshFillHole.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': ['filledMesh.stl']}, id="MeshFillHole.dox.py"),
                             pytest.param({'sample': "MeshICP.dox.py", 'input_files': ['meshA.stl', 'meshB.stl'],
                                           'output_files': ['meshA_icp.stl']}, id="MeshICP.dox.py"),
                             pytest.param({'sample': "MeshLoadSave.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': ['mesh.ply']}, id="MeshLoadSave.dox.py"),
                             pytest.param({'sample': "MeshModification.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': []}, id="MeshModification.dox.py"),
                             pytest.param({'sample': "MeshOffset.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': ['offsetMesh.stl']}, id="MeshOffset.dox.py"),
                             pytest.param({'sample': "MeshStitchHole.dox.py",
                                           'input_files': ['meshAwithHole.stl', 'meshBwithHole.stl'],
                                           'output_files': ['stitchedMesh.stl']}, id="MeshStitchHole.dox.py"),
                             pytest.param({'sample': "NoiseDenoiseExample.dox.py", 'input_files': ['mesh.stl'],
                                           'output_files': ['noised_mesh.stl', 'denoised_mesh.stl']},
                                          id="NoiseDenoiseExample.dox.py",
                                          marks=pytest.mark.bindingsV3),
                             pytest.param({'sample': "Numpy.dox.py", 'input_files': [], 'output_files': []},
                                          id="Numpy.dox.py"),
                             pytest.param(
                                 {'sample': "NumpyTriangulation.dox.py", 'input_files': [], 'output_files': []},
                                 id="NumpyTriangulation.dox.py"),
                             pytest.param(
                                 {'sample': "Triangulation_v2.dox.py", 'input_files': [], 'output_files': []},
                                 id="Triangulation_v2.dox.py",
                                 marks=pytest.mark.bindingsV2),
                             pytest.param(
                                 {'sample': "Triangulation_v3.dox.py", 'input_files': [], 'output_files': []},
                                 id="Triangulation_v3.dox.py",
                                 marks=pytest.mark.bindingsV3),
                             pytest.param(
                                 {'sample': "GlobalRegistration.dox.py",
                                  'input_files': ["cloud0.ply", "cloud1.ply", "cloud2.ply"],
                                  'output_files': ["out.ply"],
                                  'args': ["cloud0.ply", "cloud1.ply", "cloud2.ply", "out.ply"],
                                  'verify': 'points'
                                  },
                                 id="GlobalRegistration.dox.py",
                                 marks=pytest.mark.bindingsV3),
                         ])
@pytest.mark.smoke
def test_python_doc_samples(tmp_path, doc_case):
    """
    Test copies python files from examples to tmp_path, and executes them.
    """
    # Copy files from examples to tmp_path
    py_file = doc_case['sample']
    shutil.copy(SAMPLES_FOLDER / py_file, tmp_path / py_file)
    for mesh in doc_case['input_files']:
        shutil.copy(Path(test_files_path) / "doc_samples" / "python" / py_file / mesh, mesh)

    # Execute file in tmp_path
    if 'args' in doc_case:
        run_code_sample(tmp_path / py_file, doc_case['args'])
    else:
        run_code_sample(tmp_path / py_file, [])

    for out_mesh in doc_case['output_files']:
        shutil.copy(out_mesh, tmp_path / out_mesh)

    # Compare files in tmp_path with reference files
    if 'verify' in doc_case and doc_case['verify'] == 'points':
        for out_cloud in doc_case['output_files']:
            with check:
                compare_points_similarity(tmp_path / out_cloud,
                                    Path(test_files_path) / "doc_samples" / "python" / py_file / out_cloud)
    else:
        for out_mesh in doc_case['output_files']:
            with check:
                mesh1 = mrmeshpy.loadMesh(tmp_path / out_mesh)
                mesh2 = mrmeshpy.loadMesh(Path(test_files_path) / "doc_samples" / "python" / py_file / out_mesh)
                compare_meshes_similarity(mesh1, mesh2)
