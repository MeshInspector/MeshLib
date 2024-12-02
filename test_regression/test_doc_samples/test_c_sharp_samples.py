import subprocess
import shutil
import pytest

from module_helper import *

from constants import test_files_path
from pathlib import Path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_points_similarity
from pytest_check import check



SAMPLE_EXEC_DIR = Path("..\\source\\x64\\Release")
SAMPLE_EXEC_NAME = "c-sharp-examples.exe"

@pytest.mark.skipif(
    "not config.getoption('--run-c-sharp-samples')",
    reason="Only run when --run-c-sharp-samples is given",
)
@pytest.mark.smoke
@pytest.mark.parametrize("case",
                         [
                             pytest.param({'sample': "GlobalRegistrationExample",
                                           'input_files': ["cloud0.ply", "cloud1.ply", "cloud2.ply"],
                                           'output_files': ["out.ply"],
                                           'args': ["cloud0.ply", "cloud1.ply", "cloud2.ply", "out.ply"],
                                           'verify': 'points'
                                           },
                                          id="GlobalRegistration"),
                            pytest.param({'sample': "MeshBooleanExample",
                                           'input_files': ['mesh_a.stl', 'mesh_b.stl'],
                                           'output_files': ['out_boolean.stl'],
                                           'args': ['mesh_a.stl', 'mesh_b.stl']
                                           },
                                          id="MeshBoolean"),
                            pytest.param({'sample': "MeshDecimateExample",
                                           'input_files': ['mesh.stl'],
                                           'output_files': ['decimated_mesh.stl']
                                           },
                                          id="MeshDecimate"),
                            pytest.param({'sample': "MeshOffsetExample",
                                           'input_files': ['mesh.stl'],
                                           'output_files': ['mesh_offset.stl'],
                                           'args': ['2.5']
                                           },
                                          id="MeshOffset"),
                            pytest.param({'sample': "MeshFillHoleExample",
                                           'input_files': ['mesh.stl'],
                                           'output_files': ['mesh_filled.stl'],
                                           'args': ['mesh.stl', 'mesh_filled.stl']
                                           },
                                          id="MeshFillHole"),
                            pytest.param({'sample': "MeshLoadSaveExample",
                                           'input_files': ['mesh.stl'],
                                           'output_files': ['mesh.ply'],
                                           },
                                          id="MeshLoadSave"),
                            pytest.param({'sample': "MeshExportExample",
                                           'input_files': ['mesh.stl'],
                                           'output_files': [],
                                           },
                                          id="MeshExport"),
                            pytest.param({'sample': "MeshResolveDegenerationsExample",
                                           'input_files': ['mesh.ctm'],
                                           'output_files': ['mesh_fixed.ctm'],
                                           'args': ['mesh.ctm', 'mesh_fixed.ctm']
                                           },
                                          id="MeshResolveDegenerations"),
                         ])
def test_c_sharp_samples(tmp_path, case):
    """

    """
    print(SAMPLE_EXEC_DIR)

    def list_all_files(directory):
        # Walk through directory and its subdirectories
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Create full file path and print it
                file_path = os.path.join(root, file)
                print(file_path)

    list_all_files(SAMPLE_EXEC_DIR)

    sample = case['sample']
    for mesh in case['input_files']:
        shutil.copy(Path(test_files_path) / "doc_samples" / "c-sharp" / sample /mesh, tmp_path / mesh)

    args = case['args'] if 'args' in case else []
    a = subprocess.run([SAMPLE_EXEC_DIR / SAMPLE_EXEC_NAME, sample] + args, capture_output=True, cwd=tmp_path)
    assert a.returncode == 0, f"Return code is {a.returncode}"
    if a.stdout:
        print("stdout:")
        print(a.stdout.decode('utf-8'))
    if a.stderr:
        print("stderr:")
        print(a.stderr.decode('utf-8'))

    # Compare files in tmp_path with reference files
    if 'verify' in case and case['verify'] == 'points':
        for out_cloud in case['output_files']:
            with check:
                compare_points_similarity(tmp_path / out_cloud,
                                    Path(test_files_path) / "doc_samples" / "c-sharp" / sample / out_cloud)
    else:
        for out_mesh in case['output_files']:
            with check:
                mesh1 = mrmeshpy.loadMesh(tmp_path / out_mesh)
                mesh2 = mrmeshpy.loadMesh(Path(test_files_path) / "doc_samples" / "c-sharp" / sample / out_mesh)
                compare_meshes_similarity(mesh1, mesh2)
