"""
Runs the doc examples of the compiled languages (examples/{cpp,c,c-sharp,rust,java}-examples) as CI built them.
Each language runs the example in its own empty directory with the same input files, and its output files and the
numbers it prints must match those of the C++ example, which itself must produce non-empty results.
The languages and their commands come from the environment variable DOC_EXAMPLE_CMDS, one `LANG=COMMAND` per line,
where `{name}` in COMMAND is the example name, e.g. `cpp=build/cpp-examples/{name}`.
"""
import math
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from pytest_check import check

from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_points_similarity
from module_helper import *

DATA = Path(test_files_path).resolve() / "doc_samples" / "python"
FONT = Path(__file__).resolve().parents[2] / "thirdparty" / "Noto_Sans" / "NotoSans-Regular.ttf"


def torus():
    return mrmeshpy.makeTorus()


def shifted_sphere():
    mesh = mrmeshpy.makeUVSphere(1.0, 32, 32)
    mesh.transform(mrmeshpy.AffineXf3f.translation(mrmeshpy.Vector3f(0.5, 0.0, 0.0)))
    return mesh


# inputs: file name in the run directory -> file in doc_samples/python (converted if the extension differs) or a mesh maker
# outputs: files compared as meshes, `points`: files compared as point clouds
EXAMPLES = {
    "Collision": {},
    "CollisionPrecise": {},
    "CollisionSelf": {},
    "ContourTriangulation": {"inputs": {"logo.jpg": "ContourTriangulation.dox.py/logo.jpg"}, "outputs": ["LogoMesh.ctm"]},
    "FreeFormDeformation": {"inputs": {"mesh.stl": "FreeFormDeformation.dox.py/mesh.stl"}, "outputs": ["deformed_mesh.stl"]},
    "Fusion": {"inputs": {"Points.ply": "Fusion.dox.py/Points.ply"}, "outputs": ["Mesh.ctm"]},
    "GlobalRegistration": {
        "inputs": {f"cloud{i}.ply": f"GlobalRegistration.dox.py/cloud{i}.ply" for i in range(3)},
        "args": ["cloud0.ply", "cloud1.ply", "cloud2.ply", "out.ply"],
        "points": ["out.ply"],
    },
    "LaplacianDeformation": {"inputs": {"mesh.stl": "LaplacianExample.dox.py/mesh.stl"}, "outputs": ["deformed_mesh.stl"]},
    "MeshBoolean": {"outputs": ["out_boolean.stl"]},
    "MeshDecimate": {"inputs": {"mesh.stl": "MeshDecimate.dox.py/mesh.stl"}, "outputs": ["decimated_mesh.stl"]},
    "MeshExport": {},
    "MeshExtrude": {"inputs": {"mesh.stl": "MeshExtrude.dox.py/mesh.stl"}, "outputs": ["extrudedMesh.stl"]},
    "MeshFillHole": {"inputs": {"mesh.stl": "MeshFillHole.dox.py/mesh.stl"}, "outputs": ["filledMesh.stl"]},
    "MeshFixDegeneracies": {"inputs": {"mesh.stl": "MeshFixDegeneracies.dox.py/mesh.ctm"}, "outputs": ["fixedMesh.stl"]},
    "MeshFromText": {"args": [str(FONT), "MeshLib"], "outputs": ["mesh.ply"]},
    "MeshICP": {"inputs": {"meshA.stl": "MeshICP.dox.py/meshA.stl", "meshB.stl": "MeshICP.dox.py/meshB.stl"},
                "outputs": ["meshA_icp.stl"]},
    "MeshLoadSave": {"inputs": {"mesh.stl": "MeshLoadSave.dox.py/mesh.stl"}, "outputs": ["mesh.ply"]},
    "MeshModification": {},
    "MeshOffset": {"outputs": ["mesh_offset.stl"]},
    "MeshOffsetWeighted": {"outputs": ["offset_weighted.ctm"]},
    "MeshStitchHole": {"inputs": {"meshAwithHole.stl": "MeshStitchHole.dox.py/meshAwithHole.stl",
                                  "meshBwithHole.stl": "MeshStitchHole.dox.py/meshBwithHole.stl"},
                       "outputs": ["stitchedMesh.stl"]},
    "NoiseDenoise": {"inputs": {"mesh.stl": "NoiseDenoiseExample.dox.py/mesh.stl"},
                     "outputs": ["noised_mesh.stl", "denoised_mesh.stl"]},
    "PointsToMesh": {"inputs": {"Points.ply": "PointsToMesh.dox.py/Points.ply"}, "outputs": ["Mesh.ctm"]},
    "SignedDistance": {"inputs": {"mesh1.ctm": torus, "mesh2.ctm": shifted_sphere}},
    "SignedDistancePointToMesh": {"inputs": {"mesh1.ctm": torus}},
    "SignedDistances": {"inputs": {"mesh1.ctm": torus, "mesh2.ctm": shifted_sphere}},
    "TerrainTriangulation": {"inputs": {"TerrainPoints.ply": "TerrainTriangulation.dox.py/TerrainPoints.ply"},
                             "outputs": ["TerrainMesh.ctm"]},
    "Triangulation": {"outputs": ["result.stl"]},
}

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def printed_numbers(text: str):
    """The numbers in the output of an example, with true/false as 1/0, since each language prints booleans its own way."""
    text = re.sub(r"\b[Tt]rue\b", "1", text)
    text = re.sub(r"\b[Ff]alse\b", "0", text)
    return [float(x) for x in NUMBER.findall(text)]


def prepare_inputs(case: dict, run_dir: Path):
    for name, source in case.get("inputs", {}).items():
        target = run_dir / name
        if callable(source):
            mrmeshpy.saveMesh(source(), target)
        elif Path(source).suffix == target.suffix:
            shutil.copy(DATA / source, target)
        else:
            mrmeshpy.saveMesh(mrmeshpy.loadMesh(DATA / source), target)


def run_example(name: str, lang: str, cmd: str, case: dict, run_dir: Path):
    run_dir.mkdir()
    prepare_inputs(case, run_dir)
    argv = shlex.split(cmd.replace("{name}", name), posix=os.name != "nt") + case.get("args", [])  # keeps Windows backslashes
    res = subprocess.run(argv, cwd=run_dir, capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=600)
    print(f"--- {lang}: {' '.join(argv)}\n{res.stdout}{res.stderr}")
    assert res.returncode == 0, f"{lang} {name} exited with {res.returncode}:\n{res.stderr}"
    for out in case.get("outputs", []) + case.get("points", []):
        assert (run_dir / out).is_file(), f"{lang} {name} did not write {out}"
    return res.stdout


@pytest.fixture
def doc_example_cmds():
    lines = os.environ.get("DOC_EXAMPLE_CMDS", "").splitlines()
    return dict(line.strip().split("=", 1) for line in lines if line.strip())


@pytest.mark.smoke
@pytest.mark.parametrize("name", EXAMPLES.keys())
def test_compiled_doc_example(tmp_path, name, doc_example_cmds):
    if not doc_example_cmds:
        pytest.skip("DOC_EXAMPLE_CMDS is not set")
    case = EXAMPLES[name]
    langs = sorted(doc_example_cmds, key=lambda lang: lang != "cpp")  # C++ first, the others are compared with it
    base = langs[0]

    base_dir = tmp_path / base
    base_stdout = run_example(name, base, doc_example_cmds[base], case, base_dir)
    for out in case.get("outputs", []):
        assert mrmeshpy.loadMesh(base_dir / out).topology.numValidFaces() > 0, f"{base} {name}: {out} has no triangles"
    for out in case.get("points", []):
        assert mrmeshpy.loadPoints(base_dir / out).validPoints.count() > 0, f"{base} {name}: {out} has no points"
    base_numbers = printed_numbers(base_stdout)

    for lang in langs[1:]:
        with check:
            run_dir = tmp_path / lang
            stdout = run_example(name, lang, doc_example_cmds[lang], case, run_dir)
            numbers = printed_numbers(stdout)
            assert len(numbers) == len(base_numbers) and all(
                math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-5) for a, b in zip(numbers, base_numbers)), \
                f"{lang} {name} printed other numbers than {base}:\n{stdout}\n{base}:\n{base_stdout}"
            for out in case.get("outputs", []):
                mesh, base_mesh = mrmeshpy.loadMesh(run_dir / out), mrmeshpy.loadMesh(base_dir / out)
                if mesh != base_mesh:
                    print(f"{lang} {out} differs from {base} {out}")
                    compare_meshes_similarity(mesh, base_mesh, skip_volume=True)
            for out in case.get("points", []):
                compare_points_similarity(run_dir / out, base_dir / out, testname=f"{lang} {name}")
