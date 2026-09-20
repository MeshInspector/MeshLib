<p align="left">
<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="https://github.com/user-attachments/assets/37d3a562-581d-421b-8209-ef6b224e96a8">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/caf6bdd1-b2f1-4d6d-9e22-c213db6fc9cf">
  <img alt="MeshLib logo" src="https://github.com/user-attachments/assets/caf6bdd1-b2f1-4d6d-9e22-c213db6fc9cf" width="60%">
</picture>
</p>

# MeshLib — an SDK that supercharges your 3D data processing efficiency

**Geometry algorithms engineered to revolutionize.** We help software engineers process, repair, measure and optimize 3D data with next-generation algorithms, reducing development time and costs — mesh boolean, mesh repair, decimation, offset, point cloud triangulation, ICP registration and voxel processing, in **C++, C, C#, Python and JavaScript**.

[**▶ Try it live**](https://demo.meshlib.io) · [Website](https://meshlib.io/) · [Documentation](https://meshlib.io/documentation/index.html) · [Quick start](https://meshlib.io/documentation/InstallationGuide.html) · [Examples](https://meshlib.io/documentation/Examples.html) · [Benchmarks](https://meshlib.io/blog/comparing-3d-boolean-libraries/) · [Discussions](https://github.com/MeshInspector/MeshLib/discussions)

[![build-test-distribute](https://github.com/MeshInspector/MeshLib/actions/workflows/build-test-distribute.yml/badge.svg?branch=master)](https://github.com/MeshInspector/MeshLib/actions/workflows/build-test-distribute.yml?branch=master)
[![PyPI version](https://badge.fury.io/py/meshlib.svg)](https://pypi.org/project/meshlib/)
[![PyPI downloads](https://img.shields.io/pypi/dm/meshlib?label=pypi%20downloads&color=blue)](https://pypi.org/project/meshlib/)
[![Python versions](https://img.shields.io/pypi/pyversions/meshlib?label=python)](https://pypi.org/project/meshlib/)
[![NuGet](https://img.shields.io/nuget/v/MeshLib?label=nuget&color=green)](https://www.nuget.org/packages/MeshLib)
[![npm](https://img.shields.io/npm/v/%40meshinspector%2Fmeshlib?label=npm%20%28wasm%29&color=red)](https://www.npmjs.com/package/@meshinspector/meshlib)
[![Stars](https://img.shields.io/github/stars/MeshInspector/MeshLib?style=flat&color=yellow)](https://github.com/MeshInspector/MeshLib)

![MeshLib SDK — 3D mesh processing library](https://github.com/user-attachments/assets/a65dc95f-675d-4fb8-ac17-6857c9a91554)

## Install

```bash
pip install meshlib                  # Python
dotnet add package MeshLib           # C#
npm install @meshinspector/meshlib   # JavaScript / WebAssembly
```

For C++ and C, see the [Installation Guide](https://meshlib.io/documentation/InstallationGuide.html).

## 60 seconds to your first result

Turn a scan into a triangle mesh:

```python
from meshlib import mrmeshpy as mm

pc   = mm.loadPoints("scan.ply")           # point cloud from any scanner
mesh = mm.triangulatePointCloud(pc)        # point cloud -> triangle mesh
mm.saveMesh(mesh, "model.stl")
```

Boolean operations that just work, with no non-manifold surprises:

```python
from meshlib import mrmeshpy as mm

a = mm.makeUVSphere(1.0, 64, 64)
b = mm.copyMesh(a)
b.transform(mm.AffineXf3f.translation(mm.Vector3f(0.7, 0.0, 0.0)))

res = mm.boolean(a, b, mm.BooleanOperation.Union)
if not res.valid():
    raise RuntimeError(res.errorString)
mm.saveMesh(res.mesh, "out_boolean.stl")
```

Heal a broken scan into a watertight, printable model. Voxel-based repair rebuilds
the surface through a signed distance field, so holes, self-intersections and
non-manifold edges disappear in a single pass:

```python
from meshlib import mrmeshpy as mm

mesh = mm.loadMesh("broken.stl")

params = mm.GeneralOffsetParameters()
params.voxelSize = mesh.computeBoundingBox().diagonal() / 200
params.signDetectionMode = mm.SignDetectionMode.HoleWindingRule
params.closeHolesInHoleWindingNumber = True

fixed = mm.generalOffsetMesh(mm.MeshPart(mesh), 0.0, params)   # watertight
mm.saveMesh(fixed, "fixed.stl")
```

Voxel repair resamples the surface, so the result is denser than the input —
add [`decimateMesh`](https://meshlib.io/documentation/Examples.html) to bring the
triangle count back down within a tolerance you set. Lighter, topological repair
(`fixMeshDegeneracies`, `fillHole`, `fixSelfIntersections`) is available when you
need to preserve the original vertices.

More in [Tutorials](https://meshlib.io/documentation/Tutorials.html) and the
[`examples/`](https://github.com/MeshInspector/MeshLib/tree/master/examples) folder —
including complete, runnable programs for repair, boolean, offset, ICP and triangulation
in all five languages.

## What MeshLib does

| | |
|---|---|
| **3D Boolean** | Fast, exact mesh booleans (union, intersection, difference) plus a voxel-based mode for messy input. |
| **Mesh repair** | Fix self-intersections, degeneracies, tunnels, multiple edges and undercuts; fill and stitch holes on flat and curved surfaces. |
| **Point cloud to mesh** | Triangulation with accurate normal estimation, uniform and grid sampling, outlier removal. |
| **Simplification** | Decimation within a set tolerance, remeshing and subdivision that keep the details you care about. |
| **Offset** | Shell, partial and weighted offsets for 3D printing, machining and hollowing. |
| **Registration** | Point-to-point and point-to-plane ICP, plus global registration of multiple scans. |
| **Distance & SDF** | Mesh to signed distance field and back by marching cubes, distance maps, iso-lines, projection and ray intersection. |
| **Voxels & CT** | DICOM import/export, voxel-grid processing and 3D volume rendering. |
| **Deformation** | Laplacian, freeform and relaxation smoothing, noise reduction. |
| **Segmentation** | Semi-automatic mesh segmentation guided by a curvature metric; graph-cut segmentation of voxel volumes. |
| **Collision detection** | Exact and precise self- and pairwise intersection tests. |
| **Viewer** | An embeddable OpenGL/ImGui viewer with UI components, or run the algorithms fully headless. |
| **File formats** | Meshes, point clouds, CT scans, polylines, distance maps and G-code — [full list](https://meshlib.io/feature/file-formats-supported-by-meshlib/). |

Full feature reference: [meshlib.io/features](https://meshlib.io/features/).

## Why teams pick MeshLib

**Measured, not claimed.** Our boolean benchmark compares MeshLib across nine
libraries on 2M-triangle models — [method, data and input meshes](https://meshlib.io/blog/comparing-3d-boolean-libraries/).
The [simplification benchmark](https://meshlib.io/blog/comparing-3d-simplification-libraries/)
does the same for decimation across 11 libraries. Both are public and reproducible.

**Manifold by construction.** Meshes use a half-edge data structure, in which most
non-manifold situations are simply not representable — a non-manifold edge, or a
vertex with two closed rings of triangles around it. Broken topology is caught where
it happens, not three pipeline stages later.

**One engine, five languages.** A native C++ core with official APIs for C, C#,
Python and JavaScript/WebAssembly. The same algorithms, the same results — on the
desktop, on a server, or in a browser tab.

**Built for real workloads.** Core algorithms are multithreaded; a CUDA module
accelerates distance maps, signed distance fields, offsets and swept volumes on
NVIDIA GPUs.

**We use it ourselves.** [MeshInspector](https://meshinspector.com/), our desktop and
web application, and [SmileInspector](https://smileinspector.io/), our
[FDA-cleared](https://smileinspector.io/news/smileinspector-launches-after-fda-clearance/)
clear-aligner platform, are both built entirely on this SDK, so it is exercised every
day on real production data.

**Supported, not abandoned.** Backed by a full-time team with commercial support,
regular releases and a responsive issue tracker.

## Platforms

Windows · macOS · Linux (Ubuntu/Debian, Fedora/RHEL) · WebAssembly

## Built with MeshLib

| Project | Field | What it does |
|---|---|---|
| [Polyga](https://polyga.com/) | 3D scanning | PointKit scanning platform, built on MeshLib |
| [Verisurf](https://verisurf.com/) | Metrology | 3D measurement, inspection and reverse engineering |
| [Brius Technologies](https://bravabraces.com/) | Orthodontics | Customized lingual orthodontic appliances |
| [Axial3D](https://axial3d.com/) | Medical imaging | Scan-to-3D-model services for surgical planning |
| [customed.ai](https://customed.ai/) | Medical devices | AI-driven patient-specific surgical instruments |
| [Relu](https://relu.eu/) | Digital dentistry | AI segmentation for dental CAD workflows |
| [spherene](https://spherene.ch/) | Generative design | Adaptive density minimal surfaces for AM |
| [ToffeeX](https://toffeex.com/) | Generative design | Physics-driven design optimization |
| [Enhatch](https://www.enhatch.com/) | Medical devices | Intelligent surgery platform |
| [CIMsystem](https://cimsystem.com/) | CAM / machining | CAD/CAM software for dental and industrial milling |
| 3DONS | Digital dentistry | Dental treatment planning software |
| Henning Larsen | Architecture / AEC | Computational design and terrain modelling |
| [MeshInspector](https://meshinspector.com/) | 3D printing / QA | STL editor, viewer and mesh repair application |
| [SmileInspector](https://smileinspector.io/) | Clear aligners | FDA-cleared aligner treatment planning platform |

Shipping something built on MeshLib? Open a PR and add it here — we like showing what people build.

## Getting started

1. **[Try the live demo](https://demo.meshlib.io)** — booleans, repair, decimation and ICP in the browser. No install, no sign-up.
2. **Evaluate locally.** `pip install meshlib`, then work through the [tutorials](https://meshlib.io/documentation/Tutorials.html). Free for non-commercial and educational use under our [licence](https://github.com/MeshInspector/MeshLib?tab=License-1-ov-file#readme).
3. **See it in an application.** [MeshInspector](https://meshinspector.com/) is the GUI built on MeshLib — desktop and web, 30-day trial.
4. **[Book a call](https://meshlib.io/book-a-call/).** Integration guidance, architecture questions and commercial licensing tailored to your project.

## Licence

MeshLib is **source-available**: read the code, fork it, and use it free for
non-commercial and educational work. Commercial use requires a licence — see the
[full terms](https://github.com/MeshInspector/MeshLib?tab=License-1-ov-file#readme)
and the [licence page](https://meshlib.io/license/).

## Citing MeshLib

If MeshLib supports your research, please cite the exact version you used — find it with `pip show meshlib`.

```bibtex
@software{meshlib,
  author       = {{MeshLib Development Team}},
  title        = {{MeshLib}: 3D Mesh Processing Library},
  organization = {AMV Consulting LLC},
  year         = {2026},
  version      = {3.1.3.429},
  url          = {https://meshlib.io},
  note         = {Please cite the exact version used in your work}
}
```

Other styles (APA, IEEE, Chicago) and guidance on citing a specific algorithm:
[meshlib.io/citation-guide](https://meshlib.io/citation-guide/). Published something
that uses MeshLib? Tell us in [Discussions](https://github.com/MeshInspector/MeshLib/discussions) — we keep a list.

## Community and support

- **Bugs and feature requests** → [GitHub Issues](https://github.com/MeshInspector/MeshLib/issues/)
- **Questions, ideas, show-and-tell** → [GitHub Discussions](https://github.com/MeshInspector/MeshLib/discussions)
- **Commercial support** → [meshlib.io/book-a-call](https://meshlib.io/book-a-call/)

If MeshLib saves you time, a ⭐ helps other engineers find it.
