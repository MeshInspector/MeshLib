[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/MeshInspector/MeshLib)

[![build-test-distribute](https://github.com/MeshInspector/MeshLib/actions/workflows/build-test-distribute.yml/badge.svg?branch=master)](https://github.com/MeshInspector/MeshLib/actions/workflows/build-test-distribute.yml?branch=master) 
[![PyPI version](https://badge.fury.io/py/meshlib.svg)](https://badge.fury.io/py/meshlib)
[![Python](https://img.shields.io/pypi/pyversions/meshlib.svg?style=plastic)](https://badge.fury.io/py/meshlib)
[![Downloads](https://pepy.tech/badge/meshlib/month?style=flat-square)](https://pepy.tech/project/meshlib)

![MeshInspector/MeshLib](https://user-images.githubusercontent.com/10034350/176395489-6349670a-b9eb-4f53-886a-35a75b55e6ac.png)

# Welcome to the MeshLib!
3D scanning is becoming more and more ubiquitous. Robotic automation, self-driving cars and multitude of other industrial, medical and scientific applications require advanced computer vision to deliver the levels of automation customers expect these days. The great rise of AI gave another boost to computer vision and the need to utilize 3D data to make machines smarter. Not only are tasks at hand becoming more complex, but the size of data grows exponentially. 

There is a multitude of general purpose libraries which deal with 3D data. Some stem from popular CAD packages, some are open source. The commercial ones tend to be quite expensive while open source are free though tend to be limited in functionality provided. Also those libraries value generality above other features  to allow maximum number of applications, but with the growing amounts of 3D data, performance is critical as it never has  been. Some of it can be addressed by using the scale of a commercial cloud, last generation CPU or GPU but underlying complexity of data representation makes it very hard and laborsome.

The goal which we set when designing MeshLib was to value simplicity and performance while providing a wide gamut of useful computational algorithms. The library also supports the most important data structures today’s sensors can produce - pointcloud, mesh, volume and more. For example, mesh is represented by half-edge data structure and cannot be made non-manifold. Some applications may require non-manifoldness, but most practical scans can be represented as manifoldness meshes without an issue. 

## Some useful links
* [MeshInspector](https://github.com/MeshInspector/MeshInspector/) is a free application based on MeshLib

* MeshInspector [online web-browser version](https://demo.meshinspector.com/)

* MeshLib and MeshInspector [slides](https://docs.google.com/presentation/d/1D0Ry6SE2J25PBtO_G9ZIp1cavoX2wyyY8jgvtjeayC4/edit?usp=sharing)

* Comparison [slides](https://docs.google.com/presentation/d/1Tw5ppmWoF-aRwuVqa6xdMSEjmEd5Y6O2ny7Gu8iQBos/edit?usp=sharing) of mesh operations with VTK library

* MeshInspector [YouTube channel](https://www.youtube.com/channel/UCv9bNhwoVDPaLPPyWJeVPNg)

* MeshLib [documentation](https://meshinspector.github.io/MeshLib/html/index.html)

* [Email us](mailto:support@meshinspector.com)

* Contact us anonymously [form](https://meshinspector.github.io/ReportIssue/)

## Major features
This list is not full and updating each day
### Math basics
 - Math primitives and operations support (Vectors 2D, 3D, 4D; Lines; Planes; Bounding Boxes; Matrices; Affine transformations; Quaternions; Histograms; etc.)
### 3D data handling, creation, modification
 - 3D data various representations support: Mesh, Voxel and Point Cloud.
 - Data creation 
   - Mesh creation by given vertices and triangles,
   - Surface primitives (e.g. tor, cube, sphere, etc).
 - Representation conversions
   - Triangulation of a Point Cloud to Mesh,
   - Mesh to Cloud Point conversion,
   - Mesh to Voxel conversion,
   - Voxel To Mesh conversion.
 - Deformations
   - Laplassian deformation,
   - Freeform deformation,
   - Relax, mesh smoothing,
   - Position Verts Smoothly, arrangement of vertices in accordance with the gradient along the neighbors.
 - Offsets
   - Mesh offset, 
   - Mesh partial offset.
 - Resolution
   - Breaking a mesh into smaller triangles (increasing the resolution)
   - Mesh decimation (decreasing the number of triangles, decreasing the resolution) with a specified error relative to the old mesh.
 - Cutting a contour on a surface. The mesh is modified so that the resulting contour contains new edges cut through triangles at the specified points.
 - Splitting
   - Splitting mesh into sub-meshes (components)
### 3D data operations
 - Boolean ops (union, intersection, difference) 
   - Boolean ops on meshes via voxels. Efficient but not so accurate as explicit mesh operations.
   - Explicit mesh boolean ops, very exact, fast and accurate.
 - Construction of Convex Hull of a point cloud or a mesh.
### 3D Data problems fixing
 - Fixing holes in mesh
   - Holes stitching (removing two holes by stitching their boundaries) 
   - Hole filling,
   - Holes fixing metrics
      - Basic set of triangulation metrics,
      - Extended set of triangulation metrics,
      - Custom triangulation metrics.
 - Delaunay triangulation optimization, changing triangles without changing vertices, according to Delaunay criterion,
 - Tunnels fixing,
 - Multiple edges detection
 - Degenerate triangles fixing,
 - Undercuts fixing, via voxels, 
 - Surface self-intersections fixing
   - Guaranteed fix via voxels, 
   - Fix via Relax (mesh smoothing).
### Functions on 3D data
 - BVH hierarchies (AABB trees) for meshes and polylines to accelerate all other operations
 - Projection
   - Projecting a point onto a mesh - closest point queries
 - Intersection
   - Intersection of a ray with a mesh (ray tracing),
   - Intersection of a plane with a mesh, result is a contour,
   - Finding a contour representing intersection of two meshes, meshes remain unchanged,
 - Distance
   - Distance from a point to a mesh,
   - Minimal distance between two meshes,
   - Mesh distance map (height map),
   - 2D contour distance map.
 - Segmentation
   - Semi-auto voxel segmentation (volumes classification).
   - Semi-auto mesh segmentation by curvature.
 - Sampling
   - Mesh sampling. The result is a separate thinned set of vertices, the mesh remains unchanged. 
   - Point cloud sampling. The result is a separate thinned set of points, the cloud remains unchanged
      - Uniform cloud sampling,
      - Grid cloud sampling.
 - Path 
   - Finding a shortest path through the mesh vertices,
   - Finding a geodesic shortest path on the surface, not necessarily through mesh vertices.
 - Iterative Closest Points (ICP), two meshes aligning
   - Point to point,
   - Point to plane.
### Upcoming features
We plan to add computed-tomography reconstruction in MeshLib (already present in MeshInspector). Please write us if you like to see some other features.

# Python integration

Although MeshLib is written in C++, all functions are exposed to Python as well. Python code can be executed both from within a MeshLib-based C++ application (embedded mode) and from python interpreter directly, which imports MeshLib modules.

Moreover, MeshLib can be easily installed as a Python 3.8 - 3.11 package using `pip install`:
* On Windows via
```
py -3 -m pip install --upgrade pip
py -3 -m pip install meshlib
```
* On Linuxes supporting [manylinux_2_31+](https://github.com/mayeut/pep600_compliance#distro-compatibility), including Ubuntu 20+ and Fedora 32+ via
```
sudo apt install python3-pip
python3 -m pip install --upgrade pip
python3 -m pip install meshlib
```
* On macOS 12.0+ via
```
pip install --upgrade pip
pip install meshlib
```


# Build
## Build with VS2019 on Windows
```sh
git clone https://github.com/MeshInspector/MeshLib.git
cd MeshLib
git submodule update --init --recursive
cd ..
```
### Preparing Third Parties
Some third parties are taken from vcpkg, while others (missing in vcpkg) are configured as git submodules.

### Vcpkg
1. Please install vcpkg, and integrate it into Visual Studio (note that vcpkg requires English language pack in Visual Studio, and vcpkg cannot be installed on FAT volumes, only on NTFS):
    ```sh
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    git checkout 2022.11.14
    .\bootstrap-vcpkg.bat
    .\vcpkg integrate install (with admin rights)
    cd ..
    ```
    More details here: [vcpkg](https://github.com/microsoft/vcpkg).

2. Execute install.bat
    ```sh
    cd vcpkg # or add vcpkg to PATH
    ../MeshLib/thirdparty/install.bat
    cd ..
    ```
3. Open solution file MeshInspector/source/MeshLib.sln in Visual Studio 2019. Build it and run.

## Build with CMake on Linux
This installation was checked on Ubuntu 20.04.4.

Use automated installation process. It takes ~40 minutes if no required packages are already installed.
This approach is useful for new MR developers
**Install/Build dependencies. Build project. Run Test Application** Run the following in terminal:

```sh
git clone https://github.com/MeshInspector/MeshLib.git
cd MeshLib
sudo ./scripts/build_thirdparty.sh # need sudo to check and install dependencies
./scripts/install_thirdparty.sh
./scripts/build_sources.sh
./scripts/distribution.sh
sudo apt install ./distr/meshlib-dev.deb
```

> **_NOTE:_** `./scripts/install*.sh` scripts could be used as well, but apt install is preferable.

> **_NOTE:_** `./scripts/install*.sh` scripts copy MR files directly to `/usr/local/lib`. Remove this directory manually if exists before apt install deb package.

> **_NOTE:_** You could specify build type to Debug by `export MESHLIB_BUILD_TYPE=Debug`. Release is default.

## Build with Emscripten on Linux
This installation was checked on Ubuntu 20.04.4 with emscripten 3.1.23.

Install Emscripten (find more on [emscripten official page](https://emscripten.org/docs/getting_started/downloads.html))
```
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
optional git pull # optional
./emsdk install 3.1.23 # (or enother version / latest)
./emsdk activate 3.1.23
source ./emsdk_env.sh
```

Build
```
cd ~/MeshInspectorCode
./scripts/build_thirdparty.sh # select emscripten in the corresponding question
./scripts/build_sorces.sh # select emscripten in the corresponding question
```

Run
```
python3 ./scripts/local_server.py
# open in browser 127.0.0.1:8000
```

# Integration

## Linux Ubuntu/Fedora
You can download [dev package](https://github.com/MeshInspector/MeshLib/releases) and install it with your Linux OS package manager. 
Examples of integration with cmake can be found in the `./examples` directory.


## MacOS
Same as for Linux, but requires brew packages to be installed:
`xargs brew install < /Library/Frameworks/MeshLib.framework/Versions/Current/scripts/macos.txt`


## Windows
There are two general options of integrating MeshLib into your project:
1. [Submodule](#submodule)
2. [Distribution](#distribution)

**Common for both options:** [install thirdparty](#vcpkg)
### Submodule
You can have MeshLib as submodule in your repository, and inculde all MeshLib's projects to your solution.
> **_NOTE:_** You should use `MeshLib/source/common.props` in other projects of your solution.

> **_NOTE:_** You can customize props by defining `CustomMRProps.props` in directory above `common.props`

### Distribution
You can download [distribution](https://github.com/MeshInspector/MeshLib/releases) and integrate it in your projects.

Project settings:
1. `C/C++ -> General -> Additional Include Directories` add `distribution\install\include;`
2. `Linker -> General -> Additional Library Directories` add `distribution\install\app\$(Configuration);`
3. `Linker -> Input -> Additional Dependencies` add `distribution\install\lib\$(Configuration)\*.lib;`
4. Debug: `C/C++ -> Preprocessor -> Preprocessor Defenitions` add `_ITERATOR_DEBUG_LEVEL=0;`
5. `vcpkg -> Triplet` set `x64-windows-meshrus`

Make sure you copy all dlls from `distribution\install\app\$(Configuration);` to your `$(TargetDir)`
> **_NOTE:_** MeshLib distribution has x64 build only

> **_NOTE:_** Distribution is build with ITERATOR_DEBUG_LEVEL=0 in debug so you will need to setup this for your projects

# Our clients and users

The following companies use MeshLib in their software
* [Smile Direct Club](https://smiledirectclub.com/)
* [Mantle](https://www.mantle3d.com/)
* Many smaller firms and start-ups
