[![build-test-distribute](https://github.com/MeshRUs/MeshLib/actions/workflows/build-test-distribute.yml/badge.svg?branch=master)](https://github.com/MeshRUs/MeshLib/actions/workflows/build-test-distribute.yml?branch=master)

# Welcome to the MeshLib!
3D scanning is becoming more and more ubiquitous. Robotic automation, self-driving cars and multitude of other industrial, medical and scientific applications require advanced computer vision to deliver the levels of automation customers expect these days. The great rise of AI gave another boost to computer vision and the need to utilize 3D data to make machines smarter. Not only are tasks at hand becoming more complex, but the size of data grows exponentially. 

There is a multitude of general purpose libraries which deal with 3D data. Some stem from popular CAD packages, some are open source. The commercial ones tend to be quite expensive while open source are free though tend to be limited in functionality provided. Also those libraries value generality above other features  to allow maximum number of applications, but with the growing amounts of 3D data, performance is critical as it never has  been. Some of it can be addressed by using the scale of a commercial cloud, last generation CPU or GPU but underlying complexity of data representation makes it very hard and laborsome.

The goal which we set when designing MeshRus was to value simplicity and performance while providing a wide gamut of useful computational algorithms. The library also supports the most important data structures today’s sensors can produce - pointcloud, mesh, volume and more. For example, mesh is represented by half-edge data structure and cannot be made non-manifold. Some applications may require non-manifoldness, but most practical scans can be represented as manifoldness meshes without an issue. 

## Some features
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
### 3D Data problems fixing
 - Fixing holes in mesh
   - Holes stitching (removing two holes by stitching their boundaries) 
   - Hole filling,
   - Holes fixing metrics
      - Basic set of triangulation metrics,
      - Extended set of triangulation metrics,
      - Custom triangulation metrics.
 - Delone triangulation optimization, changing triangles without changing vertices, according to Delone criterion,
 - Tunnels fixing,
 - Multiple edges detection
 - Degenerate triangles fixing,
 - Undercats fixing, via voxels, 
 - Surface self-intersections fixing
   - Guaranteed fix via voxels, 
   - Fix via Relax (mesh smoothing).
### Functions on 3D data
 - Projection
   - Projecting a point onto a mesh
 - Intersection
   - Intersection of a ray with a mesh,
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

## Some useful links
[MeshInspector releases page](https://github.com/MeshRUs/MeshInspectorReleases/releases)

[Our YouTube channel with updates](https://www.youtube.com/channel/UCv9bNhwoVDPaLPPyWJeVPNg)

[Documentation](https://meshrus.github.io/)


# Build
## Build with VS2019 on Windows
```sh
git clone https://github.com/MeshRUs/MeshLib.git
cd MeshLib
git submodule update --init --recursive
```
### Preparing Third Parties
Some third parties are taken from vcpkg, while others (missing in vcpkg) are configured as git submodules.

### Vcpkg
1. Please install vcpkg, and integrate it into Visual Studio (note that vcpkg requires English laguage pack installed in Visual Studio):
    ```sh
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    git checkout 5c54cc06554e450829d72013c4b9e4baae41529a
    .\bootstrap-vcpkg.bat
    .\vcpkg integrate install (with admin rights)
    ```
    More details here: [vcpkg](https://github.com/microsoft/vcpkg).

2. Copy **thirdparty/vcpkg/triplets/x64-windows-meshrus.cmake** to **vcpkg/triplets** folder of vcpkg installation.
3. Execute install.bat
    ```sh
    cd vcpkg # or add vcpcg to PATH
    <path_to_MeshLib>/thirdparty/install.bat
    ```    
## Build with CMake on Linux
This installation was checked on Ubuntu 20.04.4.

Use automated installation process. It takes ~40 minutes if no required packages are already installed.
This approach is useful for new MR developers
**Install/Build dependencies. Build project. Run Test Application** Run the following in terminal:

```sh
git clone https://github.com/MeshRUs/MeshLib.git
cd MeshLib
sudo ./scripts/build_thirdparty.sh # need sudo to check and install dependencies
./scripts/install_thirdparty.sh
./scripts/build_sources.sh
./scripts/distribution.sh
sudo apt install ./distr/meshrus-dev.deb
```

Note! ./scripts/install*.sh scripts could be used as well, but apt install is prefferable.
Note! ./scripts/install*.sh scripts copy MR files directly to /usr/local/lib. Remove this directory manually if exists before apt install deb package
Note! You could specify build type to Debug by ```export MESHRUS_BUILD_TYPE=Debug```. Release is default.

