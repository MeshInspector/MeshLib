#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include "MRVoxelsVolume.h"
#include "MRProgressCallback.h"
#include "MRSignDetectionMode.h"
#include "MRExpected.h"
#include <climits>

namespace MR
{

// Callback type for positioning marching cubes vertices
// args: position0, position1, value0, value1, iso
using VoxelPointPositioner = std::function<Vector3f( const Vector3f&, const Vector3f&, float, float, float )>;

struct MarchingCubesParams
{
    /// origin point of voxels box in 3D space with output mesh
    Vector3f origin;

    /// progress callback
    ProgressCallback cb;

    /// target iso-value of the surface to be extracted from volume
    float iso{ 0.0f };

    /// should be false for dense volumes, and true for distance volume
    bool lessInside{ false };

    /// optional output map FaceId->VoxelId
    Vector<VoxelId, FaceId>* outVoxelPerFaceMap{ nullptr };

    /// function to calculate position of result mesh points
    /// if the function isn't set, a linear positioner will be used
    /// note: this function is called in parallel from different threads
    VoxelPointPositioner positioner = {};

    /// if the mesh exceeds this number of vertices, an error returns
    int maxVertices = INT_MAX;

    /// caching mode to reduce the number of accesses to voxel volume data on the first pass of the algorithm by consuming more memory on cache;
    /// note: the cache for the second pass of the algorithm (bit sets of invalid and lower-than-iso voxels are always allocated)
    enum class CachingMode
    {
        /// choose caching mode automatically depending on volume type
        /// (current defaults: Normal for FunctionVolume, None for others)
        Automatic,
        /// don't cache any data
        None,
        /// allocates 2 full slices per parallel thread
        Normal,
    } cachingMode = CachingMode::Automatic;
};

// makes Mesh from SimpleVolume with given settings using Marching Cubes algorithm
MRMESH_API Expected<Mesh> marchingCubes( const SimpleVolume& volume, const MarchingCubesParams& params = {} );
MRMESH_API Expected<TriMesh> marchingCubesAsTriMesh( const SimpleVolume& volume, const MarchingCubesParams& params = {} );

#ifndef MRMESH_NO_OPENVDB
// makes Mesh from VdbVolume with given settings using Marching Cubes algorithm
MRMESH_API Expected<Mesh> marchingCubes( const VdbVolume& volume, const MarchingCubesParams& params = {} );
MRMESH_API Expected<TriMesh> marchingCubesAsTriMesh( const VdbVolume& volume, const MarchingCubesParams& params = {} );
#endif

// makes Mesh from FunctionVolume with given settings using Marching Cubes algorithm
MRMESH_API Expected<Mesh> marchingCubes( const FunctionVolume& volume, const MarchingCubesParams& params = {} );
MRMESH_API Expected<TriMesh> marchingCubesAsTriMesh( const FunctionVolume& volume, const MarchingCubesParams& params = {} );

} //namespace MR
