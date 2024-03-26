#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include "MRSimpleVolume.h"
#include "MRProgressCallback.h"
#include "MRSignDetectionMode.h"
#include "MRExpected.h"
#include <climits>

namespace MR
{

// Callback type for positioning marching cubes vertices
// args: position0, position1, value0, value1, iso
using VoxelPointPositioner = std::function<Vector3f( const Vector3f&, const Vector3f&, float, float, float )>;

// linear interpolation positioner
MRMESH_API Vector3f voxelPositionerLinear( const Vector3f& pos0, const Vector3f& pos1, float v0, float v1, float iso );

struct MarchingCubesParams
{
    /// origin point of voxels box
    Vector3f origin;
    /// progress callback
    ProgressCallback cb;
    float iso{ 0.0f };
    bool lessInside{ false }; // should be false for dense volumes, and true for distance volume
    Vector<VoxelId, FaceId>* outVoxelPerFaceMap{ nullptr }; // optional output map FaceId->VoxelId
    // function to calculate position of result mesh points
    // if the function isn't set, `voxelPositionerLinear` will be used
    // note: this function is called in parallel from different threads
    VoxelPointPositioner positioner = {};
    /// if the mesh exceeds this number of vertices, an error returns
    int maxVertices = INT_MAX;
    /// for simple volumes only: omit checks for NaN values
    /// use it if you're aware that the input volume has no NaN values
    bool omitNaNCheck = false;
    /// voxel volume data caching mode
    enum class CachingMode
    {
        /// choose caching mode depending on input
        /// (current defaults: Normal for FunctionVolume, None for others)
        Automatic,
        /// don't cache any data
        None,
        /// cache some voxel volume data
        Normal,
    } cachingMode = CachingMode::Automatic;
};

// makes Mesh from SimpleVolume with given settings using Marching Cubes algorithm
MRMESH_API Expected<Mesh> marchingCubes( const SimpleVolume& volume, const MarchingCubesParams& params = {} );
MRMESH_API Expected<TriMesh> marchingCubesAsTriMesh( const SimpleVolume& volume, const MarchingCubesParams& params = {} );

#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
// makes Mesh from VdbVolume with given settings using Marching Cubes algorithm
MRMESH_API Expected<Mesh> marchingCubes( const VdbVolume& volume, const MarchingCubesParams& params = {} );
MRMESH_API Expected<TriMesh> marchingCubesAsTriMesh( const VdbVolume& volume, const MarchingCubesParams& params = {} );
#endif

// makes Mesh from FunctionVolume with given settings using Marching Cubes algorithm
MRMESH_API Expected<Mesh> marchingCubes( const FunctionVolume& volume, const MarchingCubesParams& params = {} );
MRMESH_API Expected<TriMesh> marchingCubesAsTriMesh( const FunctionVolume& volume, const MarchingCubesParams& params = {} );

} //namespace MR
