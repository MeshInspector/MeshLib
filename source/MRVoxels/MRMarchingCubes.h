#pragma once

#include "MRVoxelsFwd.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRVoxelsVolume.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRSignDetectionMode.h"
#include "MRMesh/MRExpected.h"
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
        /// (current defaults: Normal for FunctionVolume and VdbVolume, None for others)
        Automatic,
        /// don't cache any data
        None,
        /// allocates 2 full slices per parallel thread
        Normal,
    } cachingMode = CachingMode::Automatic;

    /// this optional function is called when volume is no longer needed to deallocate it and reduce peak memory consumption
    std::function<void()> freeVolume;

    // To allow passing Python lambdas into `positioner`.
    MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM
};

// makes Mesh from SimpleVolume with given settings using Marching Cubes algorithm
MRVOXELS_API Expected<Mesh> marchingCubes( const SimpleVolume& volume, const MarchingCubesParams& params = {} );
MRVOXELS_API Expected<TriMesh> marchingCubesAsTriMesh( const SimpleVolume& volume, const MarchingCubesParams& params = {} );

// makes Mesh from SimpleVolumeMinMax with given settings using Marching Cubes algorithm
MRVOXELS_API Expected<Mesh> marchingCubes( const SimpleVolumeMinMax& volume, const MarchingCubesParams& params = {} );
MRVOXELS_API Expected<TriMesh> marchingCubesAsTriMesh( const SimpleVolumeMinMax& volume, const MarchingCubesParams& params = {} );

// makes Mesh from VdbVolume with given settings using Marching Cubes algorithm
MRVOXELS_API Expected<Mesh> marchingCubes( const VdbVolume& volume, const MarchingCubesParams& params = {} );
MRVOXELS_API Expected<TriMesh> marchingCubesAsTriMesh( const VdbVolume& volume, const MarchingCubesParams& params = {} );

// makes Mesh from FunctionVolume with given settings using Marching Cubes algorithm
MRVOXELS_API Expected<Mesh> marchingCubes( const FunctionVolume& volume, const MarchingCubesParams& params = {} );
MRVOXELS_API Expected<TriMesh> marchingCubesAsTriMesh( const FunctionVolume& volume, const MarchingCubesParams& params = {} );

// makes Mesh from SimpleBinaryVolume with given settings using Marching Cubes algorithm
MRVOXELS_API Expected<Mesh> marchingCubes( const SimpleBinaryVolume& volume, const MarchingCubesParams& params = {} );
MRVOXELS_API Expected<TriMesh> marchingCubesAsTriMesh( const SimpleBinaryVolume& volume, const MarchingCubesParams& params = {} );

/// converts volume split on parts by planes z=const into mesh,
/// last z-layer of previous part must be repeated as first z-layer of next part
/// usage:
/// MarchingCubesByParts x( dims, params);
/// x.addPart( part1 );
/// ...
/// x.addPart( partN );
/// Mesh mesh = Mesh::fromTriMesh( *x.finalize() );
class MarchingCubesByParts
{
public:
    /// prepares convention for given volume dimensions and given parameters
    /// \param layersPerBlock all z-slices of the volume will be partitioned on blocks of given size to process blocks in parallel (0 means auto-select layersPerBlock)
    MRVOXELS_API explicit MarchingCubesByParts( const Vector3i & dims, const MarchingCubesParams& params, int layersPerBlock = 0 );

    MRVOXELS_API ~MarchingCubesByParts();
    MRVOXELS_API MarchingCubesByParts( MarchingCubesByParts && s ) noexcept;
    MRVOXELS_API MarchingCubesByParts & operator=( MarchingCubesByParts && s ) noexcept;

    /// the number of z-slices of the volume in the blocks
    MRVOXELS_API int layersPerBlock() const;

    /// the last z-layer of the previous part and the first z-layer of the next part
    MRVOXELS_API int nextZ() const;

    /// adds one more part of volume into consideration, with first z=nextZ()
    MRVOXELS_API Expected<void> addPart( const SimpleVolume& part );

    /// finishes processing and outputs produced trimesh
    MRVOXELS_API Expected<TriMesh> finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} //namespace MR
