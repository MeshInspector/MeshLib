#pragma once
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include "MRSimpleVolume.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{

struct BaseVolumeConversionParams
{
    AffineXf3f basis; // position of lowest left voxel, and axes vectors (A.transposed().x == x axis of volume)
    ProgressCallback cb{}; // progress callback
};

struct MeshToSimpleVolumeParams : BaseVolumeConversionParams
{
    /// num voxels along each axis
    Vector3i dimensions{ 100,100,100 };
    /// minimum squared value in a voxel
    float minDistSq{ 0 };
    /// maximum squared value in a voxel
    float maxDistSq{ FLT_MAX };
    /// the method to compute distance sign
    enum SignDetectionMode
    {
        Unsigned,         // unsigned voxels, useful for `Shell` offset
        ProjectionNormal, // the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
        WindingRule,      // ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh
        HoleWindingRule   // computes winding number generalization with support of holes in mesh, slower than WindingRule
    } signMode{ ProjectionNormal };
};

struct VdbVolumeToMeshParams : BaseVolumeConversionParams
{
    float iso{ 0.0f };
    bool lessInside{ false }; // should be false for dense volumes, and true for distance volume
    Vector<VoxelId, FaceId>* outVoxelPerFaceMap{ nullptr }; // optional output map FaceId->VoxelId
};

// makes SimpleVolume from Mesh with given params
// returns nullopt if operation was canceled
MRMESH_API std::optional<SimpleVolume> meshToSimpleVolume( const Mesh& mesh, const MeshToSimpleVolumeParams& params = {} );

// makes Mesh from VdbVolume with given params
// returns nullopt if operation was canceled
MRMESH_API std::optional<Mesh> vdbVolumeToMesh( const VdbVolume& volume, const VdbVolumeToMeshParams& params = {} );

}
#endif