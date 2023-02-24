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
    Vector3i dimensions{ 100,100,100 }; // num voxels along each axis
    float minDistSq{ 0 };       // minimum value in voxels squared
    float maxDistSq{ FLT_MAX }; // maximum value in voxels squared
    enum SignDetectionMode
    {
        Unsigned,         // unsigned voxels, useful for `Shell` offset
        ProjectionNormal, // the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections)
        WindingRule       // ray intersection counter, useful to fix self-intersections
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