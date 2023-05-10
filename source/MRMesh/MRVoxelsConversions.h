#pragma once
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
#include "MRMeshFwd.h"
#include "MRAffineXf3.h"
#include "MRSimpleVolume.h"
#include "MRProgressCallback.h"
#include "MRSignDetectionMode.h"
#include <optional>

namespace MR
{

struct BaseVolumeConversionParams
{
    // origin point of voxels box
    Vector3f origin;
    ProgressCallback cb{}; // progress callback
};

struct MeshToSimpleVolumeParams : public BaseVolumeConversionParams
{
    /// size of voxel on each axis
    Vector3f voxelSize{ 1.0f,1.0f,1.0f };
    /// num voxels along each axis
    Vector3i dimensions{ 100,100,100 };
    /// minimum squared value in a voxel
    float minDistSq{ 0 };
    /// maximum squared value in a voxel
    float maxDistSq{ FLT_MAX };
    /// the method to compute distance sign
    SignDetectionMode signMode{ SignDetectionMode::ProjectionNormal };

    std::shared_ptr<IFastWindingNumber> fwn;
};

// Callback type for positioning marching cubes vertices
// args: position0, position1, value0, value1, iso
using VoxelPointPositioner = std::function<Vector3f( const Vector3f&, const Vector3f&, float, float, float )>;

// linear interpolation positioner
MRMESH_API Vector3f voxelPositionerLinear( const Vector3f& pos0, const Vector3f& pos1, float v0, float v1, float iso );

struct VolumeToMeshParams : public BaseVolumeConversionParams
{
    float iso{ 0.0f };
    bool lessInside{ false }; // should be false for dense volumes, and true for distance volume
    Vector<VoxelId, FaceId>* outVoxelPerFaceMap{ nullptr }; // optional output map FaceId->VoxelId
    // function to calculate position of result mesh points
    // note: this function is called in parallel from different threads
    VoxelPointPositioner positioner = &voxelPositionerLinear;
    /// if the mesh exceeds this number of vertices, an error returns
    int maxVertices = INT_MAX;
};

// makes SimpleVolume from Mesh with given params
// returns nullopt if operation was canceled
MRMESH_API std::optional<SimpleVolume> meshToSimpleVolume( const Mesh& mesh, const MeshToSimpleVolumeParams& params = {} );

// makes Mesh from SimpleVolume with given params
// using marching cubes algorithm
// returns nullopt if operation was canceled
MRMESH_API tl::expected<Mesh, std::string> simpleVolumeToMesh( const SimpleVolume& volume, const VolumeToMeshParams& params = {} );

// makes Mesh from VdbVolume with given params
// using marching cubes algorithm
// returns nullopt if operation was canceled
MRMESH_API tl::expected<Mesh, std::string> vdbVolumeToMesh( const VdbVolume& volume, const VolumeToMeshParams& params = {} );

}
#endif