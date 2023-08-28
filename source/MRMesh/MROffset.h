#pragma once
#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMeshPart.h"
#include "MRSignDetectionMode.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <optional>
#include <string>

namespace MR
{

struct BaseOffsetParameters
{
    /// Size of voxel in grid conversions
    /// if value is negative, it is calculated automatically (mesh bounding box are divided to 5e6 voxels)
    float voxelSize{ -1.0f };

    /// if not nullopt then SimpleVolume will be used for offsetting,
    /// and this value will determine the method to compute distance sign
    std::optional<SignDetectionMode> simpleVolumeSignMode;

    /// Progress callback 
    ProgressCallback callBack{};
    /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
    std::shared_ptr<IFastWindingNumber> fwn;
};

/// This struct represents parameters for offsetting with voxels conversions
struct OffsetParameters : BaseOffsetParameters
{
    /// Decimation ratio of result mesh [0..1], this is applied on conversion from voxels to mesh
    /// note: it does not work good, better use common decimation after offsetting
    float adaptivity{0.0f};

    /// Type of offsetting
    enum class Type
    {
        Offset, /// can be positive or negative, input mesh should be closed
        Shell   /// can be only positive, offset in both directions of surface
    } type{ Type::Offset };
};

struct SharpOffsetParameters : BaseOffsetParameters
{
    /// if non-null then created sharp edges will be saved here
    UndirectedEdgeBitSet* outSharpEdges = nullptr;
    /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
    float minNewVertDev = 1.0f / 25;
    /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
    float maxNewRank2VertDev = 5;
    /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
    float maxNewRank3VertDev = 2;
    /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
    /// big correction can be wrong and result from self-intersections in the reference mesh
    float maxOldVertPosCorrection = 0.5f;
};

/// Offsets mesh by converting it to voxels and back
/// so result mesh is always closed
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> offsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params = {} );

/// in case of positive offset, returns the mesh consisting of offset mesh merged with inversed original mesh (thickening mode);
/// in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode);
/// if your input mesh is closed then please specify params.type == Offset, and you will get closed mesh on output;
/// if your input mesh is open then please specify params.type == Shell, and you will get open mesh on output
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> thickenMesh( const Mesh& mesh, float offset, const OffsetParameters & params = {} );

/// Offsets mesh by converting it to voxels and back two times
/// only closed meshes allowed (only Offset mode)
/// typically offsetA and offsetB have distinct signs
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params = {} );

/// Offsets mesh by converting it to voxels and back using standard Marching Cubes, as opposed to Dual Marching Cubes in offsetMesh(...)
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> mcOffsetMesh( const Mesh& mesh, float offset, 
    const BaseOffsetParameters& params = {}, Vector<VoxelId, FaceId>* outMap = nullptr );

/// Offsets mesh by converting it to voxels and back
/// post process result using reference mesh to sharpen features
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> sharpOffsetMesh( const Mesh& mesh, float offset, const SharpOffsetParameters& params = {} );

/// Offsets polyline by converting it to voxels and building iso-surface
/// do offset in all directions
/// so result mesh is always closed
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params = {} );

}
#endif
