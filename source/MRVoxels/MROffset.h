#pragma once
#include "MRVoxelsFwd.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRSignDetectionMode.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRPartMapping.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MREnums.h"
#include <optional>
#include <string>

namespace MR
{

struct BaseShellParameters
{
    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// Progress callback
    ProgressCallback callBack;
};

/// computes size of a cubical voxel to get approximately given number of voxels during rasterization
[[nodiscard]] MRVOXELS_API float suggestVoxelSize( const MeshPart & mp, float approxNumVoxels );

struct OffsetParameters : BaseShellParameters
{
    /// determines the method to compute distance sign
    SignDetectionMode signDetectionMode = SignDetectionMode::OpenVDB;

    /// whether to construct closed mesh in signMode = SignDetectionModeShort::HoleWindingNumber
    bool closeHolesInHoleWindingNumber = true;

    /// only for SignDetectionMode::HoleWindingRule:
    /// positive distance if winding number below or equal this threshold;
    /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
    float windingNumberThreshold = 0.5f;

    /// only for SignDetectionMode::HoleWindingRule:
    /// determines the precision of fast approximation: the more the better, minimum value is 1
    float windingNumberBeta = 2;

    /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
    /// this only used if signDetectionMode == SignDetectionMode::HoleWindingRule, otherwise it is ignored
    /// providing this will disable memoryEfficient (as if memoryEfficient == false)
    std::shared_ptr<IFastWindingNumber> fwn;

    /// use FunctionVolume for voxel grid representation:
    ///  - memory consumption for voxel storage is approx. (dims.z / (2 * thread_count)) lesser
    ///  - computations are about 15% slower (because some z-layers are computed twice)
    /// this setting is ignored (as if memoryEfficient == false) if
    ///  a) signDetectionMode = SignDetectionMode::OpenVDB, or
    ///  b) \ref fwn is provided (CUDA computations require full memory storage)
    /// used only by \ref mcOffsetMesh and \ref sharpOffsetMesh methods
    bool memoryEfficient = true;
};

struct SharpOffsetParameters : OffsetParameters
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

/// Offsets mesh by converting it to distance field in voxels using OpenVDB library,
/// signDetectionMode = Unsigned(from OpenVDB) | OpenVDB | HoleWindingRule,
/// and then converts back using OpenVDB library (dual marching cubes),
/// so result mesh is always closed
[[nodiscard]] MRVOXELS_API Expected<Mesh> offsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params = {} );

/// Offsets mesh by converting it to voxels and back two times
/// only closed meshes allowed (only Offset mode)
/// typically offsetA and offsetB have distinct signs
[[nodiscard]] MRVOXELS_API Expected<Mesh> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params = {} );

/// Offsets mesh by converting it to distance field in voxels (using OpenVDB library if SignDetectionMode::OpenVDB or our implementation otherwise)
/// and back using standard Marching Cubes, as opposed to Dual Marching Cubes in offsetMesh(...)
[[nodiscard]] MRVOXELS_API Expected<Mesh> mcOffsetMesh( const MeshPart& mp, float offset,
    const OffsetParameters& params = {}, Vector<VoxelId, FaceId>* outMap = nullptr );

/// Constructs a shell around selected mesh region with the properties that every point on the shall must
///  1. be located not further than given distance from selected mesh part,
///  2. be located not closer to not-selected mesh part than to selected mesh part.
[[nodiscard]] MRVOXELS_API Expected<Mesh> mcShellMeshRegion( const Mesh& mesh, const FaceBitSet& region, float offset,
    const BaseShellParameters& params, Vector<VoxelId, FaceId> * outMap = nullptr );

/// Offsets mesh by converting it to voxels and back
/// post process result using reference mesh to sharpen features
[[nodiscard]] MRVOXELS_API Expected<Mesh> sharpOffsetMesh( const MeshPart& mp, float offset, const SharpOffsetParameters& params = {} );

/// allows the user to select in the parameters which offset algorithm to call
struct GeneralOffsetParameters : SharpOffsetParameters
{
    using Mode = MR::OffsetMode;
    Mode mode = Mode::Standard;
};

/// Offsets mesh by converting it to voxels and back using one of three modes specified in the parameters
/// \snippet cpp-examples/MeshOffset.dox.cpp 0
[[nodiscard]] MRVOXELS_API Expected<Mesh> generalOffsetMesh( const MeshPart& mp, float offset, const GeneralOffsetParameters& params );

/// in case of positive offset, returns the mesh consisting of offset mesh merged with inversed original mesh (thickening mode);
/// in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode);
/// if your input mesh is open then please specify params.signDetectionMode = SignDetectionMode::Unsigned, and you will get open mesh (with several components) on output
/// if your input mesh is closed then please specify another sign detection mode, and you will get closed mesh (with several components) on output;
[[nodiscard]] MRVOXELS_API Expected<Mesh> thickenMesh( const Mesh& mesh, float offset, const GeneralOffsetParameters & params = {},
    const PartMapping & map = {} ); ///< mapping between original mesh and thicken result

/// offsets given MeshPart in one direction only (positive or negative)
/// if your input mesh is open then please specify params.signDetectionMode = SignDetectionMode::Unsigned
/// if your input mesh is closed this function is equivalent to `generalOffsetMesh`, but in SignDetectionMode::Unsigned mode it will only keep one side (just like for open mesh)
/// unlike `thickenMesh` this functions does not keep original mesh in result
[[nodiscard]] MRVOXELS_API Expected<Mesh> offsetOneDirection( const MeshPart& mp, float offset, const GeneralOffsetParameters& params = {} );

/// Offsets polyline by converting it to voxels and building iso-surface
/// do offset in all directions
/// so result mesh is always closed
/// params.signDetectionMode is ignored (always assumed SignDetectionMode::Unsigned)
[[nodiscard]] MRVOXELS_API Expected<Mesh> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params = {} );

}
