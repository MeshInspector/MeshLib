#pragma once

#include "MRVoxelsFwd.h"

#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRBox.h"
#include <climits>
#include <string>
#include <optional>

namespace MR
{

// closed surface is required
// surfaceOffset - number voxels around surface to calculate distance in (should be positive)
// returns null if was canceled by progress callback
MRVOXELS_API FloatGrid meshToLevelSet( const MeshPart& mp, const AffineXf3f& xf,
                                     const Vector3f& voxelSize, float surfaceOffset = 3,
                                     ProgressCallback cb = {} );

// does not require closed surface, resulting grid cannot be used for boolean operations,
// surfaceOffset - the number of voxels around surface to calculate distance in (should be positive)
// returns null if was canceled by progress callback
MRVOXELS_API FloatGrid meshToDistanceField( const MeshPart& mp, const AffineXf3f& xf,
                                          const Vector3f& voxelSize, float surfaceOffset = 3,
                                          ProgressCallback cb = {} );

// Parameters structure for meshToVolume function
struct MeshToVolumeParams
{
    // Conversion type
    enum class Type
    {
        Signed, // only closed meshes can be converted with signed type
        Unsigned // this type leads to shell like iso-surfaces
    } type{ Type::Unsigned };
    float surfaceOffset{ 3.0 }; // the number of voxels around surface to calculate distance in (should be positive)
    Vector3f voxelSize = Vector3f::diagonal( 1.0f );
    AffineXf3f worldXf; // mesh initial transform
    AffineXf3f* outXf{ nullptr }; // optional output: xf to original mesh (respecting worldXf)
    ProgressCallback cb;
};

// eval min max value from FloatGrid
MRVOXELS_API void evalGridMinMax( const FloatGrid& grid, float& min, float& max );

// convert mesh to volume in (0,0,0)-(dim.x,dim.y,dim.z) grid box
MRVOXELS_API Expected<VdbVolume> meshToVolume( const Mesh& mesh, const MeshToVolumeParams& params = {} );

// fills VdbVolume data from FloatGrid (does not fill voxels size, cause we expect it outside)
MRVOXELS_API VdbVolume floatGridToVdbVolume( FloatGrid grid );

// make FloatGrid from SimpleVolume
// make copy of data
// background - the new background value for FloatGrid
// grid can be used to make iso-surface later with gridToMesh function
MRVOXELS_API FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolume, float background = 0.0f, ProgressCallback cb = {} );
// set the simpleVolume.min as the background value
MRVOXELS_API VdbVolume simpleVolumeToVdbVolume( const SimpleVolumeMinMax& simpleVolume, ProgressCallback cb = {} );

/// @brief Copy given \p simpleVolume into the \p grid, starting at \p minCoord
/// Instantiated for AccessorOrGrid in { openvdb::FloatGrid::Accessor, FloatGrid, openvdb::FloatGrid }.
/// The template is used to not include openvdb's mess into this header (forward declaring classes in openvdb is also non-trivial).
/// When used with a Grid, multithreaded implementation of copying is used (so the function is not thread safe).
/// When used with an Accessor, this function could be called from different threads on the same volume (provided that accessors are different, of course).
template <typename AccessorOrGrid>
MRVOXELS_API void putSimpleVolumeInDenseGrid(
        AccessorOrGrid& gridAccessor,
        const Vector3i& minCoord, const SimpleVolume& simpleVolume, ProgressCallback cb = {}
    );

/// Make \p volume dense without setting any values
MRVOXELS_API void makeVdbTopologyDense( VdbVolume& volume );

// make SimpleVolume from VdbVolume
// make copy of data
MRVOXELS_API Expected<SimpleVolumeMinMax> vdbVolumeToSimpleVolume(
    const VdbVolume& vdbVolume, const Box3i& activeBox = Box3i(), ProgressCallback cb = {} );
/// Makes normalized SimpleVolume from VdbVolume
/// Normalisation consist of scaling values linearly from the source scale to the interval [0;1]
/// @note Makes copy of data
/// @param sourceScale if specified, defines the initial scale of voxels.
///     If not specified, it is estimated as min. and max. values from the voxels
MRVOXELS_API Expected<SimpleVolumeMinMax> vdbVolumeToSimpleVolumeNorm(
    const VdbVolume& vdbVolume, const Box3i& activeBox = Box3i(), std::optional<MinMaxf> sourceScale = {}, ProgressCallback cb = {} );
/// Makes SimpleVolumeU16 from VdbVolume
/// Values are linearly scaled from the source scale to the range corresponding to uint16_t
/// @note Makes copy of data
/// @param sourceScale if specified, defines the initial scale of voxels.
///     If not specified, it is estimated as min. and max. values from the voxels
MRVOXELS_API Expected<SimpleVolumeMinMaxU16> vdbVolumeToSimpleVolumeU16(
    const VdbVolume& vdbVolume, const Box3i& activeBox = Box3i(), std::optional<MinMaxf> sourceScale = {}, ProgressCallback cb = {} );

/// parameters of OpenVDB Grid to Mesh conversion using Dual Marching Cubes algorithm
struct GridToMeshSettings
{
    /// the size of each voxel in the grid
    Vector3f voxelSize;
    /// layer of grid with this value would be converted in mesh; isoValue can be negative only in level set grids
    float isoValue = 0;
    /// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones (curvature can be lost on high values)
    float adaptivity = 0;
    /// if the mesh exceeds this number of faces, an error returns
    int maxFaces = INT_MAX;
    /// if the mesh exceeds this number of vertices, an error returns
    int maxVertices = INT_MAX;
    bool relaxDisorientedTriangles = true;
    /// to receive progress and request cancellation
    ProgressCallback cb;
};

/// converts OpenVDB Grid into mesh using Dual Marching Cubes algorithm
MRVOXELS_API Expected<Mesh> gridToMesh( const FloatGrid& grid, const GridToMeshSettings & settings );

/// converts OpenVDB Grid into mesh using Dual Marching Cubes algorithm;
/// deletes grid in the middle to reduce peak memory consumption
MRVOXELS_API Expected<Mesh> gridToMesh( FloatGrid&& grid, const GridToMeshSettings & settings );

struct MakeSignedByWindingNumberSettings
{
    /// defines the mapping from mesh reference from to grid reference frame
    AffineXf3f meshToGridXf;

    /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
    std::shared_ptr<IFastWindingNumber> fwn;

    /// positive distance if winding number below or equal this threshold;
    /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
    float windingNumberThreshold = 0.5f;

    /// determines the precision of fast approximation: the more the better, minimum value is 1
    float windingNumberBeta = 2;

    /// to report algorithm's progress and to cancel it
    ProgressCallback progress;
};

/// set signs for unsigned distance field grid using generalized winding number computed at voxel grid point from refMesh
MRVOXELS_API Expected<void> makeSignedByWindingNumber( FloatGrid& grid, const Vector3f& voxelSize, const Mesh& refMesh,
    const MakeSignedByWindingNumberSettings & settings );

struct DoubleOffsetSettings
{
    /// the size of voxel in intermediate voxel grid representation
    float voxelSize = 0;

    /// the amount of first offset
    float offsetA = 0;

    /// the amount of second offset
    float offsetB = 0;

    /// in [0; 1] - ratio of combining small triangles into bigger ones (curvature can be lost on high values)
    float adaptivity = 0;

    /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
    std::shared_ptr<IFastWindingNumber> fwn;

    /// positive distance if winding number below or equal this threshold;
    /// ideal threshold: 0.5 for closed meshes; 0.0 for planar meshes
    float windingNumberThreshold = 0.5f;

    /// determines the precision of fast approximation: the more the better, minimum value is 1
    float windingNumberBeta = 2;

    /// to report algorithm's progress and to cancel it
    ProgressCallback progress;
};

/// performs convention from mesh to voxel grid and back with offsetA, and than same with offsetB;
/// if input mesh is not closed then the sign of distance field will be obtained using generalized winding number computation
MRVOXELS_API Expected<Mesh> doubleOffsetVdb( const MeshPart& mp, const DoubleOffsetSettings & settings );

} //namespace MR
