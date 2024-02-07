#pragma once

#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMeshPart.h"
#include "MRProgressCallback.h"
#include "MRAffineXf3.h"
#include "MRExpected.h"
#include "MRBox.h"
#include "MRParallelFor.h"
#include "MRSimpleVolume.h"
#include "MRVolumeIndexer.h"
#include "MRVDBFloatGrid.h"
#include <climits>
#include <string>

namespace MR
{

// closed surface is required
// surfaceOffset - number voxels around surface to calculate distance in (should be positive)
// returns null if was canceled by progress callback
MRMESH_API FloatGrid meshToLevelSet( const MeshPart& mp, const AffineXf3f& xf,
                                     const Vector3f& voxelSize, float surfaceOffset = 3,
                                     ProgressCallback cb = {} );

// does not require closed surface, resulting grid cannot be used for boolean operations,
// surfaceOffset - the number of voxels around surface to calculate distance in (should be positive)
// returns null if was canceled by progress callback
MRMESH_API FloatGrid meshToDistanceField( const MeshPart& mp, const AffineXf3f& xf,
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
MRMESH_API void evalGridMinMax( const FloatGrid& grid, float& min, float& max );

// convert mesh to volume in (0,0,0)-(dim.x,dim.y,dim.z) grid box
MRMESH_API Expected<VdbVolume, std::string> meshToVolume( const Mesh& mesh, const MeshToVolumeParams& params = {} );

// fills VdbVolume data from FloatGrid (does not fill voxels size, cause we expect it outside)
MRMESH_API VdbVolume floatGridToVdbVolume( const FloatGrid& grid );

// make FloatGrid from SimpleVolume
// make copy of data
// grid can be used to make iso-surface later with gridToMesh function
MRMESH_API FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolume, ProgressCallback cb = {} );
MRMESH_API VdbVolume simpleVolumeToVdbVolume( const SimpleVolume& simpleVolume, ProgressCallback cb = {} );

// make VoxelsVolume (e.g. SimpleVolume or SimpleVolumeU16) from VdbVolume
// if VoxelsVolume values type is integral, performs mapping from [vdbVolume.min, vdbVolume.max] to
// nonnegative range of target type
template<typename T>
Expected<VoxelsVolume<std::vector<T>>, std::string> vdbVolumeToSimpleVolume(
    const VdbVolume& vdbVolume, const Box3i &activeBox = Box3i(), ProgressCallback cb = {})
{
    constexpr bool isFloat = std::is_same_v<float, T> || std::is_same_v<double, T> || std::is_same_v<long double, T>;

    VoxelsVolume<std::vector<T>> res;

    res.dims = !activeBox.valid() ? vdbVolume.dims : activeBox.size();
    Vector3i org = activeBox.valid() ? activeBox.min : Vector3i{};
    res.voxelSize = vdbVolume.voxelSize;
    if constexpr ( isFloat )
    {
        res.min = vdbVolume.min;
        res.max = vdbVolume.max;
    }
    else
    {
        res.min = 0;
        res.max = std::numeric_limits<T>::max();
    }
    [[maybe_unused]] const float oMax = float( res.max );
    [[maybe_unused]] const float k =
        vdbVolume.max > vdbVolume.min ? oMax / ( vdbVolume.max - vdbVolume.min ) : 0.0f;

    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    if ( !vdbVolume.data )
    {
        std::fill( res.data.begin(), res.data.end(), T{} );
        return res;
    }

    tbb::enumerable_thread_specific accessorPerThread( vdbVolume.data->getConstAccessor() );
    if ( !ParallelFor( size_t( 0 ), indexer.size(), [&] ( size_t i )
    {
        auto& accessor = accessorPerThread.local();
        auto coord = indexer.toPos( VoxelId( i ) );
        float value = accessor.getValue( openvdb::Coord( coord.x + org.x, coord.y + org.y, coord.z + org.z ) );
        if constexpr ( isFloat )
            res.data[i] = value;
        else
            res.data[i] = T( std::clamp( ( value - vdbVolume.min ) * k, 0.0f, oMax ) );
    }, cb ) )
        return unexpectedOperationCanceled();
    return res;
}

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
MRMESH_API Expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const GridToMeshSettings & settings );

/// converts OpenVDB Grid into mesh using Dual Marching Cubes algorithm;
/// deletes grid in the middle to reduce peak memory consumption
MRMESH_API Expected<Mesh, std::string> gridToMesh( FloatGrid&& grid, const GridToMeshSettings & settings );

[[deprecated( "use gridToMesh(..., GridToMeshSettings) instead" )]]
MRMESH_API Expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );
[[deprecated( "use gridToMesh(..., GridToMeshSettings) instead" )]]
MRMESH_API Expected<Mesh, std::string> gridToMesh( FloatGrid&& grid, const Vector3f& voxelSize,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );
[[deprecated( "use gridToMesh(..., GridToMeshSettings) instead" )]]
MRMESH_API Expected<Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );
[[deprecated( "use gridToMesh(..., GridToMeshSettings) instead" )]]
MRMESH_API Expected<Mesh, std::string> gridToMesh( VdbVolume&& vdbVolume,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );

[[deprecated( "use gridToMesh(..., GridToMeshSettings) instead" )]]
MRMESH_API Expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    int maxFaces, float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );
[[deprecated( "use gridToMesh(..., GridToMeshSettings) instead" )]]
MRMESH_API Expected<Mesh, std::string> gridToMesh( FloatGrid&& grid, const Vector3f& voxelSize,
    int maxFaces, float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );
[[deprecated( "use gridToMesh(..., GridToMeshSettings) instead" )]]
MRMESH_API Expected<Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume, int maxFaces,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );
[[deprecated( "use gridToMesh(..., GridToMeshSettings) instead" )]]
MRMESH_API Expected<Mesh, std::string> gridToMesh( VdbVolume&& vdbVolume, int maxFaces,
    float isoValue = 0.0f, float adaptivity = 0.0f, ProgressCallback cb = {} );

/// set signs for unsigned distance field grid using refMesh FastWindingNumber;
/// \param meshToGridXf defines the mapping from mesh reference from to grid reference frame
/// \param fwn defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
MRMESH_API VoidOrErrStr makeSignedWithFastWinding( FloatGrid& grid, const Vector3f& voxelSize, const Mesh& refMesh,
    const AffineXf3f& meshToGridXf = {}, std::shared_ptr<IFastWindingNumber> fwn = {}, ProgressCallback cb = {} );

// performs convention from mesh to levelSet and back with offsetA, and than same with offsetB
// allowed only for closed meshes
// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones 
//                       (curvature can be lost on high values)
/// \param fwn defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
MRMESH_API Expected<Mesh, std::string> levelSetDoubleConvertion( const MeshPart& mp, const AffineXf3f& xf,
    float voxelSize, float offsetA, float offsetB, float adaptivity = 0.0f, std::shared_ptr<IFastWindingNumber> fwn = {}, ProgressCallback cb = {} );

}
#endif
