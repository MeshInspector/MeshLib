#include "MRVDBConversions.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVDBFloatGrid.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRPch/MROpenvdb.h"
#include "MRPch/MRSpdlog.h"
#include "MRBox.h"
#include "MRFastWindingNumber.h"
#include "MRVolumeIndexer.h"
#include "MRRegionBoundary.h"
#include "MRParallelFor.h"
#include "MRVDBProgressInterrupter.h"

namespace MR
{
constexpr float denseVolumeToGridTolerance = 1e-6f;

void convertToVDMMesh( const MeshPart& mp, const AffineXf3f& xf, const Vector3f& voxelSize,
                       std::vector<openvdb::Vec3s>& points, std::vector<openvdb::Vec3I>& tris )
{
    MR_TIMER
        const auto& pointsRef = mp.mesh.points;
    const auto& topology = mp.mesh.topology;
    points.resize( pointsRef.size() );
    tris.resize( mp.region ? mp.region->count() : topology.numValidFaces() );

    int i = 0;
    VertId v[3];
    for ( FaceId f : topology.getFaceIds( mp.region ) )
    {
        if ( mp.region && !topology.hasFace( f ) )
            continue; // f is in given region but not in mesh topology
        topology.getTriVerts( f, v );
        tris[i++] = openvdb::Vec3I{ ( uint32_t )v[0], ( uint32_t )v[1], ( uint32_t )v[2] };
    }
    i = 0;
    for ( const auto& p0 : pointsRef )
    {
        auto p = xf( p0 );
        points[i][0] = p[0] / voxelSize[0];
        points[i][1] = p[1] / voxelSize[1];
        points[i][2] = p[2] / voxelSize[2];
        ++i;
    }
}

template<typename GridType>
VoidOrErrStr gridToPointsAndTris(
    const GridType& grid,
    VertCoords & points, Triangulation & t,
    const GridToMeshSettings & settings )
{
    MR_TIMER

    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    openvdb::tools::VolumeToMesh mesher( settings.isoValue, settings.adaptivity, settings.relaxDisorientedTriangles );
    mesher(grid);

    if ( !reportProgress( settings.cb, 0.7f ) )
        return unexpectedOperationCanceled();

    if ( mesher.pointListSize() > settings.maxVertices )
        return unexpected( "Vertices number limit exceeded." );

    // Preallocate the point list
    points.clear();
    points.resize(mesher.pointListSize());

    // Copy points
    auto & inPts = mesher.pointList();
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, points.size() ), [&]( const tbb::blocked_range<size_t> & range )
    {
        for ( auto i = range.begin(); i < range.end(); ++i )
        {
            auto inPt = inPts[i];
            points[ VertId{ i } ] = Vector3f{
                inPt.x() * settings.voxelSize.x,
                inPt.y() * settings.voxelSize.y,
                inPt.z() * settings.voxelSize.z };
        }
    } );
    inPts.reset(nullptr);

    if ( !reportProgress( settings.cb, 0.8f ) )
        return unexpectedOperationCanceled();

    auto& polygonPoolList = mesher.polygonPoolList();

    // Preallocate primitive lists
    size_t numQuads = 0, numTriangles = 0;
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
        numTriangles += polygons.numTriangles();
        numQuads += polygons.numQuads();
    }

    const size_t tNum = numTriangles + 2 * numQuads;

    if ( tNum > settings.maxFaces )
        return unexpected( "Triangles number limit exceeded." );

    t.clear();
    t.reserve( tNum );

    // Copy primitives
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) 
    {
        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];

        for ( size_t i = 0, I = polygons.numQuads(); i < I; ++i )
        {
            auto quad = polygons.quad(i);

            ThreeVertIds newTri
            {
                VertId( ( int )quad[2] ),
                VertId( ( int )quad[1] ),
                VertId( ( int )quad[0] ),
            };
            t.push_back( newTri );

            newTri =
            {
                VertId( ( int )quad[0] ),
                VertId( ( int )quad[3] ),
                VertId( ( int )quad[2] ),
            };
            t.push_back( newTri );
        }

        for ( size_t i = 0, I = polygons.numTriangles(); i < I; ++i )
        {
            auto tri = polygons.triangle(i);

            ThreeVertIds newTri
            {
                VertId( ( int )tri[2] ),
                VertId( ( int )tri[1] ),
                VertId( ( int )tri[0] )
            };
            t.push_back( newTri );
        }
    }

    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return {};
}

FloatGrid meshToLevelSet( const MeshPart& mp, const AffineXf3f& xf,
                          const Vector3f& voxelSize, float surfaceOffset,
                          ProgressCallback cb )
{
    MR_TIMER;
    if ( surfaceOffset <= 0.0f )
    {
        assert( false );
        return {};
    }
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;

    convertToVDMMesh( mp, xf, voxelSize, points, tris );

    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    ProgressInterrupter interrupter( cb );
    auto resGrid = MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter, *xform, points, tris, surfaceOffset ) );
    if ( interrupter.getWasInterrupted() )
        return {};
    return resGrid;
}

FloatGrid meshToDistanceField( const MeshPart& mp, const AffineXf3f& xf,
    const Vector3f& voxelSize, float surfaceOffset /*= 3 */,
    ProgressCallback cb )
{
    MR_TIMER;
    if ( surfaceOffset <= 0.0f )
    {
        assert( false );
        return {};
    }
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;

    convertToVDMMesh( mp, xf, voxelSize, points, tris );

    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    ProgressInterrupter interrupter( cb );

    auto resGrid = MakeFloatGrid( openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter, *xform, points, tris, {}, surfaceOffset ) );

    if ( interrupter.getWasInterrupted() )
        return {};
    return resGrid;
}

void evalGridMinMax( const FloatGrid& grid, float& min, float& max )
{
    if ( !grid )
        return;
#if (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER >= 9 && (OPENVDB_LIBRARY_MINOR_VERSION_NUMBER >= 1 || OPENVDB_LIBRARY_PATCH_VERSION_NUMBER >= 1)) || \
    (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER >= 10)
    auto minMax = openvdb::tools::minMax( grid->tree() );
    min = minMax.min();
    max = minMax.max();
#else
    grid->evalMinMax( min, max );
#endif
}

Expected<VdbVolume, std::string> meshToVolume( const Mesh& mesh, const MeshToVolumeParams& params /*= {} */ )
{
    if ( params.type == MeshToVolumeParams::Type::Signed && !mesh.topology.isClosed() )
        return unexpected( "Only closed mesh can be converted to signed volume" );

    auto shift = AffineXf3f::translation( mesh.computeBoundingBox( &params.worldXf ).min - params.surfaceOffset * params.voxelSize );
    FloatGrid grid;
    if ( params.type == MeshToVolumeParams::Type::Signed )
        grid = meshToLevelSet( mesh, shift.inverse() * params.worldXf, params.voxelSize, params.surfaceOffset, params.cb );
    else
        grid = meshToDistanceField( mesh, shift.inverse() * params.worldXf, params.voxelSize, params.surfaceOffset, params.cb );

    if ( !grid )
        return unexpected( "Operation canceled" );

    // to get proper normal orientation both for signed and unsigned cases
    grid->setGridClass( openvdb::GRID_LEVEL_SET );

    if ( params.outXf )
        *params.outXf = shift;

    VdbVolume res;
    res.data = grid;
    evalGridMinMax( grid, res.min, res.max );
    auto dim = grid->evalActiveVoxelBoundingBox().extents();
    res.dims = Vector3i( dim.x(), dim.y(), dim.z() );
    res.voxelSize = params.voxelSize;

    return res;
}

VdbVolume floatGridToVdbVolume( const FloatGrid& grid )
{
    if ( !grid )
        return {};
    VdbVolume res;
    res.data = grid;
    evalGridMinMax( grid, res.min, res.max );
    auto dim = grid->evalActiveVoxelDim();
    res.dims = Vector3i( dim.x(), dim.y(), dim.z() );
    return res;
}

FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolume,
                                   ProgressCallback cb )
{
    MR_TIMER;
    if ( cb )
        cb( 0.0f );
    openvdb::math::Coord minCoord( 0, 0, 0 );
    openvdb::math::Coord dimsCoord( simpleVolume.dims.x, simpleVolume.dims.y, simpleVolume.dims.z );
    openvdb::math::CoordBBox denseBBox( minCoord, minCoord + dimsCoord.offsetBy( -1 ) );
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense( denseBBox, const_cast< float* >( simpleVolume.data.data() ) );
    if ( cb )
        cb( 0.5f );
    std::shared_ptr<openvdb::FloatGrid> grid = std::make_shared<openvdb::FloatGrid>( FLT_MAX );
    openvdb::tools::copyFromDense( dense, *grid, denseVolumeToGridTolerance );
    openvdb::tools::changeBackground( grid->tree(), 0.f );
    if ( cb )
        cb( 1.0f );
    return MakeFloatGrid( std::move( grid ) );
}

VdbVolume simpleVolumeToVdbVolume( const SimpleVolume& simpleVolume, ProgressCallback cb /*= {} */ )
{
    VdbVolume res;
    res.data = simpleVolumeToDenseGrid( simpleVolume, cb );
    res.dims = simpleVolume.dims;
    res.voxelSize = simpleVolume.voxelSize;
    return res;
}

Expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const GridToMeshSettings & settings )
{
    MR_TIMER;
    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    VertCoords pts;
    Triangulation t;
    {
        auto s = settings;
        s.cb = subprogress( settings.cb, 0.0f, 0.2f );
        if ( auto x = gridToPointsAndTris( *grid, pts, t, s ); !x )
            return unexpected( std::move( x.error() ) );
    }

    if ( !reportProgress( settings.cb, 0.2f ) )
        return unexpectedOperationCanceled();

    Mesh res = Mesh::fromTriangles( std::move( pts ), t, {}, subprogress( settings.cb, 0.2f, 1.0f ) );
    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

Expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    int maxFaces, float offsetVoxels, float adaptivity, ProgressCallback cb )
{
    return gridToMesh( grid, GridToMeshSettings{
        .voxelSize = voxelSize,
        .isoValue = offsetVoxels,
        .adaptivity = adaptivity,
        .maxFaces = maxFaces,
        .cb = cb
    } );
}

Expected<Mesh, std::string> gridToMesh( FloatGrid&& grid, const GridToMeshSettings & settings )
{
    MR_TIMER;
    if ( !reportProgress( settings.cb, 0.0f ) )
        return unexpectedOperationCanceled();

    VertCoords pts;
    Triangulation t;
    {
        auto s = settings;
        s.cb = subprogress( settings.cb, 0.0f, 0.2f );
        if ( auto x = gridToPointsAndTris( *grid, pts, t, s ); !x )
            return unexpected( std::move( x.error() ) );
    }
    grid.reset(); // free grid's memory

    if ( !reportProgress( settings.cb, 0.2f ) )
        return unexpectedOperationCanceled();

    Mesh res = Mesh::fromTriangles( std::move( pts ), t, {}, subprogress( settings.cb, 0.2f, 1.0f ) );
    if ( !reportProgress( settings.cb, 1.0f ) )
        return unexpectedOperationCanceled();
    return res;
}

Expected<Mesh, std::string> gridToMesh( FloatGrid&& grid, const Vector3f& voxelSize,
    int maxFaces, float offsetVoxels, float adaptivity, ProgressCallback cb )
{
    return gridToMesh( std::move( grid ), GridToMeshSettings{
        .voxelSize = voxelSize,
        .isoValue = offsetVoxels,
        .adaptivity = adaptivity,
        .maxFaces = maxFaces,
        .cb = cb
    } );
}

Expected<MR::Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume, int maxFaces,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( vdbVolume.data, GridToMeshSettings{
        .voxelSize = vdbVolume.voxelSize,
        .isoValue = isoValue,
        .adaptivity = adaptivity,
        .maxFaces = maxFaces,
        .cb = cb
    } );
}

Expected<MR::Mesh, std::string> gridToMesh( VdbVolume&& vdbVolume, int maxFaces,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( std::move( vdbVolume.data ), GridToMeshSettings{
        .voxelSize = vdbVolume.voxelSize,
        .isoValue = isoValue,
        .adaptivity = adaptivity,
        .maxFaces = maxFaces,
        .cb = cb
    } );
}

Expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( grid, GridToMeshSettings{
        .voxelSize = voxelSize,
        .isoValue = isoValue,
        .adaptivity = adaptivity,
        .cb = cb
    } );
}

Expected<Mesh, std::string> gridToMesh( FloatGrid&& grid, const Vector3f& voxelSize,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( std::move( grid ), GridToMeshSettings{
        .voxelSize = voxelSize,
        .isoValue = isoValue,
        .adaptivity = adaptivity,
        .cb = cb
    } );
}

Expected<MR::Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( vdbVolume.data, GridToMeshSettings{
        .voxelSize = vdbVolume.voxelSize,
        .isoValue = isoValue,
        .adaptivity = adaptivity,
        .cb = cb
    } );
}

Expected<MR::Mesh, std::string> gridToMesh( VdbVolume&& vdbVolume,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( std::move( vdbVolume.data ), GridToMeshSettings{
        .voxelSize = vdbVolume.voxelSize,
        .isoValue = isoValue,
        .adaptivity = adaptivity,
        .cb = cb
    } );
}

VoidOrErrStr makeSignedWithFastWinding( FloatGrid& grid, const Vector3f& voxelSize, const Mesh& refMesh, const AffineXf3f& meshToGridXf, std::shared_ptr<IFastWindingNumber> fwn, ProgressCallback cb /*= {} */ )
{
    MR_TIMER

    const auto gridToMeshXf = meshToGridXf.inverse();

    auto activeBox = grid->evalActiveVoxelBoundingBox();
    // make dense topology tree to copy its nodes topology to original grid
    std::unique_ptr<openvdb::TopologyTree> topologyTree = std::make_unique<openvdb::TopologyTree>();
    // make it dense
    topologyTree->denseFill( activeBox, {} );
    grid->tree().topologyUnion( *topologyTree ); // after this all voxels should be active and trivial parallelism is ok
    // free topology tree
    topologyTree.reset();

    auto minCoord = activeBox.min();
    auto dims = activeBox.dim();
    VolumeIndexer indexer( Vector3i( dims.x(), dims.y(), dims.z() ) );
    const size_t volume = activeBox.volume();

    std::vector<float> windVals;
    if ( !fwn )
        fwn = std::make_shared<FastWindingNumber>( refMesh );

    if ( auto res = fwn->calcFromGrid( windVals, 
        Vector3i{ dims.x(),  dims.y(), dims.z() }, 
        Vector3f{ float( minCoord.x() ), float( minCoord.y() ), float( minCoord.z() ) }, 
        voxelSize, gridToMeshXf, 2.0f, subprogress( cb, 0.0f, 0.8f ) ); !res )
    {
        return res;
    }
    
    tbb::enumerable_thread_specific<openvdb::FloatGrid::Accessor> perThreadAccessor( grid->getAccessor() );

    if ( !ParallelFor( size_t( 0 ), volume, [&]( size_t i )
        {
            auto & accessor = perThreadAccessor.local();

            auto pos = indexer.toPos( VoxelId( i ) );
            auto coord = minCoord;
            for ( int j = 0; j < 3; ++j )
                coord[j] += pos[j];

            auto windVal = std::clamp( 1.0f - 2.0f * windVals[i], -1.0f, 1.0f );
            if ( windVal < 0.0f )
                windVal *= -windVal;
            else
                windVal *= windVal;
            accessor.modifyValue( coord, [windVal] ( float& val )
            {
                val *= windVal;
            } );
        }, subprogress( cb, 0.8f, 1.0f ) ) )
    {
        return unexpectedOperationCanceled();
    }

    grid->pruneGrid( 0.0f );
    return {};
}

Expected<Mesh, std::string> levelSetDoubleConvertion( const MeshPart& mp, const AffineXf3f& xf, float voxelSize,
    float offsetA, float offsetB, float adaptivity, std::shared_ptr<IFastWindingNumber> fwn, ProgressCallback cb /*= {} */ )
{
    MR_TIMER

    auto offsetInVoxelsA = offsetA / voxelSize;
    auto offsetInVoxelsB = offsetB / voxelSize;

    if ( cb && !cb( 0.0f ) )
        return unexpectedOperationCanceled();

    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;
    std::vector<openvdb::Vec4I> quads;
    convertToVDMMesh( mp, xf, Vector3f::diagonal( voxelSize ), points, tris );

    if (cb && !cb( 0.1f ) )
        return unexpectedOperationCanceled();

    bool needSignUpdate = !findLeftBoundary( mp.mesh.topology, mp.region ).empty();

    auto sp = subprogress( cb, 0.1f, needSignUpdate ? 0.2f : 0.3f );
    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    ProgressInterrupter interrupter1( sp );
    auto grid = MakeFloatGrid( 
        needSignUpdate ?
        openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter1, *xform, points, tris, {}, std::abs( offsetInVoxelsA ) + 1 ) :
        openvdb::tools::meshToLevelSet<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter1, *xform, points, tris, std::abs( offsetInVoxelsA ) + 1 ) );

    if ( interrupter1.getWasInterrupted() )
        return unexpectedOperationCanceled();

    if ( needSignUpdate )
    {
        sp = subprogress( cb, 0.2f, 0.3f );
        auto signRes = makeSignedWithFastWinding( grid, Vector3f::diagonal(voxelSize), mp.mesh, {}, fwn, sp );
        if ( !signRes.has_value() )
            return unexpected( signRes.error() );
    }

    openvdb::tools::volumeToMesh( *grid, points, tris, quads, offsetInVoxelsA, adaptivity );

    if ( cb && !cb( 0.5f ) )
        return unexpectedOperationCanceled();
    sp = subprogress( cb, 0.5f, 0.7f );

    ProgressInterrupter interrupter2( sp );
    grid = MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, ProgressInterrupter>
        ( interrupter2, *xform, points, tris, quads, std::abs( offsetInVoxelsB ) + 1 ) );

    if ( interrupter2.getWasInterrupted() || ( cb && !cb( 0.9f ) ) )
        return unexpectedOperationCanceled();

    VertCoords pts;
    Triangulation t;
    if ( auto x = gridToPointsAndTris( *grid, pts, t, GridToMeshSettings{
        .voxelSize = Vector3f::diagonal( voxelSize ),
        .isoValue = offsetInVoxelsB,
        .adaptivity = adaptivity,
        .cb = subprogress( cb, 0.9f, 0.95f )
    } ); !x )
        return unexpectedOperationCanceled();

    Mesh res = Mesh::fromTriangles( std::move( pts ), t );
    cb && !cb( 1.0f );
    return res;
}

} //namespace MR
#endif
