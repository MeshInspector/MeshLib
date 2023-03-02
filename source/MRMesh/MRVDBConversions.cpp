#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVDBConversions.h"
#include "MRFloatGrid.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRPch/MROpenvdb.h"
#include "MRBox.h"
#include "MRFastWindingNumber.h"
#include "MRVolumeIndexer.h"
#include "MRRegionBoundary.h"
#include <thread>

namespace MR
{
constexpr float denseVolumeToGridTolerance = 1e-6f;

struct Interrupter
{
    Interrupter( ProgressCallback cb ) :
        cb_{ cb }
    {};

    void start( const char* name = nullptr )
    {
        ( void )name;
    }
    void end()
    {}
    bool wasInterrupted( int percent = -1 )
    {
        wasInterrupted_ = false;
        if ( cb_ )
            wasInterrupted_ = !cb_( float( std::clamp( percent, 0, 100 ) ) / 100.0f );
        return wasInterrupted_;
    }
    bool getWasInterrupted() const
    {
        return wasInterrupted_;
    }
private:
    bool wasInterrupted_{ false };
    ProgressCallback cb_;
};

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
inline typename std::enable_if<std::is_scalar<typename GridType::ValueType>::value, void>::type
gridToPointsAndTris(
    const GridType& grid, const Vector3f& voxelSize,
    VertCoords & points, Triangulation & t,
    double isovalue,
    double adaptivity,
    bool relaxDisorientedTriangles )
{
    MR_TIMER

    openvdb::tools::VolumeToMesh mesher(isovalue, adaptivity, relaxDisorientedTriangles);
    mesher(grid);

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
                inPt.x() * voxelSize.x,
                inPt.y() * voxelSize.y,
                inPt.z() * voxelSize.z };
        }
    } );
    inPts.reset(nullptr);

    auto& polygonPoolList = mesher.polygonPoolList();

    // Preallocate primitive lists
    size_t numQuads = 0, numTriangles = 0;
    for (size_t n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
        openvdb::tools::PolygonPool& polygons = polygonPoolList[n];
        numTriangles += polygons.numTriangles();
        numQuads += polygons.numQuads();
    }

    const size_t tNum = numTriangles + 2 * numQuads;
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
    Interrupter interrupter( cb );
    auto resGrid = MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, Interrupter>
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
    Interrupter interrupter( cb );

    auto resGrid = MakeFloatGrid( openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid, Interrupter>
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

tl::expected<VdbVolume, std::string> meshToVolume( const Mesh& mesh, const MeshToVolumeParams& params /*= {} */ )
{
    if ( params.type == MeshToVolumeParams::Type::Signed && !mesh.topology.isClosed() )
        return tl::make_unexpected( "Only closed mesh can be converted to signed volume" );

    auto shift = AffineXf3f::translation( mesh.computeBoundingBox( &params.worldXf ).min - params.surfaceOffset * params.voxelSize );
    FloatGrid grid;
    if ( params.type == MeshToVolumeParams::Type::Signed )
        grid = meshToLevelSet( mesh, shift.inverse() * params.worldXf, params.voxelSize, params.surfaceOffset, params.cb );
    else
        grid = meshToDistanceField( mesh, shift.inverse() * params.worldXf, params.voxelSize, params.surfaceOffset, params.cb );

    if ( !grid )
        return tl::make_unexpected( "Operation canceled" );

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

tl::expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    int maxFaces, float offsetVoxels, float adaptivity, ProgressCallback cb )
{
    MR_TIMER;
    if ( cb && !cb( 0.0f ) )
        return tl::make_unexpected( "Operation was canceled." );

    VertCoords pts;
    Triangulation t;
    gridToPointsAndTris( *grid, voxelSize, pts, t, offsetVoxels, adaptivity, true );

    if ( t.size() > maxFaces )
        return tl::make_unexpected( "Triangles number limit exceeded." );
    
    if ( cb && !cb( 0.2f ) )
        return tl::make_unexpected( "Operation was canceled." );

    Mesh res = Mesh::fromTriangles( std::move( pts ), t, {}, subprogress( cb, 0.2f, 0.8f ) );
    cb && !cb( 1.0f );
    return res;
}

tl::expected<Mesh, std::string> gridToMesh( FloatGrid&& grid, const Vector3f& voxelSize,
    int maxFaces, float offsetVoxels, float adaptivity, ProgressCallback cb )
{
    MR_TIMER;
    if ( cb && !cb( 0.0f ) )
        return tl::make_unexpected( "Operation was canceled." );

    VertCoords pts;
    Triangulation t;
    gridToPointsAndTris( *grid, voxelSize, pts, t, offsetVoxels, adaptivity, true );
    grid.reset(); // free grid's memory

    if ( t.size() > maxFaces )
        return tl::make_unexpected( "Triangles number limit exceeded." );
    
    if ( cb && !cb( 0.2f ) )
        return tl::make_unexpected( "Operation was canceled." );

    Mesh res = Mesh::fromTriangles( std::move( pts ), t );
    cb && !cb( 1.0f );
    return res;
}

tl::expected<MR::Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume, int maxFaces,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( vdbVolume.data, vdbVolume.voxelSize, maxFaces, isoValue, adaptivity, cb );
}

tl::expected<MR::Mesh, std::string> gridToMesh( VdbVolume&& vdbVolume, int maxFaces,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( std::move( vdbVolume.data ), vdbVolume.voxelSize, maxFaces, isoValue, adaptivity, cb );
}

tl::expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( grid, voxelSize, INT_MAX, isoValue, adaptivity, cb );
}

tl::expected<Mesh, std::string> gridToMesh( FloatGrid&& grid, const Vector3f& voxelSize,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( std::move( grid ), voxelSize, INT_MAX, isoValue, adaptivity, cb );
}

tl::expected<MR::Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( vdbVolume.data, vdbVolume.voxelSize, isoValue, adaptivity, cb );
}

tl::expected<MR::Mesh, std::string> gridToMesh( VdbVolume&& vdbVolume,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( std::move( vdbVolume.data ), vdbVolume.voxelSize, isoValue, adaptivity, cb );
}

VoidOrErrStr makeSignedWithFastWinding( FloatGrid& grid, const Vector3f& voxelSize, const Mesh& refMesh, ProgressCallback cb /*= {} */ )
{
    MR_TIMER

    std::atomic<bool> keepGoing{ true };
    auto mainThreadId = std::this_thread::get_id();

    FastWindingNumber fwn( refMesh );

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
    tbb::parallel_for( tbb::blocked_range<size_t>( size_t( 0 ), size_t( activeBox.volume() ) ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto accessor = grid->getAccessor();
        for ( auto i = range.begin(); i < range.end(); ++i )
        {
            if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            auto pos = indexer.toPos( VoxelId( i ) );
            auto coord = minCoord;
            for ( int j = 0; j < 3; ++j )
                coord[j] += pos[j];

            auto coord3i = Vector3i( coord.x(), coord.y(), coord.z() );
            auto pointInSpace = mult( voxelSize, Vector3f( coord3i ) );
            auto windVal = fwn.calc( pointInSpace, 2.0f );
            windVal = std::clamp( 1.0f - 2.0f * windVal, -1.0f, 1.0f );
            if ( windVal < 0.0f )
                windVal *= -windVal;
            else
                windVal *= windVal;
            accessor.modifyValue( coord, [windVal] ( float& val )
            {
                val *= windVal;
            } );
            if ( cb && mainThreadId == std::this_thread::get_id() && !cb( float( i ) / float( range.size() ) ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    }, tbb::static_partitioner() );
    if ( !keepGoing )
        return tl::make_unexpected( "Operation was canceled." );
    grid->pruneGrid( 0.0f );
    return {};
}

tl::expected<Mesh, std::string> levelSetDoubleConvertion( const MeshPart& mp, const AffineXf3f& xf, float voxelSize,
    float offsetA, float offsetB, float adaptivity, ProgressCallback cb /*= {} */ )
{
    MR_TIMER

    auto offsetInVoxelsA = offsetA / voxelSize;
    auto offsetInVoxelsB = offsetB / voxelSize;

    if ( cb && !cb( 0.0f ) )
        return tl::make_unexpected( "Operation was canceled." );

    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;
    std::vector<openvdb::Vec4I> quads;
    convertToVDMMesh( mp, xf, Vector3f::diagonal( voxelSize ), points, tris );

    if (cb && !cb( 0.1f ) )
        return tl::make_unexpected( "Operation was canceled." );

    bool needSignUpdate = !findRegionBoundary( mp.mesh.topology, mp.region ).empty();

    auto sp = subprogress( cb, 0.1f, needSignUpdate ? 0.2f : 0.3f );
    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    Interrupter interrupter1( sp );
    auto grid = MakeFloatGrid( 
        needSignUpdate ?
        openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid, Interrupter>
        ( interrupter1, *xform, points, tris, {}, std::abs( offsetInVoxelsA ) + 1 ) :
        openvdb::tools::meshToLevelSet<openvdb::FloatGrid, Interrupter>
        ( interrupter1, *xform, points, tris, std::abs( offsetInVoxelsA ) + 1 ) );

    if ( interrupter1.getWasInterrupted() )
        return tl::make_unexpected( "Operation was canceled." );

    if ( needSignUpdate )
    {
        sp = subprogress( cb, 0.2f, 0.3f );
        auto signRes = makeSignedWithFastWinding( grid,Vector3f::diagonal(voxelSize),mp.mesh,sp );
        if ( !signRes.has_value() )
            return tl::make_unexpected( signRes.error() );
    }

    openvdb::tools::volumeToMesh( *grid, points, tris, quads, offsetInVoxelsA, adaptivity );

    if ( cb && !cb( 0.5f ) )
        return tl::make_unexpected( "Operation was canceled." );
    sp = subprogress( cb, 0.5f, 0.7f );

    Interrupter interrupter2( sp );
    grid = MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, Interrupter>
        ( interrupter2, *xform, points, tris, quads, std::abs( offsetInVoxelsB ) + 1 ) );

    if ( interrupter2.getWasInterrupted() || ( cb && !cb( 0.9f ) ) )
        return tl::make_unexpected( "Operation was canceled." );

    VertCoords pts;
    Triangulation t;
    gridToPointsAndTris( *grid, Vector3f::diagonal( voxelSize ), pts, t, offsetInVoxelsB, adaptivity, true );

    Mesh res = Mesh::fromTriangles( std::move( pts ), t );
    cb && !cb( 1.0f );
    return res;
}

} //namespace MR
#endif
