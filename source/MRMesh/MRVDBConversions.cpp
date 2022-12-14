#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRVDBConversions.h"
#include "MRFloatGrid.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRPch/MROpenvdb.h"
#include "MRBox.h"

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
            wasInterrupted_ = !cb_( float( percent ) / 100.0f );
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

tl::expected<Mesh, std::string> convertFromVDMMesh( const Vector3f& voxelSize,
    std::vector<openvdb::Vec3s>&& pointsArg, 
    std::vector<openvdb::Vec3I>&& trisArg,
    std::vector<openvdb::Vec4I>&& quadArg,
    ProgressCallback cb )
{
    if ( cb && !cb( 0.0f ) )
            return tl::make_unexpected( "Operation was canceled." );
    std::vector<Vector3f> points( pointsArg.size() );
    for ( int i = 0; i < points.size(); ++i )
    {
        points[i][0] = pointsArg[i][0] * voxelSize[0];
        points[i][1] = pointsArg[i][1] * voxelSize[1];
        points[i][2] = pointsArg[i][2] * voxelSize[2];

        if ( cb && !(i & 0x3FF) && !cb( 0.5f * ( float( i ) / float( points.size() ) ) ) )
                return tl::make_unexpected( "Operation was canceled." );
    }
    pointsArg = std::vector<openvdb::Vec3s>{};

    Triangulation t;
    const size_t tNum = trisArg.size() + 2 * quadArg.size();
    t.reserve( tNum );
    size_t tCounter = 0;
    for ( const auto& tri : trisArg )
    {

        ThreeVertIds newTri
        {
            VertId( ( int )tri[2] ),
            VertId( ( int )tri[1] ),
            VertId( ( int )tri[0] )
        };
        t.push_back( newTri );

        ++tCounter;
        if ( cb && !(tCounter & 0x3FF) && !cb( 0.5f + 0.5f * ( float( tCounter ) / tNum ) ) )
                return tl::make_unexpected( "Operation was canceled." );
    }
    trisArg = std::vector<openvdb::Vec3I>{};

    for ( const auto& quad : quadArg )
    {
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

        tCounter += 2;
        if ( cb && !(tCounter & 0x3FE) && !cb( 0.5f + 0.5f * ( float( tCounter ) / tNum ) ) )
                return tl::make_unexpected( "Operation was canceled." );
    }
    quadArg = std::vector<openvdb::Vec4I>{};

    Mesh res;
    res.topology = MeshBuilder::fromTriangles( t );
    res.points.vec_ = std::move( points );

    if ( cb && !cb( 1.0f ) )
            return tl::make_unexpected( "Operation was canceled." );

    return res;
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
    std::shared_ptr<openvdb::FloatGrid> grid = std::make_shared<openvdb::FloatGrid>();
    openvdb::tools::copyFromDense( dense, *grid, denseVolumeToGridTolerance );
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
    if ( cb )
        if ( !cb( 0.0f ) )
            return tl::make_unexpected( "Operation was canceled." );
    std::vector<openvdb::Vec3s> pointsRes;
    std::vector<openvdb::Vec3I> trisRes;
    std::vector<openvdb::Vec4I> quadRes;
    openvdb::tools::volumeToMesh( *grid, pointsRes, trisRes, quadRes,
                                  offsetVoxels, adaptivity );

    if ( trisRes.size() + 2 * quadRes.size() > maxFaces )
        return tl::make_unexpected( "Triangles number limit exceeded." );
    
    ProgressCallback passCallback;
    if ( cb )
    {
        if ( !cb( 0.2f ) )
            return tl::make_unexpected( "Operation was canceled." );
        passCallback = [cb] ( float p )
        {
            return cb( 0.2f + 0.8f * p );
        };
    }
    return convertFromVDMMesh( voxelSize, std::move( pointsRes ), std::move( trisRes ), std::move( quadRes ), passCallback );
}

tl::expected<MR::Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume, int maxFaces,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( vdbVolume.data, vdbVolume.voxelSize, maxFaces, isoValue, adaptivity, cb );
}

tl::expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( grid, voxelSize, INT_MAX, isoValue, adaptivity, cb );
}

tl::expected<MR::Mesh, std::string> gridToMesh( const VdbVolume& vdbVolume,
    float isoValue /*= 0.0f*/, float adaptivity /*= 0.0f*/, ProgressCallback cb /*= {} */ )
{
    return gridToMesh( vdbVolume.data, vdbVolume.voxelSize, isoValue, adaptivity, cb );
}

tl::expected<Mesh, std::string> levelSetDoubleConvertion( const MeshPart& mp, const AffineXf3f& xf, float voxelSize,
    float offsetA, float offsetB, float adaptivity, ProgressCallback cb /*= {} */ )
{
    MR_TIMER

    auto offsetInVoxelsA = offsetA / voxelSize;
    auto offsetInVoxelsB = offsetB / voxelSize;

    if ( cb )
        if ( !cb( 0.0f ) )
            return tl::make_unexpected( "Operation was canceled." );

    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;
    std::vector<openvdb::Vec4I> quads;
    convertToVDMMesh( mp, xf, Vector3f::diagonal( voxelSize ), points, tris );

    ProgressCallback passCb;
    if ( cb )
    {
        if ( !cb( 0.1f ) )
            return tl::make_unexpected( "Operation was canceled." );
        passCb = [cb] ( float p )
        {
            return cb( 0.1f + 0.2f * p );
        };
    }
    openvdb::math::Transform::Ptr xform = openvdb::math::Transform::createLinearTransform();
    Interrupter interrupter1( passCb );
    auto grid = MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, Interrupter>
        ( interrupter1, *xform, points, tris, std::abs( offsetInVoxelsA ) + 1 ) );

    if ( interrupter1.getWasInterrupted() )
        return tl::make_unexpected( "Operation was canceled." );

    openvdb::tools::volumeToMesh( *grid, points, tris, quads, offsetInVoxelsA, adaptivity );

    if ( cb )
    {
        if ( !cb( 0.5f ) )
            return tl::make_unexpected( "Operation was canceled." );
        passCb = [cb] ( float p )
        {
            return cb( 0.5f + 0.2f * p );
        };
    }

    Interrupter interrupter2( passCb );
    grid = MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, Interrupter>
        ( interrupter2, *xform, points, tris, quads, std::abs( offsetInVoxelsB ) + 1 ) );

    if ( interrupter2.getWasInterrupted() )
        return tl::make_unexpected( "Operation was canceled." );

    openvdb::tools::volumeToMesh( *grid, points, tris, quads,
                                  offsetInVoxelsB, adaptivity );

    if ( cb )
    {
        if ( !cb( 0.9f ) )
            return tl::make_unexpected( "Operation was canceled." );
        passCb = [cb] ( float p )
        {
            return cb( 0.9f + 0.1f * p );
        };
    }

    return convertFromVDMMesh( Vector3f::diagonal( voxelSize ), std::move( points ), std::move( tris ), std::move( quads ), passCb );
}

} //namespace MR
#endif
