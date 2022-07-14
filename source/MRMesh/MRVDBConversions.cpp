#ifndef __EMSCRIPTEN__
#include "MRVDBConversions.h"
#include "MRFloatGrid.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRAffineXf3.h"
#include "MRTimer.h"
#include "MRSimpleVolume.h"
#include "MRPch/MROpenvdb.h"

namespace MR
{
constexpr float denseVolumeToGridTolerance = 1e-6f;

struct Interrupter
{
    Interrupter( const ProgressCallback& cb ) :
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
        if ( cb_ )
            return !cb_( float( percent ) / 100.0f );
        return false;
    }

private:
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

FloatGrid meshToLevelSet( const MeshPart& mp, const AffineXf3f& xf,
                          const Vector3f& voxelSize, float surfaceOffset,
                          const ProgressCallback& cb )
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
    return MakeFloatGrid( openvdb::tools::meshToLevelSet<openvdb::FloatGrid, Interrupter>
        ( interrupter, *xform, points, tris, surfaceOffset ) );
}

FloatGrid meshToDistanceField( const MeshPart& mp, const AffineXf3f& xf,
    const Vector3f& voxelSize, float surfaceOffset /*= 3 */,
    const ProgressCallback& cb )
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
    return MakeFloatGrid( openvdb::tools::meshToUnsignedDistanceField<openvdb::FloatGrid, Interrupter>
        ( interrupter, *xform, points, tris, {}, surfaceOffset ) );
}

FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolue,
                                   const ProgressCallback& cb )
{
    MR_TIMER;
    if ( cb )
        cb( 0.0f );
    openvdb::math::Coord minCoord( 0, 0, 0 );
    openvdb::math::Coord dimsCoord( simpleVolue.dims.x, simpleVolue.dims.y, simpleVolue.dims.z );
    openvdb::math::CoordBBox denseBBox( minCoord, minCoord + dimsCoord.offsetBy( -1 ) );
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense( denseBBox, const_cast< float* >( simpleVolue.data.data() ) );
    if ( cb )
        cb( 0.5f );
    std::shared_ptr<openvdb::FloatGrid> grid = std::make_shared<openvdb::FloatGrid>();
    openvdb::tools::copyFromDense( dense, *grid, denseVolumeToGridTolerance );
    if ( cb )
        cb( 1.0f );
    return MakeFloatGrid( std::move( grid ) );
}

tl::expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    float offsetVoxels, float adaptivity, int maxFaces,
    const ProgressCallback& cb )
{
    MR_TIMER;
    if ( cb )
        cb( 0.0f );
    std::vector<openvdb::Vec3s> pointsRes;
    std::vector<openvdb::Vec3I> trisRes;
    std::vector<openvdb::Vec4I> quadRes;
    openvdb::tools::volumeToMesh( *grid, pointsRes, trisRes, quadRes,
                                  offsetVoxels, adaptivity );

    if ( trisRes.size() + 2 * quadRes.size() > maxFaces )
        return tl::make_unexpected( "Triangles number limit exceeded." );
    if ( cb )
        cb( 0.2f );
    std::vector<Vector3f> points( pointsRes.size() );
    std::vector<MeshBuilder::Triangle> tris;
    tris.reserve( trisRes.size() + 2 * quadRes.size() );
    for ( int i = 0; i < points.size(); ++i )
    {
        points[i][0] = pointsRes[i][0] * voxelSize[0];
        points[i][1] = pointsRes[i][1] * voxelSize[1];
        points[i][2] = pointsRes[i][2] * voxelSize[2];

        if ( cb )
            cb( 0.2f + 0.4f * ( float( i ) / float( points.size() ) ) );
    }
    int t = 0;
    for ( const auto& tri : trisRes )
    {
        MeshBuilder::Triangle newTri
        {
            VertId( ( int )tri[2] ),
            VertId( ( int )tri[1] ),
            VertId( ( int )tri[0] ),
            FaceId( t )
        };
        tris.push_back( newTri );
        ++t;

        if ( cb )
            cb( 0.6f + 0.4f * ( float( t ) / float( tris.capacity() ) ) );
    }
    for ( const auto& quad : quadRes )
    {
        MeshBuilder::Triangle newTri
        {
            VertId( ( int )quad[2] ),
            VertId( ( int )quad[1] ),
            VertId( ( int )quad[0] ),
            FaceId( t )
        };
        tris.push_back( newTri );
        ++t;

        if ( cb )
            cb( 0.6f + 0.4f * ( float( t ) / float( tris.capacity() ) ) );
        newTri =
        {
            VertId( ( int )quad[0] ),
            VertId( ( int )quad[3] ),
            VertId( ( int )quad[2] ),
            FaceId( t )
        };
        tris.push_back( newTri );
        ++t;

        if ( cb )
            cb( 0.6f + 0.4f * ( float( t ) / float( tris.capacity() ) ) );
    }

    Mesh res;
    res.topology = MeshBuilder::fromTriangles( tris );
    res.points.vec_ = std::move( points );

    if ( cb )
        cb( 1.0f );

    return std::move( res );
}

} //namespace MR
#endif
