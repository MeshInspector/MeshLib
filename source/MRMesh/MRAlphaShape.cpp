#include "MRAlphaShape.h"
#include "MRPointCloud.h"
#include "MRPointsInBall.h"
#include "MRInSphere.h"
#include "MRBox.h"
#include "MRBitSetParallelFor.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRProgressCallback.h"
#include "MRPch/MRTBB.h"

namespace MR
{

PreciseVertCoords AlphaShapeData::coords( const PointCloud & cloud, VertId v ) const
{
    return { v, intPoints.empty() ? toInt( cloud.points[v] ) : intPoints[v] };
}

AlphaShapeData getAlphaShapeData( const PointCloud & cloud, float radius, bool allPoints )
{
    MR_TIMER;
    assert( radius > 0 );
    AlphaShapeData res;
    auto box = Box3d( cloud.getBoundingBox() );
    if ( !box.valid() )
        return res; // no valid points in the cloud

    // getToIntConverter maps the largest box dimension onto the whole integer range, so the box is
    // enlarged up to the radius here to keep the integer radius in the same range,
    // where its square still fits in std::int64_t and the predicates below do not overflow
    const auto boxCenter = box.center();
    const auto halfRadius = Vector3d::diagonal( 0.5 * radius );
    box.include( Box3d{ boxCenter - halfRadius, boxCenter + halfRadius } );
    res.toInt = getToIntConverter( box );

    if ( allPoints )
        res.intPoints = computeIntCoords( res.toInt, cloud.points, &cloud.validPoints );

    // rounding down to be sure that the integer ball is not larger than the given one
    const auto intRadius = std::int64_t( double( radius ) * res.toInt.invRange );
    res.intRadiusSq = sqr( intRadius );

    // rounding of integer coordinates can move each of two points by half of the grid step,
    // increasing their distance by one step; two steps are added for the sake of safety
    res.searchRadius = 2 * radius + float( 2 / res.toInt.invRange );
    return res;
}

void findAlphaShapeNeiTriangles( const PointCloud & cloud, VertId v, const AlphaShapeData & data,
    Triangulation & appendTris, std::vector<PreciseVertCoords> & neis, bool onlyLargerVids )
{
    neis.clear();
    findPointsInBall( cloud, { cloud.points[v], sqr( data.searchRadius ) },
        [&]( const PointsProjectionResult & found, const Vector3f&, Ball3f & )
        {
            if ( v != found.vId )
                neis.push_back( data.coords( cloud, found.vId ) );
            return Processing::Continue;
        } );

    const auto p0 = data.coords( cloud, v );
    InSphereTesterSoS tester;
    // the tester must be already reset on the ball in question, and a, b are the ids of its points
    auto ballEmpty = [&tester, &neis]( VertId a, VertId b )
    {
        for ( const auto & pn : neis )
            if ( pn.id != a && pn.id != b && tester( pn ) == InSphereResult::Inside )
                return false;
        return true;
    };
    for ( size_t i = 0; i + 1 < neis.size(); ++i )
    {
        const auto & pi = neis[i];
        if ( onlyLargerVids && pi.id < v )
            continue;
        for ( size_t j = i + 1; j < neis.size(); ++j )
        {
            const auto & pj = neis[j];
            if ( onlyLargerVids && pj.id < v )
                continue;
            // the two balls touching all three points differ only by the side of the triangle,
            // so one reset is enough for the both
            if ( !tester.reset( p0, pi, pj, data.intRadiusSq ) )
                continue;
            if ( ballEmpty( pi.id, pj.id ) )
                appendTris.push_back( { v, pi.id, pj.id } );
            tester.flip();
            if ( ballEmpty( pj.id, pi.id ) )
                appendTris.push_back( { v, pj.id, pi.id } );
        }
    }
}

std::optional<Triangulation> findAlphaShapeAllTriangles( const PointCloud & cloud, float radius, const ProgressCallback& cb )
{
    return findAlphaShapeAllTriangles( cloud, getAlphaShapeData( cloud, radius, true ), cb );
}

std::optional<Triangulation> findAlphaShapeAllTriangles( const PointCloud & cloud, const AlphaShapeData & data, const ProgressCallback& cb )
{
    MR_TIMER;
    struct ThreadData
    {
        Triangulation tris;
        std::vector<PreciseVertCoords> neis;
    };

    tbb::enumerable_thread_specific<ThreadData> threadData;
    cloud.getAABBTree(); // to avoid multiple calls to tree construction from parallel region,
                         // which can result that two different vertices will start being processed by one thread

    if ( !BitSetParallelFor( cloud.validPoints, [&]( VertId v )
    {
        auto & tls = threadData.local();
        findAlphaShapeNeiTriangles( cloud, v, data, tls.tris, tls.neis, true );
    }, subprogress( cb, 0.0f, 0.9f ) ) )
        return std::nullopt;

    size_t numTris = 0;
    for ( const auto & tls : threadData )
        numTris += tls.tris.size();

    Triangulation res;
    res.reserve( numTris );
    for ( const auto & tls : threadData )
        res.vec_.insert( end( res ), begin( tls.tris ), end( tls.tris ) );

    if ( !reportProgress( cb, 0.95f ) )
        return std::nullopt;

    /// to avoid dependency on work distribution among threads
    tbb::parallel_sort( begin( res ), end( res ) );

    if ( !reportProgress( cb, 1.0f ) )
        return std::nullopt;

    return res;
}

Triangulation findAlphaShapeAllTriangles( const PointCloud & cloud, float radius )
{
    auto maybe = findAlphaShapeAllTriangles( cloud, radius, ProgressCallback{} );
    assert( maybe.has_value() );
    Triangulation res;
    if ( maybe.has_value() )
        res = std::move( *maybe );
    return res;
}

std::optional<Mesh> findAlphaShape( const PointCloud & cloud, float radius, const ProgressCallback& cb )
{
    MR_TIMER;
    const auto sd = getAlphaShapeData( cloud, radius, true );
    auto maybeTris = findAlphaShapeAllTriangles( cloud, sd, subprogress( cb, 0.0f, 0.8f ) );
    if ( !maybeTris )
        return std::nullopt;

    // the best triangle-continuation during vertex duplication is the first one rotating counter-clockwise
    // from the reference triangle around the shared edge directed from the neighbor vertex (e1) to the center (e0)
    auto betterCont = [&sd, &cloud]( VertId e0, VertId e1, VertId vRef, VertId vCand, VertId vBest )
    {
        if ( vCand == vBest )
            return false; // two remaining vertices can be equal as originals if two triangles over the shared edge
                          // had equal third vertices before duplication; ccwAroundLine requires all distinct points
        return ccwAroundLine( { sd.coords( cloud, e1 ), sd.coords( cloud, e0 ),
            sd.coords( cloud, vRef ), sd.coords( cloud, vCand ), sd.coords( cloud, vBest ) } );
    };

    int skippedFaceCount = 0;
    auto res = Mesh::fromTrianglesDuplicatingNonManifoldVertices( cloud.points, *maybeTris, nullptr,
        { .skippedFaceCount = &skippedFaceCount }, betterCont );
    assert( skippedFaceCount == 0 );
    return res;
}

Mesh findAlphaShape( const PointCloud & cloud, float radius )
{
    auto maybe = findAlphaShape( cloud, radius, ProgressCallback{} );
    assert( maybe.has_value() );
    Mesh res;
    if ( maybe.has_value() )
        res = std::move( *maybe );
    return res;
}

} //namespace MR
