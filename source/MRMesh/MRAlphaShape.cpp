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
#include <algorithm>

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
    Triangulation & appendTris, std::vector<PreciseVertCoords> & neis, bool onlyLargerVids, AlphaShapeStats * stats )
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
    // the ball emptiness test below stops on the first point inside the ball, and the points closest
    // to #v have the best chance to be there; the integer coordinates of neis are taken as they are
    // already here, and squared in double because their squares can overflow std::int64_t
    auto distSqFromV = [&p0]( const PreciseVertCoords & c )
    {
        const double dx = double( c.pt.x ) - p0.pt.x;
        const double dy = double( c.pt.y ) - p0.pt.y;
        const double dz = double( c.pt.z ) - p0.pt.z;
        return dx * dx + dy * dy + dz * dz;
    };
    std::sort( neis.begin(), neis.end(), [&distSqFromV]( const PreciseVertCoords & a, const PreciseVertCoords & b )
    {
        return distSqFromV( a ) < distSqFromV( b );
    } );

    // a neighbor p makes a farther neighbor x redundant if p is strictly inside every ball of the given
    // radius through #v having x inside or on it: then x can neither make a triangle with #v (p blocks
    // both its balls) nor be the only point blocking a triangle of others; see the PR for the derivation
    auto makesRedundant = [rSq4 = 4 * Int128( data.intRadiusSq )]( const Vector3i64 & p, const Int128 & pp, const Vector3i64 & x, const Int128 & xx )
    {
        const auto px = dot( Vector3i128{ p }, Vector3i128{ x } );
        if ( px <= pp )
            return false; // x is not behind the plane through p orthogonal to #v-p
        const auto crossSq = Int256( xx ) * Int256( pp ) - sqr( Int256( px ) ); // |cross(p,x)|^2
        const auto d = Int256( rSq4 - pp );
        if ( 4 * crossSq >= Int256( pp ) * d )
            return false; // x is not closer to line #v-p than the centers of the balls through #v and p
        // x is strictly outside every ball of the given radius through #v and p
        return Int256( pp ) * sqr( Int256( xx - px ) ) > d * crossSq;
    };
    // exact squared distances from #v of the sorted neis, each serving as pp and as xx above;
    // thread_local to avoid allocations on every call
    thread_local std::vector<Int128> pps;
    pps.resize( neis.size() );
    for ( size_t i = 0; i < neis.size(); ++i )
    {
        const Vector3i64 p{ neis[i].pt - p0.pt };
        pps[i] = dot( Vector3i128{ p }, Vector3i128{ p } );
    }
    size_t goodSize = neis.size();
    for ( size_t i = 0; i < goodSize; ++i )
    {
        const Vector3i64 p{ neis[i].pt - p0.pt };
        size_t good = i + 1;
        for ( size_t j = i + 1; j < goodSize; ++j )
        {
            if ( !makesRedundant( p, pps[i], Vector3i64{ neis[j].pt - p0.pt }, pps[j] ) )
            {
                neis[good] = neis[j];
                pps[good] = pps[j];
                ++good;
            }
        }
        goodSize = good;
    }
    neis.resize( goodSize );

    InSphereTesterSoS tester;
    AlphaShapeStats myStats; // local to keep the counters out of the caller's memory in the loops below
    // the tester must be already reset on the ball in question, and a, b are the ids of its points
    auto ballEmpty = [&tester, &neis, &myStats]( VertId a, VertId b )
    {
        for ( const auto & pn : neis )
        {
            if ( pn.id == a || pn.id == b )
                continue;
            ++myStats.inBallTests;
            if ( tester( pn ) == InSphereResult::Inside )
                return false;
        }
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
            // so one reset is enough for the both; the tester cheaply rejects the points farther than
            // the diameter from the first of the three, and every candidate is within the search
            // radius from p0, so p0 is not given first
            ++myStats.consideredTris;
            if ( !tester.reset( pj, p0, pi, data.intRadiusSq ) )
                continue;
            ++myStats.touchableTris;
            if ( ballEmpty( pi.id, pj.id ) )
                appendTris.push_back( { v, pi.id, pj.id } );
            tester.flip();
            if ( ballEmpty( pj.id, pi.id ) )
                appendTris.push_back( { v, pj.id, pi.id } );
        }
    }
    if ( stats )
        *stats += myStats;
}

std::optional<Triangulation> findAlphaShapeAllTriangles( const PointCloud & cloud, float radius, const ProgressCallback& cb,
    AlphaShapeStats * stats )
{
    return findAlphaShapeAllTriangles( cloud, getAlphaShapeData( cloud, radius, true ), cb, stats );
}

std::optional<Triangulation> findAlphaShapeAllTriangles( const PointCloud & cloud, const AlphaShapeData & data, const ProgressCallback& cb,
    AlphaShapeStats * stats )
{
    MR_TIMER;
    struct ThreadData
    {
        Triangulation tris;
        std::vector<PreciseVertCoords> neis;
        AlphaShapeStats stats;
    };

    tbb::enumerable_thread_specific<ThreadData> threadData;
    cloud.getAABBTree(); // to avoid multiple calls to tree construction from parallel region,
                         // which can result that two different vertices will start being processed by one thread

    const bool completed = BitSetParallelFor( cloud.validPoints, [&]( VertId v )
    {
        auto & tls = threadData.local();
        findAlphaShapeNeiTriangles( cloud, v, data, tls.tris, tls.neis, true, &tls.stats );
    }, subprogress( cb, 0.0f, 0.9f ) );

    if ( stats )
        for ( const auto & tls : threadData )
            *stats += tls.stats; // the work already done is reported even if the search was cancelled

    if ( !completed )
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

Triangulation findAlphaShapeAllTriangles( const PointCloud & cloud, float radius, AlphaShapeStats * stats )
{
    auto maybe = findAlphaShapeAllTriangles( cloud, radius, ProgressCallback{}, stats );
    assert( maybe.has_value() );
    Triangulation res;
    if ( maybe.has_value() )
        res = std::move( *maybe );
    return res;
}

std::optional<Mesh> findAlphaShape( const PointCloud & cloud, float radius, const ProgressCallback& cb, AlphaShapeStats * stats )
{
    MR_TIMER;
    const auto sd = getAlphaShapeData( cloud, radius, true );
    auto maybeTris = findAlphaShapeAllTriangles( cloud, sd, subprogress( cb, 0.0f, 0.8f ), stats );
    if ( !maybeTris )
        return std::nullopt;

    // the best triangle-continuation during vertex duplication is the first one rotating
    // counter-clockwise from the reference triangle around the directed shared edge (e0, e1)
    auto betterCont = [&sd, &cloud]( VertId e0, VertId e1, VertId vRef, VertId vCand, VertId vBest )
    {
        if ( vCand == vBest )
            return false; // two remaining vertices can be equal as originals if two triangles over the shared edge
                          // had equal third vertices before duplication; ccwAroundLine requires all distinct points
        return ccwAroundLine( { sd.coords( cloud, e0 ), sd.coords( cloud, e1 ),
            sd.coords( cloud, vRef ), sd.coords( cloud, vCand ), sd.coords( cloud, vBest ) } );
    };

    int skippedFaceCount = 0;
    auto res = Mesh::fromTrianglesDuplicatingNonManifoldVertices( cloud.points, *maybeTris, nullptr,
        { .skippedFaceCount = &skippedFaceCount }, betterCont );
    assert( skippedFaceCount == 0 );
    return res;
}

Mesh findAlphaShape( const PointCloud & cloud, float radius, AlphaShapeStats * stats )
{
    auto maybe = findAlphaShape( cloud, radius, ProgressCallback{}, stats );
    assert( maybe.has_value() );
    Mesh res;
    if ( maybe.has_value() )
        res = std::move( *maybe );
    return res;
}

} //namespace MR
