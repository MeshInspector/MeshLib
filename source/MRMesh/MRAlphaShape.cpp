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

#if __GNUC__ >= 12 // false positive array-bounds warnings in boost widening conversions like Int1024( Int256 )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

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
    Triangulation & appendTris, std::vector<AlphaShapeNei> & neis, bool onlyLargerVids, AlphaShapeStats * stats )
{
    const auto p0 = data.coords( cloud, v );
    neis.clear();
    findPointsInBall( cloud, { cloud.points[v], sqr( data.searchRadius ) },
        [&]( const PointsProjectionResult & found, const Vector3f&, Ball3f & )
        {
            if ( v != found.vId )
            {
                const auto c = data.coords( cloud, found.vId );
                const Vector3i64 d{ c.pt - p0.pt };
                neis.push_back( { c, dot( Vector3i128{ d }, Vector3i128{ d } ) } );
            }
            return Processing::Continue;
        } );

    // the ball emptiness test below stops on the first point inside the ball,
    // and the points closest to #v have the best chance to be there
    std::sort( neis.begin(), neis.end(), []( const AlphaShapeNei & a, const AlphaShapeNei & b )
    {
        return a.distSq < b.distSq;
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
    size_t goodSize = neis.size();
    for ( size_t i = 0; i < goodSize; ++i )
    {
        const Vector3i64 p{ neis[i].coords.pt - p0.pt };
        size_t good = i + 1;
        for ( size_t j = i + 1; j < goodSize; ++j )
        {
            if ( !makesRedundant( p, neis[i].distSq, Vector3i64{ neis[j].coords.pt - p0.pt }, neis[j].distSq ) )
                neis[good++] = neis[j];
        }
        goodSize = good;
    }
    neis.resize( goodSize );

    // the sphere quantities computed by reset() are reused by the shadow filter below
    struct Tester : InSphereTesterSoS
    {
        const Vector3i64 & normal() const { return w; }
        const Int256 & normalSq() const { return W; }
        const Int512 & heightSq() const { return E; }

        /// whether d is strictly outside both balls touching the triangle, which is
        /// A * W > S * |t| in the notation of operator()
        bool outsideBothBalls( const Vector3i & d ) const
        {
            const Vector3i64 q{ d - a };
            const auto qq = dot( Vector3i128{ q }, Vector3i128{ q } );
            if ( qq > 4 * Int128( rSq ) )
                return true;
            const auto A = W * Int256( qq ) - dot( Vector3i256{ q }, M );
            if ( A <= 0 )
                return false;
            const auto t = dot( Vector3i128{ q }, Vector3i128{ w } );
            return sqr( Int1024( A ) ) * Int1024( W ) > Int1024( E ) * sqr( Int1024( t ) );
        }
    } tester;
    AlphaShapeStats myStats; // local to keep the counters out of the caller's memory in the loops below
    // the tester must be already reset on the ball in question, and a, b are the ids of its points
    auto ballEmpty = [&tester, &neis, &myStats]( VertId a, VertId b )
    {
        for ( const auto & pn : neis )
        {
            if ( pn.coords.id == a || pn.coords.id == b )
                continue;
            ++myStats.inBallTests;
            if ( tester( pn.coords ) == InSphereResult::Inside )
                return false;
        }
        return true;
    };

    // whether b * sqrt( ew ) > | c * nd | exactly, given b > 0 and ew >= 0
    auto insideWedge = []( const Int256 & b, const Int512 & c, const Int128 & nd, const Int1024 & ew )
    {
        return sqr( Int1024( b ) ) * ew > sqr( Int1024( c ) * Int1024( nd ) );
    };

    // the tester is reset on a ball touching #v and the found triangle's other points p and q (given
    // relative to #v); a neighbour strictly outside both touching balls and strictly inside the wedge
    // of the four planes via #v, one of p and q, and one of the two centers, is redundant for the same
    // reason as in the filter above: every ball of the given radius via #v containing such a neighbour
    // has p or q strictly inside; see the PR for the derivation
    auto dropShadowed = [&]( const Vector3i64 & p, const Vector3i64 & q, size_t from )
    {
        const auto pp = dot( Vector3i128{ p }, Vector3i128{ p } );
        const auto qq = dot( Vector3i128{ q }, Vector3i128{ q } );
        const auto pq = dot( Vector3i128{ p }, Vector3i128{ q } );
        Int512 cp, cq;
        Int1024 ew;
        bool prepared = false; // most triangles shadow no point at all, so the rest is computed on demand
        size_t good = from;
        for ( size_t k = from; k < neis.size(); ++k )
        {
            bool shadowed = false;
            const auto & x = neis[k].coords;
            const Vector3i64 d{ x.pt - p0.pt };
            const auto pd = dot( Vector3i128{ p }, Vector3i128{ d } );
            const auto qd = dot( Vector3i128{ q }, Vector3i128{ d } );
            // the wedge is within the dihedral angle of the half-planes via #v containing p and q
            const auto bp = Int256( pp ) * Int256( qd ) - Int256( pq ) * Int256( pd );
            const auto bq = Int256( qq ) * Int256( pd ) - Int256( pq ) * Int256( qd );
            if ( bp > 0 && bq > 0 )
            {
                if ( !prepared )
                {
                    prepared = true;
                    const auto & W = tester.normalSq();
                    // sqr( S ) of the exact center identity 2 * W^2 * center = W * M +- S * n, where
                    // M = qq * ( pp - pq ) * p + pp * ( qq - pq ) * q = 2 * W * ( circumcenter - #v );
                    // the tester's E is the same for any point of the triangle taken as the origin
                    ew = Int1024( tester.heightSq() ) * Int1024( W );
                    cp = Int512( W ) * Int512( pp ) * Int512( qq - pq );
                    cq = Int512( W ) * Int512( qq ) * Int512( pp - pq );
                }
                // the tester's normal is cross( p, q ) up to the sign, which does not matter here
                const auto nd = dot( Vector3i128{ tester.normal() }, Vector3i128{ d } );
                shadowed = insideWedge( bp, cp, nd, ew ) && insideWedge( bq, cq, nd, ew )
                        && tester.outsideBothBalls( x.pt );
            }
            if ( shadowed )
                continue;
            if ( good != k )
                neis[good] = neis[k];
            ++good;
        }
        neis.resize( good );
    };

    for ( size_t i = 0; i + 1 < neis.size(); ++i )
    {
        const auto & pi = neis[i].coords;
        if ( onlyLargerVids && pi.id < v )
            continue;
        for ( size_t j = i + 1; j < neis.size(); ++j )
        {
            const auto & pj = neis[j].coords;
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
            bool found = ballEmpty( pi.id, pj.id );
            if ( found )
                appendTris.push_back( { v, pi.id, pj.id } );
            tester.flip();
            if ( ballEmpty( pj.id, pi.id ) )
            {
                appendTris.push_back( { v, pj.id, pi.id } );
                found = true;
            }
            // the filter is symmetric in the two balls, so one pass per pair is enough
            if ( found )
                dropShadowed( Vector3i64{ pi.pt - p0.pt }, Vector3i64{ pj.pt - p0.pt }, j + 1 );
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
        std::vector<AlphaShapeNei> neis;
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

#if __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif

} //namespace MR
