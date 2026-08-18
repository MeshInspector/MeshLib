#include "MRAlphaShape.h"
#include "MRFastInt.h"
#include "MRInt64Mul128.h"
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

#if __GNUC__ >= 12 // false positive array-bounds/stringop warnings from GCC on the fixed-width FastInt array operations below
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

namespace
{

// whether b * sqrt( ew ) > | c * nd | exactly, given b > 0 and ew >= 0;
// b <= 2^129, c <= 2^257, nd <= 2^96 and ew <= 2^450 as derived below, so the left side is at most
// 2^708, in 1024 bits as the sum of the widths of its factors, and the right one at most 2^706.
// c * nd <= 2^353 is narrowed to six words before squaring: mulWords skips the zero words of its
// left operand but iterates over every word of its right one, so the square costs six words by six
// instead of six by seven
bool insideWedge( const FastInt256 & b, const FastInt<320> & c, const FastInt128 & nd, const FastInt<512> & ew )
{
    return b * b * ew > FastInt<1024>( sqr( FastInt<384>( c * nd ) ) );
}

// excludes from neis[j + 1, end) the neighbours shadowed by the pair ( neis[i], neis[j] ), on the two
// balls of which the tester must be already reset; p0 is the integer point #v and i < j.
// A neighbour strictly outside both balls and strictly inside the wedge of the four planes via #v,
// one of the pair and one of the two centers, is redundant exactly like the ones the distance filter
// in findAlphaShapeNeiTriangles drops: every ball of the given radius via #v containing it has one
// of the pair strictly inside; see the PR for the derivation
void dropShadowed( std::vector<AlphaShapeNei> & neis, size_t i, size_t j, const Vector3i & p0,
    const FastInSphereTesterSoS & tester, AlphaShapeStats & stats )
{
    const size_t from = j + 1;
    if ( from >= neis.size() )
        return;
    const Vector3i64 p{ neis[i].coords.pt - p0 }, q{ neis[j].coords.pt - p0 };
    const FastInt128 pp = neis[i].distSq, qq = neis[j].distSq; // already exact in the neighbours
    const auto pq = dot( Vector3i64mul{ p }, Vector3i64mul{ q } );
    // bp and bq below are the dot products of d with the two vectors up and uq, and a negative
    // sign of either rejects the candidate before any bignum work, so the exact values are needed
    // only where the double ones are too close to zero. The operands have to approximate the
    // values on the integer grid, where the predicates and the ties live, and the dot products are
    // taken in double rather than converted from the exact bignums: the same accuracy up to 0.7 bits
    // and no conversion at all. Writing B < 2^31 for a difference of two points and u = 2^-53:
    //   Vector3d of a difference is exact, B being far below 2^53;
    //   the three dot products are below 3*B^2 < 2^64, and five roundings leave them off by 2^12;
    //   the components of up and uq are below 6*B^3 < 2^96, off by 2^45 - dominated by q * 2^12
    //     and p * 2^12 from the line above, not by the three roundings of 2^42 here;
    //   their dot products with d are below 18*B^4 < 2^129, off by 3 * 2^45 * B = 2^78 from the
    //     line above plus five roundings of 2^76, so below 2^79 in total.
    // A double below -2^79 is therefore negative exactly as well; the tolerance takes 32 times that
    // margin, at the price of evaluating exactly the few candidates falling in between. Measured
    // worst error over 400k configurations at the full coordinate range: 2^74.6. Contraction into
    // fused multiply-add only removes roundings, so it cannot break the bound, and a value it moves
    // across the tolerance lands in the band that goes to the exact test anyway.
    // All of this is hoisted out of the loop over the candidates.
    const Vector3d pDbl( p ), qDbl( q );
    const double ppDbl = dot( pDbl, pDbl ), qqDbl = dot( qDbl, qDbl ), pqDbl = dot( pDbl, qDbl );
    const Vector3d up{ qDbl * ppDbl - pDbl * pqDbl };
    const Vector3d uq{ pDbl * qqDbl - qDbl * pqDbl };
    constexpr double tolerance = 2e25; // 2^84 > 32 * 2^79
    FastInt<320> cp, cq;
    FastInt<512> ew;
    bool prepared = false; // most pairs shadow no point at all, so the rest is computed on demand
    size_t good = from;
    for ( size_t k = from; k < neis.size(); ++k )
    {
        bool shadowed = false;
        const auto & x = neis[k].coords;
        const Vector3i64 d{ x.pt - p0 };
        ++stats.shadowTests;
        // most candidates are rejected by the sign of bp or bq alone, which the doubles above
        // decide safely outside the tolerance; only the rest reaches the exact arithmetic
        if ( dot( up, Vector3d( d ) ) > -tolerance && dot( uq, Vector3d( d ) ) > -tolerance )
        {
            ++stats.exactShadowTests;
            // any difference of two points is below 2^31, so every dot product here is below 2^64
            const auto pd = dot( Vector3i64mul{ p }, Vector3i64mul{ d } );
            const auto qd = dot( Vector3i64mul{ q }, Vector3i64mul{ d } );
            // the wedge is within the dihedral angle of the half-planes via #v containing p and q;
            // bp and bq are the dot products of d with two vectors of magnitude below 2^96,
            // so both of them are below 2^129
            const auto bp = Int128Mul256( pp ) * Int128Mul256( qd ) - Int128Mul256( pq ) * Int128Mul256( pd );
            const auto bq = Int128Mul256( qq ) * Int128Mul256( pd ) - Int128Mul256( pq ) * Int128Mul256( qd );
            if ( bp > 0 && bq > 0 )
            {
                if ( !prepared )
                {
                    prepared = true;
                    const auto & W = tester.normalSq();
                    // sqr( S ) of the exact center identity 2 * W^2 * center = W * M +- S * n, where
                    // M = qq * ( pp - pq ) * p + pp * ( qq - pq ) * q = 2 * W * ( circumcenter - #v );
                    // the tester's E is the same for any of the three points as the origin;
                    // W <= 2^128 and E <= 2^322, so ew <= 2^450 and cp, cq <= 2^257
                    ew = FastInt<512>( tester.heightSq() * W );
                    cp = FastInt<320>( W * pp * ( qq - pq ) );
                    cq = FastInt<320>( W * qq * ( pp - pq ) );
                }
                // the tester's normal is cross( p, q ) up to the sign, which does not matter here
                const auto nd = dot( Vector3i64mul{ tester.normal() }, Vector3i64mul{ d } );
                shadowed = insideWedge( bp, cp, nd, ew ) && insideWedge( bq, cq, nd, ew )
                        && tester.outsideBothSpheres( x.pt );
            }
        }
        if ( shadowed )
        {
            ++stats.shadowedNeis;
            continue;
        }
        if ( good != k )
            neis[good] = neis[k];
        ++good;
    }
    neis.resize( good );
}

} // anonymous namespace

void findAlphaShapeNeiTriangles( const PointCloud & cloud, VertId v, const AlphaShapeData & data,
    Triangulation & appendTris, std::vector<AlphaShapeNei> & neis, bool onlyLargerVids, AlphaShapeStats * stats )
{
    const auto p0 = data.coords( cloud, v );
    AlphaShapeStats myStats; // local to keep the counters out of the caller's memory in the loops below
    neis.clear();
    findPointsInBall( cloud, { cloud.points[v], sqr( data.searchRadius ) },
        [&]( const PointsProjectionResult & found, const Vector3f&, Ball3f & )
        {
            if ( v != found.vId )
            {
                const auto c = data.coords( cloud, found.vId );
                const Vector3i64 d{ c.pt - p0.pt };
                neis.push_back( { c, dot( Vector3i64mul{ d }, Vector3i64mul{ d } ) } );
            }
            return Processing::Continue;
        } );

    myStats.collectedNeis += neis.size();

    // the ball emptiness test below stops on the first point inside the ball,
    // and the points closest to #v have the best chance to be there
    std::sort( neis.begin(), neis.end(), []( const AlphaShapeNei & a, const AlphaShapeNei & b )
    {
        return a.distSq < b.distSq;
    } );

    // a neighbor p makes a farther neighbor x redundant if p is strictly inside every ball of the given
    // radius through #v having x inside or on it: then x can neither make a triangle with #v (p blocks
    // both its balls) nor be the only point blocking a triangle of others; see the PR for the derivation
    auto makesRedundant = [rSq4 = 4 * FastInt128( data.intRadiusSq )]( const Vector3i64 & p, const FastInt128 & pp, const Vector3i64 & x, const FastInt128 & xx )
    {
        const auto px = dot( Vector3i64mul{ p }, Vector3i64mul{ x } );
        if ( px <= pp )
            return false; // x is not behind the plane through p orthogonal to #v-p
        // every dot product here is at most 2^64 in magnitude, so all the products below
        // fit in three words, and the two final ones in four
        const auto crossSq = FastInt<192>( Int128Mul256( xx ) * Int128Mul256( pp ) - sqr( Int128Mul256( px ) ) ); // |cross(p,x)|^2
        const auto d = rSq4 - pp;
        if ( FastInt<192>( 4 * crossSq ) >= FastInt<192>( Int128Mul256( pp ) * Int128Mul256( d ) ) )
            return false; // x is not closer to line #v-p than the centers of the balls through #v and p
        // x is strictly outside every ball of the given radius through #v and p
        return pp * FastInt<192>( sqr( Int128Mul256( xx - px ) ) ) > d * crossSq;
    };
    size_t goodSize = neis.size();
    for ( size_t i = 0; i < goodSize; ++i )
    {
        const Vector3i64 p{ neis[i].coords.pt - p0.pt };
        size_t good = i + 1;
        for ( size_t j = i + 1; j < goodSize; ++j )
        {
            ++myStats.redundancyTests;
            if ( !makesRedundant( p, neis[i].distSq, Vector3i64{ neis[j].coords.pt - p0.pt }, neis[j].distSq ) )
                neis[good++] = neis[j];
        }
        goodSize = good;
    }
    myStats.redundantNeis += neis.size() - goodSize;
    neis.resize( goodSize );

    // the shadow filter below reuses the sphere quantities this tester computes in reset()
    FastInSphereTesterSoS tester;
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


    // the farther point of the pair is taken in the outer loop, so that the pairs of the closest
    // neighbours, whose shadows are the largest, cast them first and shorten the loops below
    for ( size_t j = 1; j < neis.size(); ++j )
    {
        const auto & pj = neis[j].coords;
        if ( onlyLargerVids && pj.id < v )
            continue;
        for ( size_t i = 0; i < j; ++i )
        {
            const auto & pi = neis[i].coords;
            if ( onlyLargerVids && pi.id < v )
                continue;
            // the two balls touching all three points differ only by the side of the triangle,
            // so one reset is enough for the both; the tester cheaply rejects the points farther than
            // the diameter from the first of the three, and every candidate is within the search
            // radius from p0, so p0 is not given first
            ++myStats.consideredTris;
            if ( !tester.reset( pj, p0, pi, data.intRadiusSq ) )
                continue;
            ++myStats.touchableTris;
            // the shadow depends only on the existence of the touching balls, not on their emptiness,
            // and dropping before the tests below shortens their scans as well
            dropShadowed( neis, i, j, p0.pt, tester, myStats );
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
