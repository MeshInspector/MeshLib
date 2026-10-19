#include <MRMesh/MRCutTrianglePredicates.h>
#include <MRMesh/MRFastInt.h>
#include <MRMesh/MRPrecisePredicates3Poly.h>
#include <MRMesh/MRTriMath.h>
#include <MRPch/MRTBB.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>

namespace MR
{

namespace
{

using Tri = std::array<PreciseVertCoords, 3>;

// the tests on many sets of points take a fifth of their trials in Debug, where the predicates and the exact checks are an order
// of magnitude slower, while Release keeps the full coverage
#ifdef NDEBUG
constexpr int cTrialDivisor = 1;
#else
constexpr int cTrialDivisor = 5;
#endif

// the least count of the cases of a kind that a test must meet, which differs in Debug with fewer trials
constexpr int atLeast( int inRelease, int inDebug )
{
    return cTrialDivisor == 1 ? inRelease : inDebug;
}

// random integer points in a cube, with ids in a random order, so that the perturbation orders the vertices variously;
// the ids are unique within one trial; the numbers come straight from mt19937, whose sequence the standard fixes,
// unlike the ones of std::uniform_int_distribution and std::shuffle, so every platform tests the same points
class RandomVerts
{
public:
    RandomVerts( unsigned seed, int halfSize, int numIds = 64 ) : gen_( seed ), halfSize_( halfSize )
    {
        ids_.resize( numIds );
        std::iota( ids_.begin(), ids_.end(), 0 );
    }
    void newTrial()
    {
        for ( size_t i = ids_.size() - 1; i > 0; --i )
            std::swap( ids_[i], ids_[gen_() % ( i + 1 )] );
        nextId_ = 0;
    }
    PreciseVertCoords vert()
    {
        assert( nextId_ < ids_.size() );
        const VertId id( ids_[nextId_++] );
        const int x = coord_();
        const int y = coord_();
        const int z = coord_();
        return { id, Vector3i( x, y, z ) };
    }
    Tri tri()
    {
        return { vert(), vert(), vert() };
    }
    // a random number from 0 to n-1
    int index( int n )
    {
        return int( gen_() % std::uint32_t( n ) );
    }

private:
    int coord_()
    {
        return int( gen_() % std::uint32_t( 2 * halfSize_ + 1 ) ) - halfSize_;
    }

    std::mt19937 gen_;
    int halfSize_ = 0;
    std::vector<int> ids_;
    size_t nextId_ = 0;
};

PreciseCutPoint vertexPoint( const PreciseVertCoords& v )
{
    PreciseCutPoint res;
    res.seg[0] = v;
    return res;
}

bool hasVert( const Tri& t, VertId v )
{
    return t[0].id == v || t[1].id == v || t[2].id == v;
}

Vector3i64 exactNormal( const Tri& t )
{
    return dirDblArea( Vector3i64( t[0].pt ), Vector3i64( t[1].pt ), Vector3i64( t[2].pt ) );
}

// the axes that the predicates project on, as the class documents them: the two least aligned with the normal
// (dropping the first one of equal largest components), ordered counterclockwise around it
std::pair<int, int> projectionAxes( const Tri& t )
{
    const auto n = exactNormal( t );
    int drop = 0;
    for ( int i = 1; i < 3; ++i )
        if ( std::abs( n[i] ) > std::abs( n[drop] ) )
            drop = i;
    int kx = ( drop + 1 ) % 3, ky = ( drop + 2 ) % 3;
    if ( n[drop] < 0 )
        std::swap( kx, ky );
    return { kx, ky };
}

Vector3d toDouble( const PreciseVertCoords& v )
{
    return Vector3d( v.pt );
}

Vector3d normalOf( const Tri& t )
{
    return dirDblArea( toDouble( t[0] ), toDouble( t[1] ), toDouble( t[2] ) );
}

// the position of a point in doubles
Vector3d positionOf( const PreciseCutPoint& p )
{
    if ( p.isVertex() )
        return toDouble( p.seg[0] );
    const Vector3d a = toDouble( p.seg[0] ), b = toDouble( p.seg[1] );
    const Vector3d n = normalOf( p.tri );
    const double s = dot( n, toDouble( p.tri[0] ) - a ) / dot( n, b - a );
    return a + s * ( b - a );
}

// the points of triangle f cut by other triangles: its vertices, the crossings of its sides with the others, and the crossings
// of their edges with it; bit s of lines[i] is set if point i is on side s of f, from vertex s to vertex s+1,
// and bit 3+j if it is on the cut by others[j], as the construction of the points tells, independently of the predicates
struct CutPoints
{
    std::vector<PreciseCutPoint> defs;
    std::vector<int> lines;

    // whether two of points p, q, r are on a common side or cut, which ccwByLines must know
    bool twoOnSideOrCut( int p, int q, int r ) const
    {
        return ( lines[p] & lines[q] ) || ( lines[q] & lines[r] ) || ( lines[r] & lines[p] );
    }
};

CutPoints cutPoints( const Tri& f, const std::vector<Tri>& others )
{
    CutPoints res;
    auto add = [&] ( const PreciseCutPoint& p, int lines )
    {
        res.defs.push_back( p );
        res.lines.push_back( lines );
    };
    for ( int s = 0; s < 3; ++s )
        add( vertexPoint( f[s] ), ( 1 << s ) | ( 1 << ( ( s + 2 ) % 3 ) ) );
    for ( int j = 0; j < int( others.size() ); ++j )
    {
        const Tri& t = others[j];
        for ( int s = 0; s < 3; ++s )
        {
            if ( doTriangleSegmentIntersect( { t[0], t[1], t[2], f[s], f[( s + 1 ) % 3] } ) )
                add( { { f[s], f[( s + 1 ) % 3] }, t }, ( 1 << s ) | ( 8 << j ) );
            if ( doTriangleSegmentIntersect( { f[0], f[1], f[2], t[s], t[( s + 1 ) % 3] } ) )
                add( { { t[s], t[( s + 1 ) % 3] }, f }, 8 << j );
        }
    }
    return res;
}

// the same for triangles that share vertices and edges with f and with one another, as in a mesh: a segment and a triangle
// with a common vertex are never tested for a crossing, as doTriangleSegmentIntersect cannot take them, so a triangle sharing
// vertex v with f cuts it from v to one crossing; an edge of several triangles crosses f at one point, on the cuts by all of them
CutPoints meshCutPoints( const Tri& f, const std::vector<Tri>& others )
{
    CutPoints res;
    auto add = [&] ( const PreciseCutPoint& p, int lines )
    {
        res.defs.push_back( p );
        res.lines.push_back( lines );
    };
    for ( int s = 0; s < 3; ++s )
    {
        int lines = ( 1 << s ) | ( 1 << ( ( s + 2 ) % 3 ) );
        for ( int j = 0; j < int( others.size() ); ++j )
            if ( hasVert( others[j], f[s].id ) )
                lines |= 8 << j;
        add( vertexPoint( f[s] ), lines );
    }
    for ( int j = 0; j < int( others.size() ); ++j )
    {
        const Tri& t = others[j];
        for ( int s = 0; s < 3; ++s )
        {
            const auto& a = f[s];
            const auto& b = f[( s + 1 ) % 3];
            if ( !hasVert( t, a.id ) && !hasVert( t, b.id ) && doTriangleSegmentIntersect( { t[0], t[1], t[2], a, b } ) )
                add( { { a, b }, t }, ( 1 << s ) | ( 8 << j ) );
        }
        for ( int e = 0; e < 3; ++e )
        {
            const auto& a = t[e];
            const auto& b = t[( e + 1 ) % 3];
            if ( hasVert( f, a.id ) || hasVert( f, b.id ) || !doTriangleSegmentIntersect( { f[0], f[1], f[2], a, b } ) )
                continue;
            auto sameEdge = [&] ( const PreciseCutPoint& p )
            {
                return !p.isVertex() && p.tri[0].id == f[0].id
                    && ( ( p.seg[0].id == a.id && p.seg[1].id == b.id ) || ( p.seg[0].id == b.id && p.seg[1].id == a.id ) );
            };
            const auto it = std::find_if( res.defs.begin(), res.defs.end(), sameEdge );
            if ( it != res.defs.end() )
                res.lines[it - res.defs.begin()] |= 8 << j;
            else
                add( { { a, b }, f }, 8 << j );
        }
    }
    return res;
}

// exact rational coordinates of a point with small integer coordinates: n / d, d > 0
struct SmallFraction
{
    Vector3i64 n;
    std::int64_t d = 1;
};

std::int64_t smallVolume( const Tri& t, const PreciseVertCoords& p )
{
    return std::int64_t( orient3dVolume( t[0].pt, t[1].pt, t[2].pt, p.pt ) );
}

// nullopt if the point is not defined
std::optional<SmallFraction> smallFraction( const PreciseCutPoint& p )
{
    if ( p.isVertex() )
        return SmallFraction{ Vector3i64( p.seg[0].pt ), 1 };
    const auto oa = smallVolume( p.tri, p.seg[0] ), ob = smallVolume( p.tri, p.seg[1] );
    SmallFraction res{ oa * Vector3i64( p.seg[1].pt ) - ob * Vector3i64( p.seg[0].pt ), oa - ob };
    if ( res.d == 0 )
        return {};
    if ( res.d < 0 )
    {
        res.n = -res.n;
        res.d = -res.d;
    }
    return res;
}

// -1, 0, 1 as coordinate k of a is less, equal, greater than that of b
int compareSmall( const SmallFraction& a, const SmallFraction& b, int k )
{
    const auto l = a.n[k] * b.d, r = b.n[k] * a.d;
    return l < r ? -1 : ( l > r ? 1 : 0 );
}

// the sign of the area of triangle abc projected on axes kx, ky
int orientSmall( const SmallFraction& a, const SmallFraction& b, const SmallFraction& c, int kx, int ky )
{
    // the determinant of the rows ( n[kx], n[ky], d ), which has the sign of the area, since every d is positive
    const std::int64_t det =
          a.n[kx] * ( b.n[ky] * c.d - b.d * c.n[ky] )
        - a.n[ky] * ( b.n[kx] * c.d - b.d * c.n[kx] )
        + a.d * ( b.n[kx] * c.n[ky] - b.n[ky] * c.n[kx] );
    return det > 0 ? 1 : ( det < 0 ? -1 : 0 );
}

// the same for coordinates up to 2^30, in integers wider than the predicates take
struct BigFraction
{
    FastInt256 n[3];
    FastInt256 d;
};

std::optional<BigFraction> bigFraction( const PreciseCutPoint& p )
{
    BigFraction res;
    if ( p.isVertex() )
    {
        for ( int k = 0; k < 3; ++k )
            res.n[k] = FastInt256( p.seg[0].pt[k] );
        res.d = FastInt256( 1 );
        return res;
    }
    const auto oa = orient3dVolume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[0].pt );
    const auto ob = orient3dVolume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[1].pt );
    if ( oa == ob )
        return {};
    for ( int k = 0; k < 3; ++k )
        res.n[k] = Int128Mul256( oa ) * Int128Mul256( p.seg[1].pt[k] ) - Int128Mul256( ob ) * Int128Mul256( p.seg[0].pt[k] );
    res.d = FastInt256( oa ) - FastInt256( ob );
    if ( res.d.sign() < 0 )
    {
        for ( auto& x : res.n )
            x = -x;
        res.d = -res.d;
    }
    return res;
}

int compareBig( const BigFraction& a, const BigFraction& b, int k )
{
    const auto l = a.n[k] * b.d, r = b.n[k] * a.d;
    return l < r ? -1 : ( r < l ? 1 : 0 );
}

int orientBig( const BigFraction& a, const BigFraction& b, const BigFraction& c, int kx, int ky )
{
    const auto det = a.n[kx] * ( b.n[ky] * c.d - b.d * c.n[ky] )
                   - a.n[ky] * ( b.n[kx] * c.d - b.d * c.n[kx] )
                   + a.d * ( b.n[kx] * c.n[ky] - b.n[ky] * c.n[kx] );
    return det.sign();
}

// the cases that checkOrder met
struct OrderStats
{
    int perturbed = 0; // pairs that coincide before the perturbation, or with unknown exact positions
    int perturbedNoLine = 0; // such pairs on no common side or cut
    int sidePairs = 0; // pairs of crossings of one side
    int sharedSidePairs = 0; // such pairs where the crossed triangles share vertices
};

// checks less and ccwFromBehind on all pairs of the points of f: they must be exact where the exact positions are known
// and differ, and consistent elsewhere: the order must be a strict total order that the expansion of the perturbation agrees with,
// and along a side it must be the order of the boolean's cut; compare( p, q, k ) gives -1, 0, 1 as exact coordinate k of p
// is less than, equal to, greater than that of q, or nullopt if unknown; flat is for f with a zero exact normal
template <typename Compare>
void checkOrder( const CutTrianglePredicates& preds, const Tri& f, const CutPoints& pts, int kx, int ky, bool flat,
    const Compare& compare, OrderStats& stats )
{
    const int n = int( pts.defs.size() );
    // -1, 0, 1 as the exact position of p comes before, at, after the one of q along the sweep, or nullopt if unknown
    auto compareSweep = [&] ( int p, int q ) -> std::optional<int>
    {
        const auto c = compare( p, q, kx );
        return !c || *c != 0 ? c : compare( p, q, ky );
    };

    std::vector<char> less( n * n );
    for ( int p = 0; p < n; ++p )
        for ( int q = 0; q < n; ++q )
            less[p * n + q] = preds.less( VertId( p ), VertId( q ) );
    for ( int p = 0; p < n; ++p )
    {
        EXPECT_FALSE( less[p * n + p] );
        for ( int q = p + 1; q < n; ++q )
        {
            EXPECT_NE( less[p * n + q], less[q * n + p] );
            if ( const auto c = compareSweep( p, q ); c && *c != 0 )
            {
                EXPECT_EQ( bool( less[p * n + q] ), *c < 0 );
                continue;
            }
            // coincident before the perturbation, or unknown: the expansion of the perturbation gives the direction from p to q,
            // which the sweep goes along iff its first coordinate is positive, or zero and the second one positive;
            // seen from far behind p against the sweep, q makes a counterclockwise turn iff it is below p, or level and ahead
            ++stats.perturbed;
            stats.perturbedNoLine += ( pts.lines[p] & pts.lines[q] ) == 0;
            const auto dir = preds.directionByPerturbation( VertId( p ), VertId( q ) );
            ASSERT_TRUE( dir.has_value() );
            EXPECT_EQ( bool( less[p * n + q] ), dir->x != 0 ? dir->x > 0 : dir->y > 0 );
            EXPECT_EQ( preds.ccwFromBehind( VertId( q ), VertId( p ) ), dir->y != 0 ? dir->y < 0 : dir->x > 0 );
            EXPECT_EQ( preds.ccwFromBehind( VertId( p ), VertId( q ) ), dir->y != 0 ? dir->y > 0 : dir->x < 0 );
        }
    }
    int numIntransitive = 0;
    for ( int p = 0; p < n; ++p )
        for ( int q = 0; q < n; ++q )
            for ( int r = 0; r < n; ++r )
                numIntransitive += less[p * n + q] && less[q * n + r] && !less[p * n + r];
    EXPECT_EQ( numIntransitive, 0 );

    // along side s of f, from vertex s to vertex s+1 (points s and s+1): the crossings of the side come between them,
    // in the order of segmentIntersectionOrder, and the sweep goes along all of them either way
    for ( int s = 0; s < 3; ++s )
    {
        const int s1 = ( s + 1 ) % 3;
        const bool alongSweep = less[s * n + s1] != 0;
        if ( !flat )
        {
            const Vector3i u = f[s1].pt - f[s].pt;
            EXPECT_EQ( alongSweep, u[kx] != 0 ? u[kx] > 0 : u[ky] > 0 );
        }
        for ( int p = 3; p < n; ++p )
        {
            if ( !( pts.lines[p] & ( 1 << s ) ) )
                continue;
            EXPECT_EQ( bool( less[s * n + p] ), alongSweep );
            EXPECT_EQ( bool( less[p * n + s1] ), alongSweep );
            for ( int q = 3; q < n; ++q )
            {
                if ( q == p || !( pts.lines[q] & ( 1 << s ) ) )
                    continue;
                const auto& a = pts.defs[p];
                const auto& b = pts.defs[q];
                const bool aFirst = segmentIntersectionOrder( { a.seg[0], a.seg[1], a.tri[0], a.tri[1], a.tri[2], b.tri[0], b.tri[1], b.tri[2] } );
                EXPECT_EQ( bool( less[p * n + q] ) == alongSweep, aFirst );
                ++stats.sidePairs;
                stats.sharedSidePairs += hasVert( a.tri, b.tri[0].id ) || hasVert( a.tri, b.tri[1].id ) || hasVert( a.tri, b.tri[2].id );
            }
        }
    }

    // seen from behind: exact where the points differ before the perturbation (the other pairs are checked above)
    for ( int p = 0; p < n; ++p )
        for ( int q = 0; q < n; ++q )
        {
            if ( p == q )
                continue;
            if ( const auto c = compareSweep( q, p ); c && *c != 0 )
            {
                const int cy = *compare( q, p, ky );
                EXPECT_EQ( preds.ccwFromBehind( VertId( q ), VertId( p ) ), cy != 0 ? cy < 0 : *c > 0 );
            }
        }
}

// the cases that checkCcw met
struct CcwStats
{
    int exact = 0; // triangles with a known exact orientation
    int perturbed = 0; // triangles on one line before the perturbation or with unknown positions, without two points on a known line
    int sides = 0; // triangles of three points on one side
    int lines = 0; // such triangles off the sides that ccwByLines decides
    int byPerturbation = 0; // triangles checked against the expansion of the perturbation
};

// checks ccw on all triangles of the points: decided without a failure, exact where orient( p, q, r ) gives the sign of the exact
// doubled area (0 if it is zero or unknown), with three points of one side bulging inward, and satisfying the axioms of Knuth's
// CC systems as a perturbation of the points in the plane does: cyclic symmetry, antisymmetry, interiority and transitivity;
// the expansion of the perturbation alone (ccwByPerturbation) gives the same on every triangle off the sides that is degenerate
// or unknown, so also where ccwByLines decides, and on every exactStride-th triangle with a known exact orientation
// (the points of one side stay on one line after the perturbation, which the expansion finds only after all its stages,
// too slowly for all such triangles here, see CutTrianglePredicatesCollinear)
template <typename Orient>
void checkCcw( const CutTrianglePredicates& preds, const CutPoints& pts, const Orient& orient, int exactStride, CcwStats& stats )
{
    const int n = int( pts.defs.size() );
    std::vector<char> table( n * n * n );
    auto ccw = [&] ( int p, int q, int r ) { return table[( p * n + q ) * n + r] != 0; };
    for ( int p = 0; p < n; ++p )
        for ( int q = 0; q < n; ++q )
            for ( int r = 0; r < n; ++r )
                if ( p != q && q != r && r != p )
                    table[( p * n + q ) * n + r] = preds.ccw( VertId( p ), VertId( q ), VertId( r ) );
    EXPECT_FALSE( preds.failed() );

    int numAsymmetric = 0, numWrongSides = 0, numWrongLines = 0, numWrongPerturbation = 0, numNotInterior = 0, numIntransitive = 0;
    for ( int p = 0; p < n; ++p )
        for ( int q = 0; q < n; ++q )
            for ( int r = 0; r < n; ++r )
            {
                if ( p == q || q == r || r == p )
                    continue;
                const bool c = ccw( p, q, r );
                numAsymmetric += c != ccw( q, r, p ) || c == ccw( p, r, q );
                if ( const int o = orient( p, q, r ) )
                {
                    EXPECT_EQ( c, o > 0 );
                    ++stats.exact;
                }
                if ( const int side = pts.lines[p] & pts.lines[q] & pts.lines[r] & 7 )
                {
                    // counterclockwise iff the points go backward along the side, from vertex s+1 to vertex s, up to a cyclic shift
                    const int s = std::countr_zero( unsigned( side ) );
                    const bool alongSweep = preds.less( VertId( s ), VertId( ( s + 1 ) % 3 ) );
                    auto along = [&] ( int i, int j ) { return preds.less( VertId( i ), VertId( j ) ) == alongSweep; };
                    numWrongSides += c != ( int( along( p, q ) ) + int( along( q, r ) ) + int( along( r, p ) ) == 1 );
                    ++stats.sides;
                }
            }
    int numExact = 0;
    for ( int p = 0; p < n; ++p )
        for ( int q = p + 1; q < n; ++q )
            for ( int r = q + 1; r < n; ++r )
            {
                if ( pts.lines[p] & pts.lines[q] & pts.lines[r] & 7 )
                    continue;
                const int o = orient( p, q, r );
                if ( o != 0 && ++numExact % exactStride != 0 )
                    continue;
                const bool c = ccw( p, q, r );
                numWrongPerturbation += preds.ccwByPerturbation( VertId( p ), VertId( q ), VertId( r ) ) != c;
                ++stats.byPerturbation;
                if ( o != 0 )
                    continue;
                if ( const auto l = preds.ccwByLines( VertId( p ), VertId( q ), VertId( r ) ) )
                {
                    numWrongLines += *l != c;
                    ++stats.lines;
                }
                else
                    ++stats.perturbed;
            }
    // interiority: if t is inside triangle pqr, which is counterclockwise then; cyclic symmetry lets p be the first of p, q, r
    for ( int t = 0; t < n; ++t )
        for ( int p = 0; p < n; ++p )
            for ( int q = p + 1; q < n; ++q )
                for ( int r = p + 1; r < n; ++r )
                    if ( t != p && t != q && t != r && q != r )
                        numNotInterior += ccw( t, q, r ) && ccw( p, t, r ) && ccw( p, q, t ) && !ccw( p, q, r );
    // transitivity: the points on the left of ray ts are ordered counterclockwise around t
    std::vector<int> left;
    for ( int t = 0; t < n; ++t )
        for ( int s = 0; s < n; ++s )
        {
            if ( s == t )
                continue;
            left.clear();
            for ( int p = 0; p < n; ++p )
                if ( p != t && p != s && ccw( t, s, p ) )
                    left.push_back( p );
            for ( int p : left )
                for ( int q : left )
                    if ( q != p && ccw( t, p, q ) )
                        for ( int r : left )
                            if ( r != p && r != q && ccw( t, q, r ) )
                                numIntransitive += !ccw( t, p, r );
        }
    EXPECT_EQ( numAsymmetric, 0 );
    EXPECT_EQ( numWrongSides, 0 );
    EXPECT_EQ( numWrongLines, 0 );
    EXPECT_EQ( numWrongPerturbation, 0 );
    EXPECT_EQ( numNotInterior, 0 );
    EXPECT_EQ( numIntransitive, 0 );
    EXPECT_FALSE( preds.failed() );
}

} // anonymous namespace

// Points of random triangles in general position: the predicates must agree with the positions in doubles
TEST( MRMesh, CutTrianglePredicatesRandom )
{
    RandomVerts rv( 0, 1 << 20 );
    CutTrianglePredicates preds;
    int numOrders = 0, numBehind = 0, numTriangles = 0, numLineTriangles = 0, numSideTriangles = 0, numNoLine = 0;
    for ( int trial = 0; trial < 100; ++trial )
    {
        rv.newTrial();
        const Tri f = rv.tri();
        preds.reset( f );
        const auto [kx, ky] = projectionAxes( f );
        {
            // the projection keeps the orientation around the normal
            const Vector3d a = toDouble( f[0] ), b = toDouble( f[1] ), c = toDouble( f[2] );
            EXPECT_GT( ( b[kx] - a[kx] ) * ( c[ky] - a[ky] ) - ( b[ky] - a[ky] ) * ( c[kx] - a[kx] ), 0 );
        }
        std::vector<Tri> others;
        for ( int i = 0; i < 6; ++i )
            others.push_back( rv.tri() );
        const auto pts = cutPoints( f, others );
        const int n = int( pts.defs.size() );
        std::vector<Vector3d> pos;
        for ( int i = 0; i < n; ++i )
        {
            preds.addPoint( VertId( i ), pts.defs[i] );
            pos.push_back( positionOf( pts.defs[i] ) );
        }

        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
            {
                if ( std::abs( pos[p][kx] - pos[q][kx] ) > 1e-3 )
                {
                    EXPECT_EQ( preds.less( VertId( p ), VertId( q ) ), pos[p][kx] < pos[q][kx] );
                    ++numOrders;
                }
                if ( std::abs( pos[p][ky] - pos[q][ky] ) > 1e-3 )
                {
                    // seen from far behind p against the sweep, q makes a counterclockwise turn iff it is below p
                    EXPECT_EQ( preds.ccwFromBehind( VertId( q ), VertId( p ) ), pos[q][ky] < pos[p][ky] );
                    ++numBehind;
                }
            }

        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
                for ( int r = 0; r < n; ++r )
                {
                    if ( p == q || q == r || r == p )
                        continue;
                    bool expected = false;
                    if ( const int side = pts.lines[p] & pts.lines[q] & pts.lines[r] & 7 )
                    {
                        // three points on one side, which bulges inward a little: counterclockwise iff they go along it backward,
                        // from vertex s+1 to vertex s, up to a cyclic shift
                        const int s = std::countr_zero( unsigned( side ) );
                        const Vector3d a = toDouble( f[s] ), u = toDouble( f[( s + 1 ) % 3] ) - a;
                        auto along = [&] ( int i, int j ) { return dot( pos[i] - a, u ) < dot( pos[j] - a, u ); };
                        expected = int( along( p, q ) ) + int( along( q, r ) ) + int( along( r, p ) ) == 1;
                        ++numSideTriangles;
                    }
                    else
                    {
                        const Vector2d a( pos[p][kx], pos[p][ky] ), b( pos[q][kx], pos[q][ky] ), c( pos[r][kx], pos[r][ky] );
                        const double area = cross( b - a, c - a );
                        if ( std::abs( area ) < 1e-9 * ( b - a ).length() * ( c - a ).length() )
                            continue;
                        expected = area > 0;
                    }
                    EXPECT_EQ( preds.ccw( VertId( p ), VertId( q ), VertId( r ) ), expected );
                    ++numTriangles;
                    // the lines known from the definitions of the points give the same where they decide, as they must
                    // for two points on a common side or cut
                    if ( const auto l = preds.ccwByLines( VertId( p ), VertId( q ), VertId( r ) ) )
                    {
                        EXPECT_EQ( *l, expected );
                        ++numLineTriangles;
                    }
                    else
                    {
                        EXPECT_FALSE( pts.twoOnSideOrCut( p, q, r ) );
                        ++numNoLine;
                    }
                }
        EXPECT_FALSE( preds.failed() );
    }
    EXPECT_GT( numOrders, 3000 );
    EXPECT_GT( numBehind, 3000 );
    EXPECT_GT( numTriangles, 25000 );
    EXPECT_GT( numLineTriangles, 20000 );
    EXPECT_GT( numSideTriangles, 100 );
    EXPECT_GT( numNoLine, 2500 );
}

// Points on a small grid, where many of them coincide and many triangles are degenerate before the perturbation, including
// segments in the planes of their triangles and triangles with zero normals: the predicates must be exact where the exact
// positions are known, and consistent elsewhere (see checkOrder)
TEST( MRMesh, CutTrianglePredicatesGrid )
{
    RandomVerts grid2( 1, 2 ), grid1( 2, 1 );
    CutTrianglePredicates preds;
    OrderStats stats;
    int numTriangles = 0, numFlat = 0, numInPlane = 0, numDegenerate = 0, numExactNoLine = 0;
    for ( int trial = 0; trial < 800 / cTrialDivisor; ++trial )
    {
        // the coarser grid gives more segments in the planes of triangles and more flat triangles
        auto& rv = trial % 2 ? grid1 : grid2;
        rv.newTrial();
        const Tri f = rv.tri();
        preds.reset( f );
        // a triangle with a zero normal, which only the perturbation orients and projects
        const bool flat = exactNormal( f ) == Vector3i64();
        ++( flat ? numFlat : numTriangles );
        const auto [kx, ky] = projectionAxes( f );
        std::vector<Tri> others;
        for ( int i = 0; i < 4; ++i )
            others.push_back( rv.tri() );
        const auto pts = cutPoints( f, others );
        const int n = int( pts.defs.size() );
        // the exact positions in the projection, unknown on a flat triangle and for a segment in the plane of its triangle
        std::vector<std::optional<SmallFraction>> pos;
        for ( int i = 0; i < n; ++i )
        {
            preds.addPoint( VertId( i ), pts.defs[i] );
            const auto x = smallFraction( pts.defs[i] );
            numInPlane += !x;
            pos.push_back( flat ? std::nullopt : x );
        }
        checkOrder( preds, f, pts, kx, ky, flat, [&] ( int p, int q, int k ) -> std::optional<int>
        {
            if ( !pos[p] || !pos[q] )
                return {};
            return compareSmall( *pos[p], *pos[q], k );
        }, stats );

        // a triangle with two vertices on a known line, as two points on a common side or cut are: decided by that line, exact unless
        // degenerate before the perturbation or unknown, the same in any rotation of its vertices, and the opposite when flipped; ccw agrees;
        // and any triangle with a known exact orientation, also without two of its vertices on a known line
        for ( int p = 0; p < n; ++p )
            for ( int q = p + 1; q < n; ++q )
                for ( int r = q + 1; r < n; ++r )
                {
                    const int o = pos[p] && pos[q] && pos[r] ? orientSmall( *pos[p], *pos[q], *pos[r], kx, ky ) : 0;
                    if ( o != 0 )
                    {
                        EXPECT_EQ( preds.ccw( VertId( p ), VertId( q ), VertId( r ) ), o > 0 );
                        numExactNoLine += !pts.twoOnSideOrCut( p, q, r );
                    }
                    const auto c = preds.ccwByLines( VertId( p ), VertId( q ), VertId( r ) );
                    if ( !c )
                    {
                        EXPECT_FALSE( pts.twoOnSideOrCut( p, q, r ) );
                        continue;
                    }
                    EXPECT_EQ( preds.ccwByLines( VertId( q ), VertId( r ), VertId( p ) ), *c );
                    EXPECT_EQ( preds.ccwByLines( VertId( r ), VertId( p ), VertId( q ) ), *c );
                    EXPECT_EQ( preds.ccwByLines( VertId( q ), VertId( p ), VertId( r ) ), !*c );
                    EXPECT_EQ( preds.ccw( VertId( p ), VertId( q ), VertId( r ) ), *c );
                    if ( o != 0 )
                        EXPECT_EQ( *c, o > 0 );
                    else
                        ++numDegenerate;
                }
        EXPECT_FALSE( preds.failed() );
    }
    EXPECT_GE( numTriangles, atLeast( 600, 120 ) );
    EXPECT_GE( numFlat, atLeast( 50, 10 ) );
    EXPECT_GE( numInPlane, atLeast( 90, 18 ) );
    EXPECT_GE( stats.perturbed, atLeast( 1300, 250 ) );
    EXPECT_GE( stats.perturbedNoLine, atLeast( 530, 100 ) );
    EXPECT_GE( stats.sidePairs, atLeast( 300, 40 ) );
    EXPECT_GE( numDegenerate, atLeast( 4800, 900 ) );
    EXPECT_GE( numExactNoLine, atLeast( 1900, 350 ) );
}

// ccw on all triangles of the points of small sets on a grid, also on one line before the perturbation without two points
// on a known line, and on flat triangles, where all points are on one line before the perturbation: see checkCcw;
// in every other set, f is a large triangle in plane z = 0 and the other triangles have their vertices in planes z = -1 and z = 1,
// so that their edges cross f at the halves of grid points, where many points coincide and many triples are on one line
// on no known one
TEST( MRMesh, CutTrianglePredicatesCcwAxioms )
{
    RandomVerts grid2( 3, 2 ), grid1( 4, 1 );
    CutTrianglePredicates preds;
    CcwStats stats;
    int numFlat = 0;
    for ( int trial = 0; trial < 120 / cTrialDivisor; ++trial )
    {
        const bool layers = trial % 2 == 0;
        auto& rv = trial % 4 == 1 ? grid1 : grid2;
        rv.newTrial();
        Tri f = rv.tri();
        std::vector<Tri> others;
        for ( int i = 0; i < ( layers ? 4 : 3 ); ++i )
            others.push_back( rv.tri() );
        if ( layers )
        {
            // f around the grid, in either orientation
            const Vector3i corners[3] = { { -5, -5, 0 }, { 11, -5, 0 }, { -5, 11, 0 } };
            const int step = rv.index( 2 ) + 1;
            for ( int i = 0; i < 3; ++i )
                f[i].pt = corners[i * step % 3];
            for ( auto& t : others )
                for ( auto& v : t )
                    v.pt.z = 2 * rv.index( 2 ) - 1;
        }
        preds.reset( f );
        const bool flat = exactNormal( f ) == Vector3i64();
        numFlat += flat;
        const auto [kx, ky] = projectionAxes( f );
        const auto pts = cutPoints( f, others );
        const int n = int( pts.defs.size() );
        std::vector<std::optional<SmallFraction>> pos;
        for ( int i = 0; i < n; ++i )
        {
            preds.addPoint( VertId( i ), pts.defs[i] );
            pos.push_back( flat ? std::nullopt : smallFraction( pts.defs[i] ) );
        }
        checkCcw( preds, pts, [&] ( int p, int q, int r )
        {
            return pos[p] && pos[q] && pos[r] ? orientSmall( *pos[p], *pos[q], *pos[r], kx, ky ) : 0;
        }, 10, stats );
    }
    EXPECT_GE( numFlat, atLeast( 10, 2 ) );
    EXPECT_GE( stats.exact, atLeast( 28000, 7000 ) );
    EXPECT_GE( stats.perturbed, atLeast( 50, 14 ) );
    EXPECT_GE( stats.sides, atLeast( 440, 24 ) );
    EXPECT_GE( stats.lines, atLeast( 320, 60 ) );
    EXPECT_GE( stats.byPerturbation, atLeast( 800, 180 ) );
}

// Triangles of a mesh: all vertices come from a small pool on a grid, so the triangles cutting f share vertices and edges with
// one another and with f, as in selfBoolean, which cuts f along folds from its vertices (see meshCutPoints); two crossings
// of a side with triangles sharing vertices take the shared-vertex path of segmentIntersectionOrder, and two crossings of edges
// with a common vertex are on the plane through the edges
TEST( MRMesh, CutTrianglePredicatesSharedVertices )
{
    RandomVerts rv( 5, 2 );
    CutTrianglePredicates preds;
    OrderStats orderStats;
    CcwStats ccwStats;
    int numFlat = 0, numFolds = 0, numSharedEdges = 0, numEdgePlanes = 0;
    for ( int trial = 0; trial < 300 / cTrialDivisor; ++trial )
    {
        rv.newTrial();
        constexpr int cPool = 8;
        std::vector<PreciseVertCoords> pool;
        for ( int i = 0; i < cPool; ++i )
            pool.push_back( rv.vert() );
        // f and seven other triangles, all different
        std::vector<Tri> tris;
        while ( tris.size() < 8 )
        {
            const Tri t{ pool[rv.index( cPool )], pool[rv.index( cPool )], pool[rv.index( cPool )] };
            if ( t[0].id == t[1].id || t[1].id == t[2].id || t[2].id == t[0].id )
                continue;
            if ( std::none_of( tris.begin(), tris.end(), [&] ( const Tri& u ) { return hasVert( u, t[0].id ) && hasVert( u, t[1].id ) && hasVert( u, t[2].id ); } ) )
                tris.push_back( t );
        }
        const Tri f = tris[0];
        const std::vector<Tri> others( tris.begin() + 1, tris.end() );
        preds.reset( f );
        const bool flat = exactNormal( f ) == Vector3i64();
        numFlat += flat;
        const auto [kx, ky] = projectionAxes( f );
        const auto pts = meshCutPoints( f, others );
        const int n = int( pts.defs.size() );
        std::vector<std::optional<SmallFraction>> pos;
        for ( int i = 0; i < n; ++i )
        {
            preds.addPoint( VertId( i ), pts.defs[i] );
            pos.push_back( flat ? std::nullopt : smallFraction( pts.defs[i] ) );
            // a point on the cuts by two triangles: a crossing of their common edge, or a vertex of f shared by them
            numSharedEdges += i >= 3 && std::popcount( unsigned( pts.lines[i] >> 3 ) ) > 1;
        }
        // a vertex of f and a crossing on one cut: the fold along a triangle of the vertex
        for ( int i = 0; i < 3; ++i )
            for ( int j = 3; j < n; ++j )
                numFolds += ( pts.lines[i] & pts.lines[j] & ~7 ) != 0;
        // crossings of two edges with a common vertex, on no common cut: on the plane through the edges
        for ( int i = 3; i < n; ++i )
            for ( int j = i + 1; j < n; ++j )
            {
                const auto& a = pts.defs[i].seg;
                const auto& b = pts.defs[j].seg;
                numEdgePlanes += ( pts.lines[i] & 7 ) == 0 && ( pts.lines[j] & 7 ) == 0 && ( pts.lines[i] & pts.lines[j] ) == 0
                    && ( a[0].id == b[0].id || a[0].id == b[1].id || a[1].id == b[0].id || a[1].id == b[1].id );
            }

        checkOrder( preds, f, pts, kx, ky, flat, [&] ( int p, int q, int k ) -> std::optional<int>
        {
            if ( !pos[p] || !pos[q] )
                return {};
            return compareSmall( *pos[p], *pos[q], k );
        }, orderStats );
        checkCcw( preds, pts, [&] ( int p, int q, int r )
        {
            return pos[p] && pos[q] && pos[r] ? orientSmall( *pos[p], *pos[q], *pos[r], kx, ky ) : 0;
        }, 10, ccwStats );
        EXPECT_FALSE( preds.failed() );
    }
    EXPECT_GE( numFlat, atLeast( 6, 0 ) );
    EXPECT_GE( numFolds, atLeast( 200, 36 ) );
    EXPECT_GE( numSharedEdges, atLeast( 40, 11 ) );
    EXPECT_GE( numEdgePlanes, atLeast( 40, 12 ) );
    EXPECT_GE( orderStats.perturbed, atLeast( 85, 8 ) );
    EXPECT_GE( orderStats.sharedSidePairs, atLeast( 110, 25 ) );
    EXPECT_GE( ccwStats.exact, atLeast( 11000, 2700 ) );
    EXPECT_GE( ccwStats.lines, atLeast( 220, 36 ) );
    EXPECT_GE( ccwStats.byPerturbation, atLeast( 330, 65 ) );
}

// Coordinates near 2^30, the limit for which the error bounds of the rounded coordinates and the widths of the exact integers
// are derived: a coarse grid scaled by 2^30 - 2^24, with every fourth vertex moved by a few units, so that many points coincide
// or nearly coincide and many triangles are degenerate or nearly so; and random vertices in the whole range; the predicates
// must agree with the exact positions, found here in wider integers than the predicates take
TEST( MRMesh, CutTrianglePredicatesLargeCoords )
{
    constexpr int cScale = ( 1 << 30 ) - ( 1 << 24 );
    RandomVerts grid( 6, 1 ), wide( 7, ( 1 << 30 ) - 1 );
    std::mt19937 gen( 8 );
    auto scaled = [&] ( Tri t )
    {
        for ( auto& v : t )
        {
            const bool move = gen() % 4 == 0;
            for ( int i = 0; i < 3; ++i )
                v.pt[i] = v.pt[i] * cScale + ( move ? int( gen() % 9 ) - 4 : 0 );
        }
        return t;
    };
    CutTrianglePredicates preds;
    OrderStats orderStats;
    CcwStats ccwStats;
    int numPoints = 0, numNear = 0;
    for ( int trial = 0; trial < 150 / cTrialDivisor; ++trial )
    {
        const bool onGrid = trial % 2 == 0;
        auto& rv = onGrid ? grid : wide;
        rv.newTrial();
        const Tri f = onGrid ? scaled( rv.tri() ) : rv.tri();
        std::vector<Tri> others;
        for ( int i = 0; i < 4; ++i )
            others.push_back( onGrid ? scaled( rv.tri() ) : rv.tri() );
        preds.reset( f );
        const bool flat = exactNormal( f ) == Vector3i64();
        const auto [kx, ky] = projectionAxes( f );
        const auto pts = cutPoints( f, others );
        const int n = int( pts.defs.size() );
        numPoints += n;
        std::vector<std::optional<BigFraction>> pos;
        for ( int i = 0; i < n; ++i )
        {
            preds.addPoint( VertId( i ), pts.defs[i] );
            pos.push_back( flat ? std::nullopt : bigFraction( pts.defs[i] ) );
        }
        auto compare = [&] ( int p, int q, int k ) -> std::optional<int>
        {
            if ( !pos[p] || !pos[q] )
                return {};
            return compareBig( *pos[p], *pos[q], k );
        };
        // pairs of different points closer than a unit, of which the coordinates in doubles have rounding errors of about 2^-23
        for ( int p = 0; p < n; ++p )
            for ( int q = p + 1; q < n; ++q )
                if ( const auto c = compare( p, q, kx ); c && *c != 0 )
                    numNear += std::abs( positionOf( pts.defs[p] )[kx] - positionOf( pts.defs[q] )[kx] ) < 1;
        checkOrder( preds, f, pts, kx, ky, flat, compare, orderStats );
        checkCcw( preds, pts, [&] ( int p, int q, int r )
        {
            return pos[p] && pos[q] && pos[r] ? orientBig( *pos[p], *pos[q], *pos[r], kx, ky ) : 0;
        }, 10, ccwStats );
        EXPECT_FALSE( preds.failed() );
    }
    EXPECT_GE( numPoints, atLeast( 640, 140 ) );
    EXPECT_GE( numNear, atLeast( 14, 2 ) );
    EXPECT_GE( orderStats.perturbed, atLeast( 125, 30 ) );
    EXPECT_GE( orderStats.sidePairs, atLeast( 60, 20 ) );
    EXPECT_GE( ccwStats.exact, atLeast( 13500, 4000 ) );
    EXPECT_GE( ccwStats.perturbed, atLeast( 9, 4 ) );
    EXPECT_GE( ccwStats.lines, atLeast( 300, 67 ) );
}

// less called concurrently, as the parallel sort of the sweep calls it, so the directions found by expanding the perturbation
// are cached from several threads: on a coarse grid, where many points coincide, the parallel sort gives the order
// of a sequential sort
TEST( MRMesh, CutTrianglePredicatesParallelSort )
{
    RandomVerts rv( 9, 1, 1024 );
    rv.newTrial();
    const Tri f = rv.tri();
    std::vector<Tri> others;
    for ( int i = 0; i < 300; ++i )
        others.push_back( rv.tri() );
    const auto pts = cutPoints( f, others );
    const int n = int( pts.defs.size() );
    EXPECT_GT( n, 100 );
    CutTrianglePredicates preds;
    auto setup = [&]
    {
        preds.reset( f ); // with an empty cache
        for ( int i = 0; i < n; ++i )
            preds.addPoint( VertId( i ), pts.defs[i] );
    };
    auto less = [&] ( VertId a, VertId b ) { return preds.less( a, b ); };
    // every point several times, for a sequence long enough to be sorted in parallel
    std::vector<VertId> input;
    for ( int k = 0; k < 6; ++k )
        for ( int i = 0; i < n; ++i )
            input.push_back( VertId( i ) );
    setup();
    auto sorted = input;
    std::sort( sorted.begin(), sorted.end(), less );
    EXPECT_FALSE( preds.failed() );
    for ( int rep = 0; rep < 3; ++rep )
    {
        setup();
        auto v = input;
        std::rotate( v.begin(), v.begin() + rep * n / 3, v.end() );
        tbb::parallel_sort( v.begin(), v.end(), less );
        EXPECT_EQ( v, sorted );
        EXPECT_FALSE( preds.failed() );
    }
}

// setting a point again: the answers follow its new definition, also for pairs whose order the expansion of the perturbation
// found for the old definition
TEST( MRMesh, CutTrianglePredicatesSetAgain )
{
    RandomVerts rv( 10, 1 );
    CutTrianglePredicates preds, fresh;
    int numSwapped = 0;
    for ( int trial = 0; trial < 400 && numSwapped < 30; ++trial )
    {
        rv.newTrial();
        const Tri f = rv.tri();
        if ( exactNormal( f ) == Vector3i64() )
            continue;
        std::vector<Tri> others;
        for ( int i = 0; i < 6; ++i )
            others.push_back( rv.tri() );
        auto pts = cutPoints( f, others );
        const int n = int( pts.defs.size() );
        // two points that coincide before the perturbation, on no known line, which only the expansion of the perturbation orders
        int p = -1, q = -1;
        for ( int i = 3; i < n && p < 0; ++i )
            for ( int j = i + 1; j < n && p < 0; ++j )
            {
                const auto a = smallFraction( pts.defs[i] ), b = smallFraction( pts.defs[j] );
                if ( a && b && compareSmall( *a, *b, 0 ) == 0 && compareSmall( *a, *b, 1 ) == 0 && compareSmall( *a, *b, 2 ) == 0 && !( pts.lines[i] & pts.lines[j] ) )
                {
                    p = i;
                    q = j;
                }
            }
        if ( p < 0 )
            continue;
        preds.reset( f );
        for ( int i = 0; i < n; ++i )
            preds.addPoint( VertId( i ), pts.defs[i] );
        const bool pqBefore = preds.less( VertId( p ), VertId( q ) );
        for ( int i = 0; i < n; ++i )
            for ( int j = 0; j < n; ++j )
                if ( i != j )
                    (void)preds.ccwFromBehind( VertId( i ), VertId( j ) );
        // swap the definitions of p and q
        preds.addPoint( VertId( p ), pts.defs[q] );
        preds.addPoint( VertId( q ), pts.defs[p] );
        EXPECT_NE( preds.less( VertId( p ), VertId( q ) ), pqBefore );
        std::swap( pts.defs[p], pts.defs[q] );
        fresh.reset( f );
        for ( int i = 0; i < n; ++i )
            fresh.addPoint( VertId( i ), pts.defs[i] );
        for ( int i = 0; i < n; ++i )
            for ( int j = 0; j < n; ++j )
            {
                if ( i == j )
                    continue;
                EXPECT_EQ( preds.less( VertId( i ), VertId( j ) ), fresh.less( VertId( i ), VertId( j ) ) );
                EXPECT_EQ( preds.ccwFromBehind( VertId( i ), VertId( j ) ), fresh.ccwFromBehind( VertId( i ), VertId( j ) ) );
            }
        EXPECT_FALSE( preds.failed() );
        EXPECT_FALSE( fresh.failed() );
        ++numSwapped;
    }
    EXPECT_EQ( numSwapped, 30 );
}

// selfBoolean cuts f by a triangle sharing a vertex with it along a line from that vertex (a fold), so the sweep asks
// the orientation of that vertex, a crossing on the fold and a point of another cut
TEST( MRMesh, CutTrianglePredicatesFoldLine )
{
    auto vert = [] ( int id, int x, int y, int z ) { return PreciseVertCoords{ VertId( id ), Vector3i( x, y, z ) }; };
    const Tri f = { vert( 0, 0, 0, 0 ), vert( 1, 8, 0, 0 ), vert( 2, 0, 8, 0 ) };
    // triangles of vertex 0: an edge of the first crosses f at ( 2, 1 ), the second crosses side 1 of f at ( 2, 6 )
    const Tri t1 = { f[0], vert( 3, 2, 1, -1 ), vert( 4, 2, 1, 1 ) };
    const Tri t2 = { f[0], vert( 5, 6, 18, -4 ), vert( 6, 6, 18, 4 ) };
    // an edge of a triangle without vertices of f crosses it at ( 5, 1 )
    const Tri t3 = { vert( 7, 5, 1, -1 ), vert( 8, 5, 1, 1 ), vert( 9, 9, 9, 9 ) };

    CutTrianglePredicates preds;
    preds.reset( f );
    for ( int i = 0; i < 3; ++i )
        preds.addPoint( VertId( i ), vertexPoint( f[i] ) );
    preds.addPoint( 3_v, { { t1[1], t1[2] }, f } );
    preds.addPoint( 4_v, { { f[1], f[2] }, t2 } );
    preds.addPoint( 5_v, { { t3[0], t3[1] }, f } );

    // the exact positions decide ccw, and the lines through the vertex and a crossing decide ccwByLines
    auto check = [&] ( VertId a, VertId b, VertId c, bool expected )
    {
        EXPECT_EQ( preds.ccw( a, b, c ), expected );
        EXPECT_EQ( preds.ccwByLines( a, b, c ), expected );
    };
    check( 0_v, 3_v, 5_v, false );
    check( 0_v, 5_v, 3_v, true );
    check( 0_v, 4_v, 5_v, false );
    check( 5_v, 4_v, 0_v, true );
    EXPECT_FALSE( preds.failed() );
}

// points without a known line through two of them: crossings of edges of unrelated triangles with f
TEST( MRMesh, CutTrianglePredicatesWithoutLines )
{
    auto vert = [] ( int id, int x, int y, int z ) { return PreciseVertCoords{ VertId( id ), Vector3i( x, y, z ) }; };
    constexpr int X = 1 << 27;
    const Tri f = { vert( 0, -1, -1, 0 ), vert( 1, 3 * X, -1, 0 ), vert( 2, -1, 3 * X, 0 ) };
    CutTrianglePredicates preds;
    preds.reset( f );
    auto addCrossing = [&] ( VertId v, int id, Vector3i a, Vector3i b )
    {
        preds.addPoint( v, { { PreciseVertCoords{ VertId( id ), a }, PreciseVertCoords{ VertId( id + 1 ), b } }, f } );
    };
    // at ( 0, 0 ), ( X, 1 / 100001 ), ( 2X, 3 / 100001 ) and ( 2X, 2 / 100001 )
    addCrossing( 0_v, 10, { 0, 0, -1 }, { 0, 0, 1 } );
    addCrossing( 1_v, 12, { X, 0, -1 }, { X, 1, 100000 } );
    addCrossing( 2_v, 14, { 2 * X, 0, -1 }, { 2 * X, 3, 100000 } );
    addCrossing( 3_v, 16, { 2 * X, 0, -1 }, { 2 * X, 2, 100000 } );

    // the doubled area X / 100001 of triangle 012 is below the rounding errors of the coordinates of this size,
    // so the exact orientation decides
    EXPECT_FALSE( preds.ccwByLines( 0_v, 1_v, 2_v ).has_value() );
    EXPECT_TRUE( preds.ccw( 0_v, 1_v, 2_v ) );
    EXPECT_FALSE( preds.ccw( 0_v, 2_v, 1_v ) );
    EXPECT_FALSE( preds.failed() );

    // two points at ( 5, 5 ), which coincide before the perturbation: seen from behind, opposite for the edge taken backward
    addCrossing( 4_v, 18, { 5, 5, -1 }, { 5, 5, 1 } );
    addCrossing( 5_v, 20, { 4, 4, -1 }, { 6, 6, 1 } );
    const auto dir = preds.directionByPerturbation( 4_v, 5_v );
    ASSERT_TRUE( dir.has_value() );
    EXPECT_EQ( preds.less( 4_v, 5_v ), dir->x != 0 ? dir->x > 0 : dir->y > 0 );
    EXPECT_EQ( preds.ccwFromBehind( 5_v, 4_v ), dir->y != 0 ? dir->y < 0 : dir->x > 0 );
    EXPECT_NE( preds.ccwFromBehind( 5_v, 4_v ), preds.ccwFromBehind( 4_v, 5_v ) );
    EXPECT_FALSE( preds.failed() );

    // 0, 1, 3 are on one line before the perturbation, and no two of them on a known one: the expansion of the perturbation decides
    EXPECT_FALSE( preds.ccwByLines( 0_v, 1_v, 3_v ).has_value() );
    const bool c013 = preds.ccw( 0_v, 1_v, 3_v );
    EXPECT_EQ( preds.ccw( 1_v, 3_v, 0_v ), c013 );
    EXPECT_EQ( preds.ccw( 3_v, 0_v, 1_v ), c013 );
    EXPECT_NE( preds.ccw( 0_v, 3_v, 1_v ), c013 );
    EXPECT_FALSE( preds.failed() );
}

// three crossings at ( 0, 0 ), ( 1, 1 ), ( 2, 2 ) on no known line, where the perturbation of the vertex with the smallest id decides:
// vertex 0 moves to ( eps^9, eps^3, -1 + eps ), which moves the first crossing to ( eps^9, eps^3 ) / ( 2 - eps ), above the line y = x
TEST( MRMesh, CutTrianglePredicatesCollinear )
{
    auto vert = [] ( int id, int x, int y, int z ) { return PreciseVertCoords{ VertId( id ), Vector3i( x, y, z ) }; };
    const Tri f = { vert( 10, -1, -1, 0 ), vert( 11, 8, -1, 0 ), vert( 12, -1, 8, 0 ) };
    CutTrianglePredicates preds;
    preds.reset( f );
    for ( int i = 0; i < 3; ++i )
        preds.addPoint( VertId( i ), { { vert( 2 * i, i, i, -1 ), vert( 2 * i + 1, i, i, 1 ) }, f } );
    EXPECT_FALSE( preds.ccwByLines( 0_v, 1_v, 2_v ).has_value() );
    EXPECT_TRUE( preds.ccw( 0_v, 1_v, 2_v ) );
    EXPECT_TRUE( preds.ccw( 1_v, 2_v, 0_v ) );
    EXPECT_TRUE( preds.ccw( 2_v, 0_v, 1_v ) );
    EXPECT_FALSE( preds.ccw( 0_v, 2_v, 1_v ) );
    EXPECT_FALSE( preds.ccw( 2_v, 1_v, 0_v ) );
    EXPECT_EQ( preds.ccwByPerturbation( 0_v, 1_v, 2_v ), true );
    EXPECT_EQ( preds.ccwByPerturbation( 2_v, 1_v, 0_v ), false );

    // two vertices of f and a crossing of the side between them with a triangle, which stay on one line after the perturbation,
    // where they bulge inward: the vertices of side 0 come in the order of the side, so the triangle is clockwise
    preds.addPoint( 3_v, vertexPoint( f[0] ) );
    preds.addPoint( 4_v, vertexPoint( f[1] ) );
    preds.addPoint( 5_v, { { f[0], f[1] }, { vert( 20, 3, -5, -5 ), vert( 21, 3, -5, 5 ), vert( 22, 3, 5, 0 ) } } );
    EXPECT_FALSE( preds.ccwByPerturbation( 3_v, 5_v, 4_v ).has_value() );
    EXPECT_FALSE( preds.ccw( 3_v, 5_v, 4_v ) );
    EXPECT_TRUE( preds.ccw( 4_v, 5_v, 3_v ) );
    EXPECT_FALSE( preds.failed() );
}

// the same three crossings with the smallest ids at the vertices of f: perturbing them tilts the plane of f, which moves the crossings
// of the vertical segments along the segments only, so the projected points stay on one line at the first three stages of the expansion,
// and the fourth one, which perturbs the lower end of the first segment (as the smallest id did above), decides by the shortcut for
// a stage after a zero one, which takes only the products with the vertex perturbed at that stage; the same holds for two crossings
// at ( 5, 5 ), which only that vertex moves apart: the first crossing to ( eps^9, eps^3 ) / 2 from the second one, of which the sweep
// takes only the larger term
TEST( MRMesh, CutTrianglePredicatesCollinearLate )
{
    auto vert = [] ( int id, int x, int y, int z ) { return PreciseVertCoords{ VertId( id ), Vector3i( x, y, z ) }; };
    const Tri f = { vert( 0, -1, -1, 0 ), vert( 1, 8, -1, 0 ), vert( 2, -1, 8, 0 ) };
    CutTrianglePredicates preds;
    preds.reset( f );
    for ( int i = 0; i < 3; ++i )
        preds.addPoint( VertId( i ), { { vert( 10 + 2 * i, i, i, -1 ), vert( 11 + 2 * i, i, i, 1 ) }, f } );
    EXPECT_FALSE( preds.ccwByLines( 0_v, 1_v, 2_v ).has_value() );
    EXPECT_EQ( preds.ccwByPerturbation( 0_v, 1_v, 2_v ), true );
    EXPECT_EQ( preds.ccwByPerturbation( 1_v, 0_v, 2_v ), false );
    EXPECT_TRUE( preds.ccw( 0_v, 1_v, 2_v ) );
    EXPECT_FALSE( preds.ccw( 0_v, 2_v, 1_v ) );

    preds.addPoint( 3_v, { { vert( 30, 5, 5, -1 ), vert( 31, 5, 5, 1 ) }, f } );
    preds.addPoint( 4_v, { { vert( 32, 5, 5, -2 ), vert( 33, 5, 5, 2 ) }, f } );
    EXPECT_EQ( preds.directionByPerturbation( 3_v, 4_v ), Vector2i( 0, -1 ) );
    EXPECT_EQ( preds.directionByPerturbation( 4_v, 3_v ), Vector2i( 0, 1 ) );
    EXPECT_TRUE( preds.less( 4_v, 3_v ) );
    EXPECT_FALSE( preds.less( 3_v, 4_v ) );
    EXPECT_FALSE( preds.failed() );
}

// the inputs that break the preconditions, and points that the predicates cannot tell apart, make the predicates fail
TEST( MRMesh, CutTrianglePredicatesFailed )
{
    auto vert = [] ( int id, int x, int y, int z ) { return PreciseVertCoords{ VertId( id ), Vector3i( x, y, z ) }; };
    const Tri f = { vert( 0, 0, 0, 0 ), vert( 1, 8, 0, 0 ), vert( 2, 0, 8, 0 ) };
    CutTrianglePredicates preds;
    auto start = [&]
    {
        preds.reset( f );
        for ( int i = 0; i < 3; ++i )
            preds.addPoint( VertId( i ), vertexPoint( f[i] ) );
        EXPECT_FALSE( preds.failed() );
    };

    // a vertex point that is not a corner of the triangle
    start();
    preds.addPoint( 3_v, vertexPoint( vert( 9, 1, 1, 0 ) ) );
    EXPECT_TRUE( preds.failed() );

    // a segment that does not cross the plane of its triangle
    start();
    preds.addPoint( 3_v, { { vert( 10, 1, 1, 1 ), vert( 11, 2, 2, 3 ) }, f } );
    EXPECT_TRUE( preds.failed() );

    // a segment in the plane of its triangle that does not cross it after the perturbation either: the vertex of f with
    // the smallest id moves most, in z, and so it tilts the plane, which puts both ends of the segment on one side
    start();
    preds.addPoint( 3_v, { { vert( 10, 1, 1, 0 ), vert( 11, 2, 1, 0 ) }, f } );
    EXPECT_TRUE( preds.failed() );

    // repeated ids: in the triangle, in a segment, in the triangle of a crossing, and an end of a segment at a vertex of its triangle
    preds.reset( { f[0], f[1], vert( 1, 0, 8, 0 ) } );
    EXPECT_TRUE( preds.failed() );
    start();
    preds.addPoint( 3_v, { { vert( 10, 1, 1, -1 ), vert( 10, 1, 1, 1 ) }, f } );
    EXPECT_TRUE( preds.failed() );
    start();
    preds.addPoint( 3_v, { { vert( 10, 1, 1, -1 ), vert( 11, 1, 1, 1 ) }, { f[0], f[1], vert( 1, 0, 8, 0 ) } } );
    EXPECT_TRUE( preds.failed() );
    start();
    preds.addPoint( 3_v, { { f[0], vert( 10, 1, 1, 1 ) }, f } );
    EXPECT_TRUE( preds.failed() );

    // two points defined alike, which less cannot order
    start();
    const PreciseCutPoint x{ { vert( 10, 2, 1, -1 ), vert( 11, 2, 1, 1 ) }, f };
    preds.addPoint( 3_v, x );
    preds.addPoint( 4_v, x );
    EXPECT_FALSE( preds.failed() );
    (void)preds.less( 3_v, 4_v );
    EXPECT_TRUE( preds.failed() );
    start();
    preds.addPoint( 3_v, vertexPoint( f[1] ) );
    (void)preds.less( 1_v, 3_v );
    EXPECT_TRUE( preds.failed() );

    // three points on the cut by triangle t in plane x = 2: the crossings of sides 0 and 1 of f with t, and the crossing of an edge of t
    // with the plane of f at ( 2, 15.2, 0 ), outside f, which no crossing of t with f can be, so ccw cannot tell its side of t
    start();
    const Tri t = { vert( 20, 2, -4, -4 ), vert( 21, 2, -4, 4 ), vert( 22, 2, 20, -1 ) };
    preds.addPoint( 3_v, { { f[0], f[1] }, t } );
    preds.addPoint( 4_v, { { f[1], f[2] }, t } );
    preds.addPoint( 5_v, { { t[1], t[2] }, f } );
    EXPECT_FALSE( preds.failed() );
    (void)preds.ccw( 3_v, 4_v, 5_v );
    EXPECT_TRUE( preds.failed() );
}

// a triangle with its vertices on the x axis, which only the perturbation orients: the vertex with the smallest id moves most,
// in z first, so the perturbed normal is about ( 0, -2 eps, 0 ), and the predicates project the triangle on axes x, z
TEST( MRMesh, CutTrianglePredicatesFlat )
{
    auto vert = [] ( int id, int x ) { return PreciseVertCoords{ VertId( id ), Vector3i( x, 0, 0 ) }; };
    const Tri f = { vert( 0, 0 ), vert( 1, 2 ), vert( 2, 4 ) };
    CutTrianglePredicates preds;
    preds.reset( f );
    for ( int i = 0; i < 3; ++i )
        preds.addPoint( VertId( i ), vertexPoint( f[i] ) );
    EXPECT_TRUE( preds.less( 0_v, 1_v ) );
    EXPECT_TRUE( preds.less( 1_v, 2_v ) );
    EXPECT_TRUE( preds.ccw( 0_v, 1_v, 2_v ) );
    // f[1] is ahead of f[0] along the sweep, at the same z
    EXPECT_TRUE( preds.ccwFromBehind( 1_v, 0_v ) );
    EXPECT_FALSE( preds.ccwFromBehind( 0_v, 1_v ) );
    EXPECT_FALSE( preds.failed() );
}

} // namespace MR

