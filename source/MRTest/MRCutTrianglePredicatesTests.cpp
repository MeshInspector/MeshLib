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

PreciseVertCoords vert( int id, int x, int y, int z )
{
    return { VertId( id ), Vector3i( x, y, z ) };
}

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

// the points of triangle f cut by other triangles: its vertices, the crossings of its sides with the others, and the crossings
// of their edges with it; bit s of lines[i] is set if point i is on side s of f, from vertex s to vertex s+1,
// and bit 3+j if it is on the cut by others[j], as the construction of the points tells, independently of the predicates
struct CutPoints
{
    std::vector<PreciseCutPoint> defs;
    std::vector<int> lines;

    void add( const PreciseCutPoint& p, int l )
    {
        defs.push_back( p );
        lines.push_back( l );
    }

    // whether two of points p, q, r are on a common side or cut, which ccwByLines must know
    bool twoOnSideOrCut( int p, int q, int r ) const
    {
        return ( lines[p] & lines[q] ) || ( lines[q] & lines[r] ) || ( lines[r] & lines[p] );
    }
};

// bit 3+j of CutPoints::lines, for the cut by others[j]; the cuts after the first 28 are not tracked,
// as they do not fit in int, and only the tests that ignore lines take more
int cutBit( int j )
{
    return j < 28 ? 8 << j : 0;
}

// the points of f cut by others, which may share vertices and edges with f and with one another, as in a mesh: a segment and a triangle
// with a common vertex are never tested for a crossing, as doTriangleSegmentIntersect cannot take them, so a triangle sharing vertex v
// with f cuts it from v to one crossing; an edge of several triangles crosses f at one point, on the cuts by all of them
CutPoints cutPoints( const Tri& f, const std::vector<Tri>& others )
{
    CutPoints res;
    for ( int s = 0; s < 3; ++s )
    {
        int lines = ( 1 << s ) | ( 1 << ( ( s + 2 ) % 3 ) );
        for ( int j = 0; j < int( others.size() ); ++j )
            if ( hasVert( others[j], f[s].id ) )
                lines |= cutBit( j );
        res.add( vertexPoint( f[s] ), lines );
    }
    for ( int j = 0; j < int( others.size() ); ++j )
    {
        const Tri& t = others[j];
        for ( int s = 0; s < 3; ++s )
        {
            const auto& a = f[s];
            const auto& b = f[( s + 1 ) % 3];
            if ( !hasVert( t, a.id ) && !hasVert( t, b.id ) && doTriangleSegmentIntersect( { t[0], t[1], t[2], a, b } ) )
                res.add( { { a, b }, t }, ( 1 << s ) | cutBit( j ) );
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
                res.lines[it - res.defs.begin()] |= cutBit( j );
            else
                res.add( { { a, b }, f }, cutBit( j ) );
        }
    }
    return res;
}

// the exact rational coordinates of a point, n / d with d > 0, in integers wider than the predicates take: int64 for the small
// coordinates of the grids, where it is faster, and FastInt256 for coordinates up to 2^30
template <typename T>
struct Fraction
{
    T n[3] = {};
    T d = {};
};

// nullopt if the position is unknown before the perturbation: a segment in the plane of its triangle
template <typename T>
std::optional<Fraction<T>> fractionOf( const PreciseCutPoint& p )
{
    Fraction<T> res;
    if ( p.isVertex() )
    {
        for ( int k = 0; k < 3; ++k )
            res.n[k] = T( p.seg[0].pt[k] );
        res.d = T( 1 );
        return res;
    }
    const auto oa = detail::orient3dVolume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[0].pt );
    const auto ob = detail::orient3dVolume( p.tri[0].pt, p.tri[1].pt, p.tri[2].pt, p.seg[1].pt );
    if ( oa == ob )
        return {};
    for ( int k = 0; k < 3; ++k )
        res.n[k] = T( T( oa ) * T( p.seg[1].pt[k] ) - T( ob ) * T( p.seg[0].pt[k] ) );
    res.d = T( oa ) - T( ob );
    if ( res.d < T() )
    {
        for ( auto& x : res.n )
            x = -x;
        res.d = -res.d;
    }
    return res;
}

// -1, 0, 1 as coordinate k of a is less, equal, greater than that of b
template <typename T>
int compareFractions( const Fraction<T>& a, const Fraction<T>& b, int k )
{
    const auto l = a.n[k] * b.d, r = b.n[k] * a.d;
    return l < r ? -1 : ( r < l ? 1 : 0 );
}

// whether coordinates k of a and b differ, but by less than 1 / scale
template <typename T>
bool closeFractions( const Fraction<T>& a, const Fraction<T>& b, int k, int scale )
{
    const auto diff = a.n[k] * b.d - b.n[k] * a.d;
    const decltype( diff ) zero{};
    const auto scaled = ( diff < zero ? -diff : diff ) * scale;
    return diff != zero && scaled < decltype( scaled )( a.d * b.d );
}

// the sign of the area of triangle abc projected on axes kx, ky: the determinant of the rows ( n[kx], n[ky], d ), whose d are positive
template <typename T>
int orientFractions( const Fraction<T>& a, const Fraction<T>& b, const Fraction<T>& c, int kx, int ky )
{
    const auto det = a.n[kx] * ( b.n[ky] * c.d - b.d * c.n[ky] )
                   - a.n[ky] * ( b.n[kx] * c.d - b.d * c.n[kx] )
                   + a.d * ( b.n[kx] * c.n[ky] - b.n[ky] * c.n[kx] );
    const decltype( det ) zero{};
    return det < zero ? -1 : ( zero < det ? 1 : 0 );
}

// the cases that checkOrder met
struct OrderStats
{
    int perturbed = 0; // pairs that coincide before the perturbation, or with unknown exact positions
    int perturbedNoLine = 0; // such pairs on no common side or cut
    int perturbedOnCut = 0; // such pairs on a common cut, but on no common side
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
            // the signs of the projected coordinates of q - p: exact where the exact positions are known and differ, otherwise
            // from the expansion of the perturbation; the sweep goes along q - p iff its first coordinate is positive, or zero
            // and the second one positive, and seen from far behind p against the sweep, q makes a counterclockwise turn
            // iff it is below p, or level and ahead
            const auto cx = compare( p, q, kx ), cy = compare( p, q, ky );
            Vector2i d;
            if ( cx && ( *cx != 0 || *cy != 0 ) )
                d = Vector2i( -*cx, -*cy );
            else
            {
                const int common = pts.lines[p] & pts.lines[q];
                ++stats.perturbed;
                stats.perturbedNoLine += common == 0;
                stats.perturbedOnCut += common > 7 && ( common & 7 ) == 0;
                const auto dir = preds.directionByPerturbation( VertId( p ), VertId( q ) );
                ASSERT_TRUE( dir.has_value() );
                d = *dir;
            }
            EXPECT_EQ( bool( less[p * n + q] ), d.x != 0 ? d.x > 0 : d.y > 0 );
            EXPECT_EQ( preds.ccwFromBehind( VertId( q ), VertId( p ) ), d.y != 0 ? d.y < 0 : d.x > 0 );
            EXPECT_EQ( preds.ccwFromBehind( VertId( p ), VertId( q ) ), d.y != 0 ? d.y > 0 : d.x < 0 );
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
}

// the cases that checkCcw met
struct CcwStats
{
    int exact = 0; // triangles with a known exact orientation
    int exactNoLine = 0; // such triangles without two points on a known line
    int perturbed = 0; // triangles on one line before the perturbation or with unknown positions, without two points on a known line
    int sides = 0; // triangles of three points on one side
    int lines = 0; // such triangles off the sides that ccwByLines decides
    int byPerturbation = 0; // triangles checked against the expansion of the perturbation
};

// checks ccw on all triangles of the points: decided without a failure, exact where orient( p, q, r ) gives the sign of the exact
// doubled area (0 if it is zero or unknown), with three points of one side bulging inward, and satisfying the axioms of Knuth's
// CC systems as a perturbation of the points in the plane does: cyclic symmetry, antisymmetry, interiority and transitivity;
// ccwByLines must decide every triangle with two points on a common side or cut, the same as ccw in any order of the points;
// unless exactStride is 0, the expansion of the perturbation alone (ccwByPerturbation), which is slow, gives the same on every
// triangle off the sides that is degenerate or unknown, so also where ccwByLines decides, and on every exactStride-th triangle
// with a known exact orientation (the points of one side stay on one line after the perturbation, which the expansion finds
// only after all its stages, too slowly for all such triangles here, see CutTrianglePredicatesCollinear)
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

    int numAsymmetric = 0, numWrongSides = 0, numWrongLines = 0, numWrongPerturbation = 0, numNotInterior = 0, numIntransitive = 0, numExact = 0;
    for ( int p = 0; p < n; ++p )
        for ( int q = p + 1; q < n; ++q )
            for ( int r = q + 1; r < n; ++r )
            {
                // the rotations of the points keep the orientation, and the swaps flip it
                const bool c = ccw( p, q, r );
                numAsymmetric += ccw( q, r, p ) != c || ccw( r, p, q ) != c || ccw( q, p, r ) == c || ccw( p, r, q ) == c || ccw( r, q, p ) == c;
                const int o = orient( p, q, r );
                if ( o != 0 )
                {
                    EXPECT_EQ( c, o > 0 );
                    ++stats.exact;
                    stats.exactNoLine += !pts.twoOnSideOrCut( p, q, r );
                }
                const int side = pts.lines[p] & pts.lines[q] & pts.lines[r] & 7;
                if ( side )
                {
                    // counterclockwise iff the points go backward along the side, from vertex s+1 to vertex s, up to a cyclic shift
                    const int s = std::countr_zero( unsigned( side ) );
                    const bool alongSweep = preds.less( VertId( s ), VertId( ( s + 1 ) % 3 ) );
                    auto along = [&] ( int i, int j ) { return preds.less( VertId( i ), VertId( j ) ) == alongSweep; };
                    numWrongSides += c != ( int( along( p, q ) ) + int( along( q, r ) ) + int( along( r, p ) ) == 1 );
                    ++stats.sides;
                }
                const auto l = preds.ccwByLines( VertId( p ), VertId( q ), VertId( r ) );
                if ( l )
                    numWrongLines += *l != c || preds.ccwByLines( VertId( q ), VertId( r ), VertId( p ) ) != *l
                        || preds.ccwByLines( VertId( r ), VertId( p ), VertId( q ) ) != *l || preds.ccwByLines( VertId( q ), VertId( p ), VertId( r ) ) != !*l;
                else
                    numWrongLines += pts.twoOnSideOrCut( p, q, r );
                if ( o == 0 && !side )
                    ++( l ? stats.lines : stats.perturbed );
                if ( exactStride == 0 || side || ( o != 0 && ++numExact % exactStride != 0 ) )
                    continue;
                numWrongPerturbation += preds.ccwByPerturbation( VertId( p ), VertId( q ), VertId( r ) ) != c;
                ++stats.byPerturbation;
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

// sets the points of f cut by other triangles (see cutPoints) and checks them by checkOrder and checkCcw (with exactStride)
// against their exact positions in integers T, which are unknown on a flat f and for a segment in the plane of its triangle,
// unless limits gives them; returns the number of such segments
template <typename T>
int checkSet( CutTrianglePredicates& preds, const Tri& f, const CutPoints& pts, OrderStats& orderStats, CcwStats& ccwStats,
    int exactStride = 10, const std::vector<std::optional<Fraction<T>>>& limits = {} )
{
    preds.reset( f );
    const bool flat = exactNormal( f ) == Vector3i64();
    const auto [kx, ky] = projectionAxes( f );
    const int n = int( pts.defs.size() );
    std::vector<std::optional<Fraction<T>>> pos;
    int numInPlane = 0;
    for ( int i = 0; i < n; ++i )
    {
        preds.addPoint( VertId( i ), pts.defs[i] );
        auto x = fractionOf<T>( pts.defs[i] );
        numInPlane += !x;
        if ( !x && i < int( limits.size() ) )
            x = limits[i];
        pos.push_back( flat ? std::nullopt : x );
    }
    EXPECT_FALSE( preds.failed() );
    // the projection keeps the orientation of f around its normal, also around the perturbed one of a flat f, which ccw of its
    // vertices does not check, since side 0 decides it
    EXPECT_EQ( preds.ccwByPerturbation( 0_v, 1_v, 2_v ), true );
    if ( !flat )
    {
        EXPECT_GT( orientFractions( *pos[0], *pos[1], *pos[2], kx, ky ), 0 );
    }
    checkOrder( preds, f, pts, kx, ky, flat, [&] ( int p, int q, int k ) -> std::optional<int>
    {
        if ( !pos[p] || !pos[q] )
            return {};
        return compareFractions( *pos[p], *pos[q], k );
    }, orderStats );
    checkCcw( preds, pts, [&] ( int p, int q, int r )
    {
        return pos[p] && pos[q] && pos[r] ? orientFractions( *pos[p], *pos[q], *pos[r], kx, ky ) : 0;
    }, exactStride, ccwStats );
    return numInPlane;
}

} // anonymous namespace

// Points on a small grid, where many of them coincide and many triangles are degenerate before the perturbation, including
// segments in the planes of their triangles and triangles with zero normals, which only the perturbation orients and projects,
// most of them with two vertices at one point, whose side only the perturbation directs: the predicates must be exact
// where the exact positions are known, and consistent elsewhere (see checkOrder and checkCcw);
// in every 14th set, f is a large triangle in plane z = 0 and the other triangles have their vertices in planes z = -1 and z = 1,
// so that their edges cross f at the halves of grid points, where many points coincide and many triples are on one line
// on no known one; the slow comparisons with the expansion of the perturbation (see checkCcw) take these sets and every 28th other one
TEST( MRMesh, CutTrianglePredicatesGrid )
{
    RandomVerts grid2( 1, 2 ), grid1( 2, 1 );
    CutTrianglePredicates preds;
    OrderStats orderStats;
    CcwStats ccwStats;
    int numTriangles = 0, numZeroSides = 0, numInPlane = 0;
    for ( int trial = 0; trial < 800 / cTrialDivisor; ++trial )
    {
        const bool layers = trial % 14 == 0;
        // the coarser grid gives more segments in the planes of triangles and more flat triangles
        auto& rv = trial % 2 ? grid1 : grid2;
        rv.newTrial();
        Tri f = rv.tri();
        std::vector<Tri> others;
        for ( int i = 0; i < 4; ++i )
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
        numTriangles += exactNormal( f ) != Vector3i64();
        // a flat f with a side between two vertices at one point
        numZeroSides += f[0].pt == f[1].pt || f[1].pt == f[2].pt || f[2].pt == f[0].pt;
        numInPlane += checkSet<std::int64_t>( preds, f, cutPoints( f, others ), orderStats, ccwStats, layers || trial % 28 == 7 ? 10 : 0 );
    }
    EXPECT_GE( numTriangles, atLeast( 600, 120 ) );
    EXPECT_GE( numZeroSides, atLeast( 50, 10 ) );
    EXPECT_GE( numInPlane, atLeast( 90, 18 ) );
    EXPECT_GE( orderStats.perturbed, atLeast( 1300, 250 ) );
    EXPECT_GE( orderStats.perturbedNoLine, atLeast( 530, 100 ) );
    EXPECT_GE( orderStats.perturbedOnCut, atLeast( 240, 55 ) );
    EXPECT_GE( orderStats.sidePairs, atLeast( 300, 40 ) );
    EXPECT_GE( ccwStats.exact, atLeast( 13000, 2700 ) );
    EXPECT_GE( ccwStats.exactNoLine, atLeast( 4200, 950 ) );
    EXPECT_GE( ccwStats.perturbed, atLeast( 250, 50 ) );
    EXPECT_GE( ccwStats.sides, atLeast( 1000, 170 ) );
    EXPECT_GE( ccwStats.lines, atLeast( 4000, 800 ) );
    EXPECT_GE( ccwStats.byPerturbation, atLeast( 900, 200 ) );
}

// Triangles of a mesh: all vertices come from a small pool on a grid, so the triangles cutting f share vertices and edges with
// one another and with f, as in selfBoolean, which cuts f along folds from its vertices (see cutPoints); two crossings
// of a side with triangles sharing vertices take the shared-vertex path of segmentIntersectionOrder, and two crossings of edges
// with a common vertex are on the plane through the edges
TEST( MRMesh, CutTrianglePredicatesSharedVertices )
{
    RandomVerts rv( 5, 2 );
    CutTrianglePredicates preds;
    OrderStats orderStats;
    CcwStats ccwStats;
    int numFolds = 0, numSharedEdges = 0, numEdgePlanes = 0;
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
        const auto pts = cutPoints( f, std::vector<Tri>( tris.begin() + 1, tris.end() ) );
        const int n = int( pts.defs.size() );
        // a point on the cuts by two triangles: a crossing of their common edge, or a vertex of f shared by them
        for ( int i = 3; i < n; ++i )
            numSharedEdges += std::popcount( unsigned( pts.lines[i] >> 3 ) ) > 1;
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
        checkSet<std::int64_t>( preds, f, pts, orderStats, ccwStats );
    }
    EXPECT_GE( numFolds, atLeast( 200, 36 ) );
    EXPECT_GE( numSharedEdges, atLeast( 40, 11 ) );
    EXPECT_GE( numEdgePlanes, atLeast( 40, 12 ) );
    EXPECT_GE( orderStats.perturbed, atLeast( 85, 8 ) );
    EXPECT_GE( orderStats.perturbedOnCut, atLeast( 20, 2 ) );
    EXPECT_GE( orderStats.sharedSidePairs, atLeast( 110, 25 ) );
    EXPECT_GE( ccwStats.exact, atLeast( 1850, 450 ) );
    EXPECT_GE( ccwStats.lines, atLeast( 220, 36 ) );
    EXPECT_GE( ccwStats.byPerturbation, atLeast( 330, 65 ) );
}

// Coordinates near 2^30, the limit for which the error bounds of the rounded coordinates and the widths of the exact integers
// are derived: a coarse grid scaled by 2^30 - 2^24, with every fourth vertex moved by a few units, so that many points coincide
// or nearly coincide and many triangles are degenerate or nearly so; and random triangles in general position, with coordinates
// in the whole range and up to 2^20, where the rounded coordinates decide most comparisons, while the exact positions decide every pair
// and every triangle off the sides; the predicates must agree with the exact positions, found here in wider integers than the predicates take
TEST( MRMesh, CutTrianglePredicatesLargeCoords )
{
    constexpr int cScale = ( 1 << 30 ) - ( 1 << 24 );
    RandomVerts grid( 6, 1 ), wide( 7, ( 1 << 30 ) - 1 ), narrow( 0, 1 << 20 );
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
    OrderStats orderStats[2]; // on the grid, and in general position
    CcwStats ccwStats[2];
    int numPoints[2] = {}, numNear = 0;
    for ( int trial = 0; trial < 250 / cTrialDivisor; ++trial )
    {
        // of every ten trials, three on the grid, three in the whole range, and four up to 2^20 with more triangles
        const bool onGrid = trial % 10 < 3, isNarrow = !onGrid && trial % 2 == 1;
        auto& rv = onGrid ? grid : ( isNarrow ? narrow : wide );
        rv.newTrial();
        const Tri f = onGrid ? scaled( rv.tri() ) : rv.tri();
        std::vector<Tri> others;
        for ( int i = 0; i < ( isNarrow ? 6 : 4 ); ++i )
            others.push_back( onGrid ? scaled( rv.tri() ) : rv.tri() );
        const auto pts = cutPoints( f, others );
        const int n = int( pts.defs.size() );
        numPoints[!onGrid] += n;
        // pairs of different points closer than a unit, of which the coordinates in doubles have rounding errors of about 2^-23
        const int kx = projectionAxes( f ).first;
        std::vector<std::optional<Fraction<FastInt256>>> pos;
        for ( const auto& d : pts.defs )
            pos.push_back( fractionOf<FastInt256>( d ) );
        for ( int p = 0; p < n; ++p )
            for ( int q = p + 1; q < n; ++q )
                numNear += pos[p] && pos[q] && closeFractions( *pos[p], *pos[q], kx, 1 );
        checkSet<FastInt256>( preds, f, pts, orderStats[!onGrid], ccwStats[!onGrid] );
    }
    EXPECT_GE( numNear, atLeast( 22, 3 ) );
    EXPECT_GE( numPoints[0], atLeast( 380, 85 ) );
    EXPECT_GE( orderStats[0].perturbed, atLeast( 125, 30 ) );
    EXPECT_GE( orderStats[0].perturbedOnCut, atLeast( 20, 3 ) );
    EXPECT_GE( orderStats[0].sidePairs, atLeast( 48, 15 ) );
    EXPECT_GE( ccwStats[0].exact, atLeast( 1300, 480 ) );
    EXPECT_GE( ccwStats[0].perturbed, atLeast( 9, 4 ) );
    EXPECT_GE( ccwStats[0].lines, atLeast( 300, 67 ) );
    EXPECT_EQ( orderStats[1].perturbed, 0 );
    EXPECT_EQ( ccwStats[1].perturbed + ccwStats[1].lines, 0 );
    EXPECT_GE( numPoints[1], atLeast( 960, 200 ) );
    EXPECT_GE( orderStats[1].sidePairs, atLeast( 135, 40 ) );
    EXPECT_GE( ccwStats[1].exact, atLeast( 5800, 1550 ) );
    EXPECT_GE( ccwStats[1].exactNoLine, atLeast( 1700, 520 ) );
    EXPECT_GE( ccwStats[1].sides, atLeast( 370, 95 ) );
    EXPECT_GE( ccwStats[1].byPerturbation, atLeast( 500, 140 ) );
}

// Points closer than the tolerance of their rounded coordinates (2^-17), but not level, near a corner of a triangle as large as the integer
// conversion allows: crossings of edges with f at ( X + 1 / m, Y + w / m ), which differ in x by a little more than the tolerance (m = 256, 257),
// a little less (362, 363) and much less (4096, 4097), only by less than it in y (the same m) or in x (the same w = 0), and a segment
// in the plane of f with an end b at ( X, Y ), where its crossing tends to, since its other end has the smallest id and moves most,
// so that only the leading terms of the perturbed volumes give the point; the predicates must agree with the exact positions, which decide
// the triangles of crossings too thin for the rounded coordinates, with no two points on a known line
TEST( MRMesh, CutTrianglePredicatesNearTolerance )
{
    constexpr int B = ( 1 << 30 ) - 1, X = B - ( 1 << 10 ), Y = ( 1 << 9 ) - B;
    const Tri f = { vert( 1, -B, -B, 0 ), vert( 2, B, -B, 0 ), vert( 3, -B, B, 0 ) };
    auto pts = cutPoints( f, {} );
    const auto b = vert( 4, X, Y, 0 );
    pts.add( { { vert( 0, X - 5, Y + 3, 0 ), b }, f }, 0 );
    constexpr std::pair<int, int> crossings[] = { { 256, 4 }, { 257, 1 }, { 362, 2 }, { 363, 3 }, { 512, 0 }, { 513, 0 },
        { 4096, 5 }, { 4097, 5 }, { 1 << 18, 1 }, { 1 << 18, 2 } };
    int id = 10;
    for ( const auto& [m, w] : crossings )
    {
        pts.add( { { vert( id, X, Y, -1 ), vert( id + 1, X + 1, Y + w, m - 1 ) }, f }, 0 );
        id += 2;
    }
    const int n = int( pts.defs.size() );
    int numBelowTol = 0; // coordinates of two crossings that differ by less than the tolerance, but not exactly
    for ( int p = 4; p < n; ++p )
        for ( int q = p + 1; q < n; ++q )
            for ( int k = 0; k < 2; ++k )
                numBelowTol += closeFractions( *fractionOf<FastInt256>( pts.defs[p] ), *fractionOf<FastInt256>( pts.defs[q] ), k, 1 << 17 );
    EXPECT_GE( numBelowTol, 6 );
    CutTrianglePredicates preds;
    OrderStats orderStats;
    CcwStats ccwStats;
    std::vector<std::optional<Fraction<FastInt256>>> limits( 4 );
    limits[3] = fractionOf<FastInt256>( vertexPoint( b ) );
    EXPECT_EQ( checkSet( preds, f, pts, orderStats, ccwStats, 10, limits ), 1 );
    EXPECT_FALSE( preds.failed() );
}

// the const members called concurrently, as the parallel sort of the sweep calls less, so the directions found by expanding the perturbation
// and the orientations found by the slower rules are cached from several threads: on a coarse grid, where many points coincide,
// the parallel sort gives the order of a sequential sort, and ccw and ccwFromBehind give the answers of sequential calls
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

    // ccw on all triangles of m points spread over the set, in all orders, and ccwFromBehind on their pairs, from empty caches
    constexpr int m = 8;
    auto answer = [&] ( int i )
    {
        const VertId a( i / m / m * n / m ), b( i / m % m * n / m ), c( i % m * n / m );
        return a == b || b == c || c == a ? 0 : 1 + preds.ccw( a, b, c ) + 2 * preds.ccwFromBehind( b, c );
    };
    std::vector<int> par( m * m * m ), seq( m * m * m );
    setup();
    tbb::parallel_for( 0, m * m * m, [&] ( int i ) { par[i] = answer( i ); } );
    setup();
    for ( int i = 0; i < m * m * m; ++i )
        seq[i] = answer( i );
    EXPECT_EQ( par, seq );
    EXPECT_FALSE( preds.failed() );
}

// setting a point again: the answers follow its new definition, also for the pairs and the triangles of coincident points,
// whose orders and orientations the slower rules found and cached for the old definition
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
                const auto a = fractionOf<std::int64_t>( pts.defs[i] ), b = fractionOf<std::int64_t>( pts.defs[j] );
                if ( a && b && compareFractions( *a, *b, 0 ) == 0 && compareFractions( *a, *b, 1 ) == 0 && compareFractions( *a, *b, 2 ) == 0
                    && !( pts.lines[i] & pts.lines[j] ) )
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
        // the triangles of p, q and a vertex of f are degenerate before the perturbation, and they flip with the swap below
        for ( int i = 0; i < 3; ++i )
            (void)preds.ccw( VertId( p ), VertId( q ), VertId( i ) );
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
        for ( int i = 0; i < 3; ++i )
        {
            EXPECT_EQ( preds.ccw( VertId( p ), VertId( q ), VertId( i ) ), fresh.ccw( VertId( p ), VertId( q ), VertId( i ) ) );
            EXPECT_EQ( preds.ccw( VertId( q ), VertId( i ), VertId( p ) ), fresh.ccw( VertId( q ), VertId( i ), VertId( p ) ) );
        }
        EXPECT_FALSE( preds.failed() );
        EXPECT_FALSE( fresh.failed() );
        ++numSwapped;
    }
    EXPECT_EQ( numSwapped, 30 );
}

// selfBoolean cuts f by a triangle sharing a vertex with it along a line from that vertex (a fold, see SharedVertices), so the sweep asks
// the orientation of that vertex, a crossing on the fold and a point of another cut; here triangle t of vertex 0 in the plane of f cuts f
// from vertex 0 to the crossing of side 1 with t, which tends to vertex 2, since vertex 1 has the smallest id and moves most; with
// the crossing of an edge with f a little off side 2, at ( 1 / ( 2^20 + 1 ), 4 ), they make a triangle too thin for the rounded coordinates,
// which the fold decides before the exact positions, and the direction of the fold only the perturbation gives, since t is parallel
// to f (sosCrossDirection)
TEST( MRMesh, CutTrianglePredicatesFoldLine )
{
    const Tri f = { vert( 1, 0, 0, 0 ), vert( 0, 8, 0, 0 ), vert( 2, 0, 8, 0 ) };
    const Tri t = { f[0], vert( 3, -4, 12, 0 ), vert( 4, 4, 12, 0 ) };
    ASSERT_TRUE( doTriangleSegmentIntersect( { t[0], t[1], t[2], f[1], f[2] } ) );
    CutTrianglePredicates preds;
    preds.reset( f );
    for ( int i = 0; i < 3; ++i )
        preds.addPoint( VertId( i ), vertexPoint( f[i] ) );
    preds.addPoint( 3_v, { { f[1], f[2] }, t } );
    preds.addPoint( 4_v, { { vert( 5, 1, 4, -( 1 << 20 ) ), vert( 6, 0, 4, 1 ) }, f } );
    // the exact positions decide ccw and the expansion of the perturbation, and the line through the vertex and the crossing ccwByLines
    auto check = [&] ( VertId a, VertId b, VertId c, bool expected )
    {
        EXPECT_EQ( preds.ccw( a, b, c ), expected );
        EXPECT_EQ( preds.ccwByLines( a, b, c ), expected );
        EXPECT_EQ( preds.ccwByPerturbation( a, b, c ), expected );
    };
    check( 0_v, 3_v, 4_v, false );
    check( 0_v, 4_v, 3_v, true );
    EXPECT_FALSE( preds.failed() );
}

// three crossings at ( 0, 0 ), ( 1, 1 ), ( 2, 2 ) on no known line, where the expansion of the perturbation decides:
// - with the smallest id at the lower end of the first segment, its perturbation decides: vertex 0 moves to ( eps^9, eps^3, -1 + eps ),
//   which moves the first crossing to ( eps^9, eps^3 ) / ( 2 - eps ), above the line y = x;
// - with the smallest ids at the vertices of f, perturbing them tilts the plane of f, which moves the crossings of the vertical segments
//   along the segments only, so the projected points stay on one line at the first three stages of the expansion, and the fourth one,
//   which perturbs the lower end of the first segment (as the smallest id did above), decides by the shortcut for a stage after a zero one,
//   which takes only the products with the vertex perturbed at that stage; the same holds for two crossings at ( 3, 2 ), which only
//   that vertex moves apart: the first crossing to ( eps^9, eps^3 ) / 2 from the second one, of which the sweep takes only the larger term
TEST( MRMesh, CutTrianglePredicatesCollinear )
{
    CutTrianglePredicates preds;
    // fId and segId are the first ids of the vertices of f and of the segments
    auto setup = [&] ( int fId, int segId )
    {
        const Tri f = { vert( fId, -1, -1, 0 ), vert( fId + 1, 8, -1, 0 ), vert( fId + 2, -1, 8, 0 ) };
        preds.reset( f );
        for ( int i = 0; i < 3; ++i )
            preds.addPoint( VertId( i ), { { vert( segId + 2 * i, i, i, -1 ), vert( segId + 2 * i + 1, i, i, 1 ) }, f } );
        EXPECT_FALSE( preds.ccwByLines( 0_v, 1_v, 2_v ).has_value() );
        EXPECT_TRUE( preds.ccw( 0_v, 1_v, 2_v ) );
        EXPECT_TRUE( preds.ccw( 1_v, 2_v, 0_v ) );
        EXPECT_TRUE( preds.ccw( 2_v, 0_v, 1_v ) );
        EXPECT_FALSE( preds.ccw( 0_v, 2_v, 1_v ) );
        EXPECT_FALSE( preds.ccw( 2_v, 1_v, 0_v ) );
        EXPECT_EQ( preds.ccwByPerturbation( 0_v, 1_v, 2_v ), true );
        EXPECT_EQ( preds.ccwByPerturbation( 1_v, 0_v, 2_v ), false );
        return f;
    };
    const Tri f = setup( 10, 0 );
    // two vertices of f and a crossing of the side between them with a triangle, which stay on one line after the perturbation,
    // where they bulge inward: the vertices of side 0 come in the order of the side, so the triangle is clockwise
    preds.addPoint( 3_v, vertexPoint( f[0] ) );
    preds.addPoint( 4_v, vertexPoint( f[1] ) );
    preds.addPoint( 5_v, { { f[0], f[1] }, { vert( 20, 3, -5, -5 ), vert( 21, 3, -5, 5 ), vert( 22, 3, 5, 0 ) } } );
    EXPECT_FALSE( preds.ccwByPerturbation( 3_v, 5_v, 4_v ).has_value() );
    EXPECT_FALSE( preds.ccw( 3_v, 5_v, 4_v ) );
    EXPECT_TRUE( preds.ccw( 4_v, 5_v, 3_v ) );
    EXPECT_FALSE( preds.failed() );

    const Tri g = setup( 0, 10 );
    preds.addPoint( 3_v, { { vert( 30, 3, 2, -1 ), vert( 31, 3, 2, 1 ) }, g } );
    preds.addPoint( 4_v, { { vert( 32, 3, 2, -2 ), vert( 33, 3, 2, 2 ) }, g } );
    EXPECT_EQ( preds.directionByPerturbation( 3_v, 4_v ), Vector2i( 0, -1 ) );
    EXPECT_EQ( preds.directionByPerturbation( 4_v, 3_v ), Vector2i( 0, 1 ) );
    EXPECT_TRUE( preds.less( 4_v, 3_v ) );
    EXPECT_FALSE( preds.less( 3_v, 4_v ) );
    EXPECT_FALSE( preds.failed() );
}

// the inputs that break the preconditions make reset or addPoint fail, and the caller then calls no predicate; points that the predicates
// cannot tell apart, which valid input never gives either, make the predicates fail, and they stay callable
TEST( MRMesh, CutTrianglePredicatesFailed )
{
    const Tri f = { vert( 0, 0, 0, 0 ), vert( 1, 8, 0, 0 ), vert( 2, 0, 8, 0 ) };
    CutTrianglePredicates preds;
    auto start = [&]
    {
        preds.reset( f );
        for ( int i = 0; i < 3; ++i )
            preds.addPoint( VertId( i ), vertexPoint( f[i] ) );
        EXPECT_FALSE( preds.failed() );
    };
    const PreciseCutPoint x{ { vert( 10, 2, 1, -1 ), vert( 11, 2, 1, 1 ) }, f };

    // repeated ids in the triangle
    preds.reset( { f[0], f[1], vert( 1, 0, 8, 0 ) } );
    EXPECT_TRUE( preds.failed() );
    const Tri t = { vert( 20, -4, -4, 1 ), vert( 21, 12, -4, 1 ), vert( 22, -4, 12, 1 ) };
    const PreciseCutPoint broken[] =
    {
        // a vertex point that is not a corner of the triangle
        vertexPoint( vert( 9, 1, 1, 0 ) ),
        // a segment that does not cross the plane of its triangle
        { { vert( 10, 1, 1, 1 ), vert( 11, 2, 2, 3 ) }, f },
        // a segment in the plane of its triangle that does not cross it after the perturbation either: the vertex of f with
        // the smallest id moves most, in z, and so it tilts the plane, which puts both ends of the segment on one side
        { { vert( 10, 1, 1, 0 ), vert( 11, 2, 1, 0 ) }, f },
        // repeated ids: in a segment, in the triangle of a crossing, and an end of a segment at a vertex of its triangle
        { { vert( 10, 1, 1, -1 ), vert( 10, 1, 1, 1 ) }, f },
        { { vert( 10, 1, 1, -1 ), vert( 11, 1, 1, 1 ) }, { f[0], f[1], vert( 1, 0, 8, 0 ) } },
        { { f[0], vert( 10, 1, 1, 1 ) }, f },
        // a segment off the sides of f with another triangle, which it crosses inside, also with a single end at a corner of f:
        // only a side of f can cross another triangle
        { { vert( 10, 1, 1, -1 ), vert( 11, 1, 1, 3 ) }, t },
        { { f[0], vert( 10, 1, 1, 2 ) }, t },
    };
    for ( const auto& p : broken )
    {
        start();
        preds.addPoint( 3_v, p );
        EXPECT_TRUE( preds.failed() );
        // the failure stays when the point is set again
        preds.addPoint( 3_v, x );
        EXPECT_TRUE( preds.failed() );
    }

    // two points defined alike, the second one with the segment backward, which less cannot order: it orders them by their ids,
    // and alike with the other points, also with a third point at their position, so it stays a strict total order
    auto alike = [&]
    {
        start();
        preds.addPoint( 3_v, x );
        preds.addPoint( 4_v, { { x.seg[1], x.seg[0] }, x.tri } );
        preds.addPoint( 5_v, { { vert( 12, 1, 0, -1 ), vert( 13, 3, 2, 1 ) }, f } );
        EXPECT_FALSE( preds.failed() );
    };
    alike();
    for ( int p = 0; p < 6; ++p )
        for ( int q = 0; q < 6; ++q )
        {
            if ( p != q )
            {
                EXPECT_NE( preds.less( VertId( p ), VertId( q ) ), preds.less( VertId( q ), VertId( p ) ) );
            }
            for ( int r = 0; r < 6; ++r )
            {
                EXPECT_FALSE( preds.less( VertId( p ), VertId( q ) ) && preds.less( VertId( q ), VertId( r ) ) && !preds.less( VertId( p ), VertId( r ) ) );
            }
        }
    EXPECT_TRUE( preds.failed() );
    // nor can ccw tell them apart: with a vertex of f, on the plane line through it and either of them, and with a third point defined
    // alike, on no known line with them, where the expansion of the perturbation finds the three on one line after all its stages
    // (with a third point of other vertices, about 50 times slower); nor can ccwFromBehind
    alike();
    (void)preds.ccw( 3_v, 4_v, 0_v );
    EXPECT_TRUE( preds.failed() );
    alike();
    preds.addPoint( 5_v, x );
    EXPECT_FALSE( preds.ccwByLines( 3_v, 4_v, 5_v ).has_value() );
    EXPECT_FALSE( preds.failed() );
    (void)preds.ccw( 3_v, 4_v, 5_v );
    EXPECT_TRUE( preds.failed() );
    alike();
    (void)preds.ccwFromBehind( 4_v, 3_v );
    EXPECT_TRUE( preds.failed() );
    // the same vertex twice
    start();
    preds.addPoint( 3_v, vertexPoint( f[1] ) );
    EXPECT_FALSE( preds.failed() );
    (void)preds.less( 1_v, 3_v );
    EXPECT_TRUE( preds.failed() );
}

} // namespace MR
