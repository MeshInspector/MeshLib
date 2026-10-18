#include <MRMesh/MRCutTrianglePredicates.h>
#include <MRMesh/MRPrecisePredicates3.h>
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

// random integer points in a cube, with ids in a random order, so that the perturbation orders the vertices variously;
// the ids are unique within one trial; the numbers come straight from mt19937, whose sequence the standard fixes,
// unlike the ones of std::uniform_int_distribution and std::shuffle, so every platform tests the same points
class RandomVerts
{
public:
    RandomVerts( unsigned seed, int halfSize ) : gen_( seed ), halfSize_( halfSize )
    {
        ids_.resize( 64 );
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

// the axes that the predicates project on: the two least aligned with the normal, ordered counterclockwise around it
std::pair<int, int> projectionAxes( const Tri& t )
{
    const Vector3i64 a( t[0].pt );
    const auto n = cross( Vector3i64( t[1].pt ) - a, Vector3i64( t[2].pt ) - a );
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
    return cross( toDouble( t[1] ) - toDouble( t[0] ), toDouble( t[2] ) - toDouble( t[0] ) );
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
// and bit 3+j if it is on the cut by others[j]
struct CutPoints
{
    std::vector<PreciseCutPoint> defs;
    std::vector<int> lines;

    // whether points i and j are on one line known from their definitions: a side, a cut by another triangle,
    // the plane through a vertex and an edge crossing this triangle, or the plane of a triangle of a vertex crossing a side
    bool onLine( int i, int j ) const
    {
        if ( lines[i] & lines[j] )
            return true;
        const auto& a = defs[i];
        const auto& b = defs[j];
        if ( a.isVertex() == b.isVertex() )
            return false;
        const auto& v = ( a.isVertex() ? a : b ).seg[0];
        const auto& x = a.isVertex() ? b : a;
        if ( x.seg[0].id == v.id || x.seg[1].id == v.id )
            return false;
        const bool sideCrossing = ( lines[a.isVertex() ? j : i] & 7 ) != 0;
        return !sideCrossing || x.tri[0].id == v.id || x.tri[1].id == v.id || x.tri[2].id == v.id;
    }

    // whether two of points p, q, r are on one line
    bool twoOnLine( int p, int q, int r ) const
    {
        return onLine( p, q ) || onLine( q, r ) || onLine( r, p );
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

// exact rational coordinates of a point with small integer coordinates: n / d, d > 0
struct SmallFraction
{
    Vector3i64 n;
    std::int64_t d = 1;
};

std::int64_t smallVolume( const Tri& t, const PreciseVertCoords& p )
{
    const Vector3i64 x( t[0].pt - p.pt ), y( t[1].pt - p.pt ), z( t[2].pt - p.pt );
    return dot( x, cross( y, z ) );
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
                        // three points on one side, which bulges outward a little: counterclockwise iff they go along it in this order,
                        // up to a cyclic shift
                        const int s = std::countr_zero( unsigned( side ) );
                        const Vector3d a = toDouble( f[s] ), u = toDouble( f[( s + 1 ) % 3] ) - a;
                        auto along = [&] ( int i, int j ) { return dot( pos[i] - a, u ) < dot( pos[j] - a, u ); };
                        expected = int( along( p, q ) ) + int( along( q, r ) ) + int( along( r, p ) ) == 2;
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
                    // the lines known from the definitions of the points give the same, if two of the points are on one
                    if ( pts.twoOnLine( p, q, r ) )
                    {
                        EXPECT_EQ( preds.ccwByLines( VertId( p ), VertId( q ), VertId( r ) ), expected );
                        ++numLineTriangles;
                    }
                    else
                    {
                        EXPECT_FALSE( preds.ccwByLines( VertId( p ), VertId( q ), VertId( r ) ).has_value() );
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
// positions are known, and consistent elsewhere; the order must be a strict total order that the expansion of the perturbation
// agrees with, and along a side the order must be the one of the boolean's cut
TEST( MRMesh, CutTrianglePredicatesGrid )
{
    RandomVerts grid2( 1, 2 ), grid1( 2, 1 );
    CutTrianglePredicates preds;
    int numTriangles = 0, numFlat = 0, numInPlane = 0, numPerturbed = 0, numPerturbedNoLine = 0, numSidePairs = 0, numDegenerate = 0, numExactNoLine = 0;
    for ( int trial = 0; trial < 800; ++trial )
    {
        // the coarser grid gives more segments in the planes of triangles and more flat triangles
        auto& rv = trial % 2 ? grid1 : grid2;
        rv.newTrial();
        const Tri f = rv.tri();
        preds.reset( f );
        // a triangle with a zero normal, which only the perturbation orients and projects
        const bool flat = normalOf( f ) == Vector3d();
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
        // -1, 0, 1 as the exact position of p comes before, at, after the one of q along the sweep, or nullopt if unknown
        auto compare = [&] ( int p, int q ) -> std::optional<int>
        {
            if ( !pos[p] || !pos[q] )
                return {};
            const int c = compareSmall( *pos[p], *pos[q], kx );
            return c != 0 ? c : compareSmall( *pos[p], *pos[q], ky );
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
                if ( const auto c = compare( p, q ); c && *c != 0 )
                {
                    EXPECT_EQ( bool( less[p * n + q] ), *c < 0 );
                    continue;
                }
                // coincident before the perturbation, or unknown: the expansion of the perturbation gives the direction from p to q,
                // which the sweep goes along iff its first coordinate is positive, or zero and the second one positive;
                // seen from far behind p against the sweep, q makes a counterclockwise turn iff it is below p, or level and ahead
                ++numPerturbed;
                numPerturbedNoLine += !pts.onLine( p, q );
                const auto dir = preds.directionByPerturbation( VertId( p ), VertId( q ) );
                ASSERT_TRUE( dir.has_value() );
                EXPECT_EQ( bool( less[p * n + q] ), dir->x != 0 ? dir->x > 0 : dir->y > 0 );
                EXPECT_EQ( preds.ccwFromBehind( VertId( q ), VertId( p ) ), dir->y != 0 ? dir->y < 0 : dir->x > 0 );
                EXPECT_EQ( preds.ccwFromBehind( VertId( p ), VertId( q ) ), dir->y != 0 ? dir->y > 0 : dir->x < 0 );
            }
        }
        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
                for ( int r = 0; r < n; ++r )
                    if ( less[p * n + q] && less[q * n + r] )
                    {
                        EXPECT_TRUE( less[p * n + r] );
                    }

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
                    ++numSidePairs;
                }
            }
        }

        // a triangle with two vertices on one line: decided by that line, exact unless degenerate before the perturbation or unknown,
        // the same in any rotation of its vertices, and the opposite when flipped; ccw agrees
        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
            {
                if ( p == q || !pts.onLine( p, q ) )
                    continue;
                for ( int r = 0; r < n; ++r )
                {
                    if ( r == p || r == q )
                        continue;
                    const auto c = preds.ccwByLines( VertId( p ), VertId( q ), VertId( r ) );
                    ASSERT_TRUE( c.has_value() );
                    EXPECT_EQ( preds.ccwByLines( VertId( q ), VertId( r ), VertId( p ) ), *c );
                    EXPECT_EQ( preds.ccwByLines( VertId( r ), VertId( p ), VertId( q ) ), *c );
                    EXPECT_EQ( preds.ccwByLines( VertId( q ), VertId( p ), VertId( r ) ), !*c );
                    EXPECT_EQ( preds.ccw( VertId( p ), VertId( q ), VertId( r ) ), *c );
                    if ( const int o = pos[p] && pos[q] && pos[r] ? orientSmall( *pos[p], *pos[q], *pos[r], kx, ky ) : 0 )
                    {
                        EXPECT_EQ( *c, o > 0 );
                    }
                    else
                        ++numDegenerate;
                }
            }

        // any triangle with a known exact orientation, also without two of its vertices on a known line
        for ( int p = 0; p < n; ++p )
            for ( int q = p + 1; q < n; ++q )
                for ( int r = q + 1; r < n; ++r )
                    if ( const int o = pos[p] && pos[q] && pos[r] ? orientSmall( *pos[p], *pos[q], *pos[r], kx, ky ) : 0 )
                    {
                        EXPECT_EQ( preds.ccw( VertId( p ), VertId( q ), VertId( r ) ), o > 0 );
                        numExactNoLine += !pts.twoOnLine( p, q, r );
                    }

        // seen from behind: exact where the points differ before the perturbation (the other pairs are checked above)
        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
            {
                if ( p == q )
                    continue;
                if ( const auto c = compare( q, p ); c && *c != 0 )
                {
                    const int cy = compareSmall( *pos[q], *pos[p], ky );
                    EXPECT_EQ( preds.ccwFromBehind( VertId( q ), VertId( p ) ), cy != 0 ? cy < 0 : *c > 0 );
                }
            }
        EXPECT_FALSE( preds.failed() );
    }
    EXPECT_GT( numTriangles, 600 );
    EXPECT_GT( numFlat, 50 );
    EXPECT_GT( numInPlane, 80 );
    EXPECT_GT( numPerturbed, 1200 );
    EXPECT_GT( numPerturbedNoLine, 250 );
    EXPECT_GT( numSidePairs, 280 );
    EXPECT_GT( numDegenerate, 20000 );
    EXPECT_GT( numExactNoLine, 400 );
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

    // 0, 1, 3 are on one line before the perturbation, and no two of them on a known one
    EXPECT_FALSE( preds.ccwByLines( 0_v, 1_v, 3_v ).has_value() );
    (void)preds.ccw( 0_v, 1_v, 3_v );
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

