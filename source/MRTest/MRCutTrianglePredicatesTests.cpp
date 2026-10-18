#include <MRMesh/MRCutTrianglePredicates.h>
#include <MRMesh/MRPrecisePredicates3.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <functional>
#include <numeric>
#include <optional>
#include <random>

namespace MR
{

namespace
{

using Tri = std::array<PreciseVertCoords, 3>;

// random integer points in a cube, with ids in a random order, so that the perturbation orders the vertices variously;
// the ids are unique within one trial
class RandomVerts
{
public:
    RandomVerts( unsigned seed, int halfSize ) : gen_( seed ), coord_( -halfSize, halfSize )
    {
        ids_.resize( 64 );
        std::iota( ids_.begin(), ids_.end(), 0 );
    }
    void newTrial()
    {
        std::shuffle( ids_.begin(), ids_.end(), gen_ );
        nextId_ = 0;
    }
    PreciseVertCoords vert()
    {
        assert( nextId_ < ids_.size() );
        return { VertId( ids_[nextId_++] ), Vector3i( coord_( gen_ ), coord_( gen_ ), coord_( gen_ ) ) };
    }
    Tri tri()
    {
        return { vert(), vert(), vert() };
    }

private:
    std::mt19937 gen_;
    std::uniform_int_distribution<int> coord_;
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
// of their edges with it, if defined; bit s of lines[i] is set if point i is on side s of f, from vertex s to vertex s+1,
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

CutPoints cutPoints( const Tri& f, const std::vector<Tri>& others, const std::function<bool( const PreciseCutPoint& )>& defined )
{
    CutPoints res;
    auto add = [&] ( const PreciseCutPoint& p, int lines )
    {
        if ( !defined( p ) )
            return;
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
    int numOrders = 0, numBehind = 0, numTriangles = 0, numSideTriangles = 0, numUndecided = 0;
    for ( int trial = 0; trial < 100; ++trial )
    {
        rv.newTrial();
        const Tri f = rv.tri();
        ASSERT_TRUE( preds.reset( f ) );
        const auto [kx, ky] = projectionAxes( f );
        {
            // the projection keeps the orientation around the normal
            const Vector3d a = toDouble( f[0] ), b = toDouble( f[1] ), c = toDouble( f[2] );
            EXPECT_GT( ( b[kx] - a[kx] ) * ( c[ky] - a[ky] ) - ( b[ky] - a[ky] ) * ( c[kx] - a[kx] ), 0 );
        }
        std::vector<Tri> others;
        for ( int i = 0; i < 6; ++i )
            others.push_back( rv.tri() );
        const auto pts = cutPoints( f, others, [] ( const PreciseCutPoint& ) { return true; } );
        const int n = int( pts.defs.size() );
        std::vector<Vector3d> pos;
        for ( int i = 0; i < n; ++i )
        {
            EXPECT_TRUE( preds.addPoint( VertId( i ), pts.defs[i] ) );
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
                    if ( p == q || q == r || r == p || !pts.twoOnLine( p, q, r ) )
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
                }
        EXPECT_FALSE( preds.failed() );

        // no two points on one line: ccw cannot decide
        for ( int p = 0; p < n && !preds.failed(); ++p )
            for ( int q = p + 1; q < n && !preds.failed(); ++q )
                for ( int r = q + 1; r < n && !preds.failed(); ++r )
                    if ( !pts.twoOnLine( p, q, r ) )
                    {
                        (void)preds.ccw( VertId( p ), VertId( q ), VertId( r ) );
                        EXPECT_TRUE( preds.failed() );
                        ++numUndecided;
                    }
    }
    EXPECT_GT( numOrders, 3000 );
    EXPECT_GT( numBehind, 3000 );
    EXPECT_GT( numTriangles, 10000 );
    EXPECT_GT( numSideTriangles, 100 );
    EXPECT_GT( numUndecided, 10 );
}

// Points on a small grid, where many of them coincide and many triangles are degenerate before the perturbation:
// the predicates must be exact where the points are in general position, and consistent elsewhere;
// the order must be a strict total order, on coincident points the expansion of the perturbation must agree with the faster rules,
// and along a side the order must be the one of the boolean's cut
TEST( MRMesh, CutTrianglePredicatesGrid )
{
    RandomVerts rv( 1, 2 );
    CutTrianglePredicates preds;
    int numTriangles = 0, numCoincident = 0, numSidePairs = 0, numDegenerate = 0, numCoincidentBehind = 0;
    for ( int trial = 0; trial < 400; ++trial )
    {
        rv.newTrial();
        const Tri f = rv.tri();
        if ( !preds.reset( f ) )
            continue; // a degenerate triangle
        ++numTriangles;
        const auto [kx, ky] = projectionAxes( f );
        std::vector<Tri> others;
        for ( int i = 0; i < 4; ++i )
            others.push_back( rv.tri() );
        // a segment parallel to the plane of its triangle defines no point
        const auto pts = cutPoints( f, others, [] ( const PreciseCutPoint& p ) { return smallFraction( p ).has_value(); } );
        const int n = int( pts.defs.size() );
        std::vector<SmallFraction> pos;
        for ( int i = 0; i < n; ++i )
        {
            ASSERT_TRUE( preds.addPoint( VertId( i ), pts.defs[i] ) );
            pos.push_back( *smallFraction( pts.defs[i] ) );
        }

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
                int c = compareSmall( pos[p], pos[q], kx );
                if ( c == 0 )
                    c = compareSmall( pos[p], pos[q], ky );
                if ( c != 0 )
                {
                    EXPECT_EQ( bool( less[p * n + q] ), c < 0 );
                    continue;
                }
                // coincident before the perturbation
                ++numCoincident;
                const auto sos = preds.lessByPerturbation( VertId( p ), VertId( q ) );
                ASSERT_TRUE( sos.has_value() );
                EXPECT_EQ( *sos, bool( less[p * n + q] ) );
            }
        }
        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
                for ( int r = 0; r < n; ++r )
                    if ( less[p * n + q] && less[q * n + r] )
                    {
                        EXPECT_TRUE( less[p * n + r] );
                    }

        // two crossings of a side with other triangles come along it in the order of segmentIntersectionOrder
        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
            {
                const auto& a = pts.defs[p];
                const auto& b = pts.defs[q];
                const int side = pts.lines[p] & pts.lines[q] & 7;
                if ( p == q || !side || a.isVertex() || b.isVertex() )
                    continue;
                const int s = std::countr_zero( unsigned( side ) );
                const Vector3i u = f[( s + 1 ) % 3].pt - f[s].pt; // the segment of both points
                const bool alongSweep = u[kx] != 0 ? u[kx] > 0 : u[ky] > 0;
                const bool aFirst = segmentIntersectionOrder( { a.seg[0], a.seg[1], a.tri[0], a.tri[1], a.tri[2], b.tri[0], b.tri[1], b.tri[2] } );
                EXPECT_EQ( bool( less[p * n + q] ) == alongSweep, aFirst );
                ++numSidePairs;
            }

        // a triangle with two vertices on one line: exact unless degenerate before the perturbation,
        // the same in any rotation of its vertices, and the opposite when flipped
        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
            {
                if ( p == q || !pts.onLine( p, q ) )
                    continue;
                for ( int r = 0; r < n; ++r )
                {
                    if ( r == p || r == q )
                        continue;
                    const bool c = preds.ccw( VertId( p ), VertId( q ), VertId( r ) );
                    EXPECT_EQ( preds.ccw( VertId( q ), VertId( r ), VertId( p ) ), c );
                    EXPECT_EQ( preds.ccw( VertId( r ), VertId( p ), VertId( q ) ), c );
                    EXPECT_NE( preds.ccw( VertId( q ), VertId( p ), VertId( r ) ), c );
                    if ( const int o = orientSmall( pos[p], pos[q], pos[r], kx, ky ) )
                    {
                        EXPECT_EQ( c, o > 0 );
                    }
                    else
                        ++numDegenerate;
                }
            }

        // seen from behind: exact where the points differ before the perturbation, opposite for the edge taken backward
        for ( int p = 0; p < n; ++p )
            for ( int q = 0; q < n; ++q )
            {
                if ( p == q )
                    continue;
                const int cy = compareSmall( pos[q], pos[p], ky ), cx = compareSmall( pos[q], pos[p], kx );
                if ( cy != 0 || cx != 0 )
                {
                    EXPECT_EQ( preds.ccwFromBehind( VertId( q ), VertId( p ) ), cy != 0 ? cy < 0 : cx > 0 );
                }
                else if ( pts.onLine( p, q ) )
                {
                    EXPECT_NE( preds.ccwFromBehind( VertId( q ), VertId( p ) ), preds.ccwFromBehind( VertId( p ), VertId( q ) ) );
                    ++numCoincidentBehind;
                }
            }
        EXPECT_FALSE( preds.failed() );
    }
    EXPECT_GT( numTriangles, 200 );
    EXPECT_GT( numCoincident, 100 );
    EXPECT_GT( numSidePairs, 150 );
    EXPECT_GT( numDegenerate, 300 );
    EXPECT_GT( numCoincidentBehind, 50 );
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
    ASSERT_TRUE( preds.reset( f ) );
    for ( int i = 0; i < 3; ++i )
        ASSERT_TRUE( preds.addPoint( VertId( i ), vertexPoint( f[i] ) ) );
    ASSERT_TRUE( preds.addPoint( 3_v, { { t1[1], t1[2] }, f } ) );
    ASSERT_TRUE( preds.addPoint( 4_v, { { f[1], f[2] }, t2 } ) );
    ASSERT_TRUE( preds.addPoint( 5_v, { { t3[0], t3[1] }, f } ) );

    EXPECT_FALSE( preds.ccw( 0_v, 3_v, 5_v ) );
    EXPECT_TRUE( preds.ccw( 0_v, 5_v, 3_v ) );
    EXPECT_FALSE( preds.ccw( 0_v, 4_v, 5_v ) );
    EXPECT_TRUE( preds.ccw( 5_v, 4_v, 0_v ) );
    EXPECT_FALSE( preds.failed() );
}

} // namespace MR

