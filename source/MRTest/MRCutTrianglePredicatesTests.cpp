#include <MRMesh/MRCutTrianglePredicates.h>
#include <MRMesh/MRPrecisePredicates3.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cfloat>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>

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
    // a triangle around point c
    Tri triAround( const Vector3i& c )
    {
        Tri res = tri();
        for ( auto& v : res )
            v.pt += c;
        return res;
    }
    std::mt19937& gen() { return gen_; }

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

PreciseCutLine cutLine( const Tri& plane, bool alongCross = false )
{
    return { .plane = plane, .alongCross = alongCross };
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

// the common point of the planes of three triangles in doubles
Vector3d triplePoint( const Tri& f, const Tri& g, const Tri& h )
{
    const Vector3d nf = normalOf( f ), ng = normalOf( g ), nh = normalOf( h );
    const double df = dot( nf, toDouble( f[0] ) ), dg = dot( ng, toDouble( g[0] ) ), dh = dot( nh, toDouble( h[0] ) );
    return ( df * cross( ng, nh ) + dg * cross( nh, nf ) + dh * cross( nf, ng ) ) / dot( nf, cross( ng, nh ) );
}

// how far inside triangle t point p is, relative to the size of the triangle; negative outside
double insideness( const Tri& t, const Vector3d& p )
{
    const Vector3d n = normalOf( t );
    double res = DBL_MAX;
    for ( int i = 0; i < 3; ++i )
    {
        const Vector3d a = toDouble( t[i] ), b = toDouble( t[( i + 1 ) % 3] );
        res = std::min( res, dot( cross( b - a, p - a ), n ) / ( n.lengthSq() ) );
    }
    return res;
}

// the edge-triangle points at the ends of the segment where triangles s and t cross, or nothing if they do not cross
std::optional<std::array<PreciseCutPoint, 2>> crossingSegment( const Tri& s, const Tri& t )
{
    std::vector<PreciseCutPoint> pts;
    for ( int i = 0; i < 3; ++i )
    {
        if ( doTriangleSegmentIntersect( { t[0], t[1], t[2], s[i], s[( i + 1 ) % 3] } ) )
            pts.push_back( { { s[i], s[( i + 1 ) % 3] }, t } );
        if ( doTriangleSegmentIntersect( { s[0], s[1], s[2], t[i], t[( i + 1 ) % 3] } ) )
            pts.push_back( { { t[i], t[( i + 1 ) % 3] }, s } );
    }
    if ( pts.size() != 2 )
    {
        auto str = [] ( const Tri& t )
        {
            std::ostringstream ss;
            for ( const auto& v : t )
                ss << " " << int( v.id ) << ":(" << v.pt.x << "," << v.pt.y << "," << v.pt.z << ")";
            return ss.str();
        };
        EXPECT_TRUE( pts.empty() ) << pts.size() << " s" << str( s ) << " t" << str( t ); // the perturbed triangles cross along a segment with two ends, or not at all
        return {};
    }
    return std::array<PreciseCutPoint, 2>{ pts[0], pts[1] };
}

const ConvertToFloatVector cIdentity{ .range = 1, .center = {} };

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

std::optional<SmallFraction> smallTriplePoint( const Tri& f, const Tri& g, const Tri& h )
{
    auto plane = [] ( const Tri& t )
    {
        const Vector3i64 a( t[0].pt );
        const auto n = cross( Vector3i64( t[1].pt ) - a, Vector3i64( t[2].pt ) - a );
        return std::pair{ n, dot( n, a ) };
    };
    const auto [nf, df] = plane( f );
    const auto [ng, dg] = plane( g );
    const auto [nh, dh] = plane( h );
    SmallFraction res{ df * cross( ng, nh ) + dg * cross( nh, nf ) + dh * cross( nf, ng ), dot( nf, cross( ng, nh ) ) };
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

bool sameSet( const Tri& a, const Tri& b )
{
    for ( const auto& v : a )
        if ( v.id != b[0].id && v.id != b[1].id && v.id != b[2].id )
            return false;
    return true;
}

bool sameDefinition( const PreciseCutPoint& a, const PreciseCutPoint& b )
{
    if ( a.isVertex() || b.isVertex() )
        return a.isVertex() && b.isVertex() && a.seg[0].id == b.seg[0].id;
    const bool sameSeg = ( a.seg[0].id == b.seg[0].id && a.seg[1].id == b.seg[1].id ) || ( a.seg[0].id == b.seg[1].id && a.seg[1].id == b.seg[0].id );
    return sameSeg && sameSet( a.tri, b.tri );
}

} // anonymous namespace

// Points of random triangles in general position: the order and the sides must agree with the positions in doubles
TEST( MRMesh, CutTrianglePredicatesRandom )
{
    RandomVerts rv( 0, 1 << 20 );
    CutTrianglePredicates preds;
    int numOrders = 0, numSides = 0;
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

        std::vector<PreciseCutPoint> defs;
        for ( const auto& v : f )
            defs.push_back( vertexPoint( v ) );
        std::vector<Tri> planes;
        for ( int i = 0; i < 6; ++i )
        {
            const Tri t = rv.tri();
            planes.push_back( t );
            for ( int s = 0; s < 3; ++s )
            {
                if ( doTriangleSegmentIntersect( { t[0], t[1], t[2], f[s], f[( s + 1 ) % 3] } ) )
                    defs.push_back( { { f[s], f[( s + 1 ) % 3] }, t } ); // a side crossing another triangle
                if ( doTriangleSegmentIntersect( { f[0], f[1], f[2], t[s], t[( s + 1 ) % 3] } ) )
                    defs.push_back( { { t[s], t[( s + 1 ) % 3] }, f } ); // an edge of another triangle crossing this one
            }
        }
        std::vector<Vector3d> pos;
        for ( const auto& d : defs )
        {
            EXPECT_TRUE( preds.addPoint( VertId( pos.size() ), d ) );
            pos.push_back( positionOf( d ) );
        }
        const size_t numNotCrossings = pos.size();
        for ( size_t i = 0; i < planes.size(); ++i )
            for ( size_t j = i + 1; j < planes.size(); ++j )
            {
                const auto x = triplePoint( f, planes[i], planes[j] );
                if ( insideness( f, x ) < 1e-3 )
                    continue;
                const VertId v( pos.size() );
                EXPECT_TRUE( preds.addCrossing( v, cutLine( planes[i] ), cutLine( planes[j] ), cIdentity ) );
                EXPECT_LT( ( Vector3d( preds.crossingPos( v ) ) - x ).length(), 1.0 );
                pos.push_back( x );
            }

        for ( size_t p = 0; p < pos.size(); ++p )
            for ( size_t q = 0; q < pos.size(); ++q )
            {
                if ( std::abs( pos[p][kx] - pos[q][kx] ) < 1e-3 )
                    continue;
                EXPECT_EQ( preds.less( VertId( p ), VertId( q ) ), pos[p][kx] < pos[q][kx] );
                ++numOrders;
            }

        for ( const auto& plane : planes )
        {
            const Vector3d n = normalOf( plane );
            for ( size_t p = 0; p < numNotCrossings; ++p )
            {
                const double dist = dot( n, pos[p] - toDouble( plane[0] ) ) / n.length();
                if ( std::abs( dist ) < 1e-3 )
                    continue;
                for ( int along = 0; along < 2; ++along )
                    for ( int reversed = 0; reversed < 2; ++reversed )
                    {
                        EXPECT_EQ( preds.leftOf( cutLine( plane, along != 0 ), reversed != 0, VertId( p ) ), ( dist < 0 ) == ( along != reversed ) );
                    }
                ++numSides;
            }
            // the direction of the cut, cross( normal of the triangle, normal of the plane ), seen from far behind against the sweep
            const Vector3d u = cross( normalOf( f ), n );
            if ( std::abs( u[ky] ) > 1e-6 * u.length() )
            {
                EXPECT_EQ( preds.ccwFromBehind( cutLine( plane, true ), false ), u[ky] < 0 );
                EXPECT_EQ( preds.ccwFromBehind( cutLine( plane, true ), true ), u[ky] > 0 );
                EXPECT_EQ( preds.ccwFromBehind( cutLine( plane, false ), false ), u[ky] > 0 );
            }
        }
        for ( int s = 0; s < 3; ++s )
        {
            const PreciseCutLine side{ .side = s };
            for ( size_t p = 0; p < numNotCrossings; ++p )
            {
                EXPECT_TRUE( preds.leftOf( side, false, VertId( p ) ) );
                EXPECT_FALSE( preds.leftOf( side, true, VertId( p ) ) );
            }
            const Vector3i u = f[( s + 1 ) % 3].pt - f[s].pt;
            EXPECT_EQ( preds.ccwFromBehind( side, false ), u[ky] != 0 ? u[ky] < 0 : u[kx] > 0 );
            EXPECT_EQ( preds.ccwFromBehind( side, true ), u[ky] != 0 ? u[ky] > 0 : u[kx] < 0 );
        }
        EXPECT_FALSE( preds.failed() );
    }
    EXPECT_GT( numOrders, 5000 );
    EXPECT_GT( numSides, 1000 );
}

// Points on a small grid, where many of them coincide before the perturbation: the order must be a strict total order, exact
// where the points differ, and on coincident points the expansion of the perturbation must agree with the faster rules,
// which follow the boolean's own predicates
TEST( MRMesh, CutTrianglePredicatesGrid )
{
    RandomVerts rv( 1, 2 );
    CutTrianglePredicates preds;
    int numCoincident = 0, numTriangles = 0;
    for ( int trial = 0; trial < 400; ++trial )
    {
        rv.newTrial();
        const Tri f = rv.tri();
        if ( !preds.reset( f ) )
            continue; // a degenerate triangle
        ++numTriangles;
        const auto [kx, ky] = projectionAxes( f );

        std::vector<PreciseCutPoint> defs;
        for ( const auto& v : f )
            defs.push_back( vertexPoint( v ) );
        auto addDef = [&] ( const PreciseCutPoint& d )
        {
            if ( !smallFraction( d ) )
                return; // a segment parallel to the plane: not defined
            for ( const auto& e : defs )
                if ( sameDefinition( d, e ) )
                    return;
            defs.push_back( d );
        };
        std::vector<Tri> planes;
        for ( int i = 0; i < 4; ++i )
        {
            const Tri t = rv.tri();
            planes.push_back( t );
            for ( int s = 0; s < 3; ++s )
            {
                if ( doTriangleSegmentIntersect( { t[0], t[1], t[2], f[s], f[( s + 1 ) % 3] } ) )
                    addDef( { { f[s], f[( s + 1 ) % 3] }, t } );
                if ( doTriangleSegmentIntersect( { f[0], f[1], f[2], t[s], t[( s + 1 ) % 3] } ) )
                    addDef( { { t[s], t[( s + 1 ) % 3] }, f } );
            }
        }
        std::vector<SmallFraction> pos;
        for ( const auto& d : defs )
        {
            ASSERT_TRUE( preds.addPoint( VertId( pos.size() ), d ) );
            pos.push_back( *smallFraction( d ) );
        }
        for ( size_t i = 0; i < planes.size(); ++i )
            for ( size_t j = i + 1; j < planes.size(); ++j )
            {
                const auto x = smallTriplePoint( f, planes[i], planes[j] );
                if ( !x )
                    continue;
                ASSERT_TRUE( preds.addCrossing( VertId( pos.size() ), cutLine( planes[i] ), cutLine( planes[j] ), cIdentity ) );
                pos.push_back( *x );
            }

        const int n = int( pos.size() );
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
        EXPECT_FALSE( preds.failed() );
    }
    EXPECT_GT( numTriangles, 200 );
    EXPECT_GT( numCoincident, 300 );
}

// A crossing of two cuts inside a triangle is where the planes of three triangles meet: each of them must find it,
// at the same position
TEST( MRMesh, CutTrianglePredicatesTriplePoints )
{
    for ( int halfSize : { 1 << 20, 2 } )
    {
        RandomVerts rv( 2, halfSize );
        CutTrianglePredicates preds;
        int numCrossings = 0, numChecked = 0;
        for ( int trial = 0; trial < 3000; ++trial )
        {
            rv.newTrial();
            const std::array<Tri, 3> t{ rv.tri(), rv.tri(), rv.tri() };
            std::array<std::array<PreciseCutPoint, 2>, 3> segs; // segs[i] is where the two triangles other than t[i] cross
            bool allCross = true;
            for ( int i = 0; i < 3 && allCross; ++i )
            {
                const auto s = crossingSegment( t[( i + 1 ) % 3], t[( i + 2 ) % 3] );
                if ( s )
                    segs[i] = *s;
                else
                    allCross = false;
            }
            if ( !allCross )
                continue;
            std::optional<bool> found;
            std::optional<Vector3f> foundPos;
            int numFaces = 0;
            for ( int i = 0; i < 3; ++i )
            {
                // in triangle t[i], the cuts by the other two triangles are segs[(i+2)%3] (by t[(i+1)%3]) and segs[(i+1)%3] (by t[(i+2)%3])
                if ( !preds.reset( t[i] ) )
                    break;
                const auto& sa = segs[( i + 2 ) % 3];
                const auto& sb = segs[( i + 1 ) % 3];
                bool defined = true;
                for ( int j = 0; j < 2; ++j )
                {
                    defined = preds.addPoint( VertId( j ), sa[j] ) && defined;
                    defined = preds.addPoint( VertId( 2 + j ), sb[j] ) && defined;
                }
                if ( !defined )
                    break;
                ++numFaces;
                const auto la = cutLine( t[( i + 1 ) % 3] ), lb = cutLine( t[( i + 2 ) % 3] );
                const bool cross = preds.leftOf( la, false, VertId( 2 ) ) != preds.leftOf( la, false, VertId( 3 ) )
                    && preds.leftOf( lb, false, VertId( 0 ) ) != preds.leftOf( lb, false, VertId( 1 ) );
                EXPECT_FALSE( preds.failed() );
                if ( found )
                {
                    EXPECT_EQ( *found, cross );
                }
                found = cross;
                if ( !cross || !preds.addCrossing( 4_v, la, lb, cIdentity ) )
                    continue;
                const auto pos = preds.crossingPos( 4_v );
                if ( foundPos )
                {
                    EXPECT_EQ( *foundPos, pos );
                }
                foundPos = pos;
            }
            if ( numFaces < 3 )
                continue;
            ++numChecked;
            if ( *found )
                ++numCrossings;
            if ( halfSize > 2 )
            {
                // far from the sides of the triangles, the common point decides
                const auto x = triplePoint( t[0], t[1], t[2] );
                const double in = std::min( { insideness( t[0], x ), insideness( t[1], x ), insideness( t[2], x ) } );
                if ( std::abs( in ) > 1e-6 )
                {
                    EXPECT_EQ( *found, in > 0 );
                }
            }
        }
        EXPECT_GT( numChecked, 100 );
        EXPECT_GT( numCrossings, 10 );
    }
}

// Several crossings on one cut: both triangles holding the cut must order them the same way along it
TEST( MRMesh, CutTrianglePredicatesOrderAlongCut )
{
    for ( int halfSize : { 1 << 20, 3 } )
    {
        RandomVerts rv( 3, halfSize );
        CutTrianglePredicates preds;
        int numChecked = 0;
        for ( int trial = 0; trial < 100000 && numChecked < 150; ++trial )
        {
            // triangles f and g cross along the cut, and h1, h2 around its middle may cross both of them
            rv.newTrial();
            const Tri f = rv.tri(), g = rv.tri();
            const auto cut = crossingSegment( f, g );
            if ( !cut )
                continue;
            const Vector3d mid = 0.5 * ( positionOf( ( *cut )[0] ) + positionOf( ( *cut )[1] ) );
            if ( !std::isfinite( mid.x ) || !std::isfinite( mid.y ) || !std::isfinite( mid.z ) )
                continue; // an end of the cut is not defined without the perturbation
            const Vector3i c( int( std::round( mid.x ) ), int( std::round( mid.y ) ), int( std::round( mid.z ) ) );
            const Tri h1 = rv.triAround( c ), h2 = rv.triAround( c );
            // in triangle a, cut by b: whether the crossings of the cut by h1 and h2 are defined, and if both exist, their order along
            // the cut from its first end
            struct Outcome
            {
                bool defined = false;
                std::optional<bool> order;
            };
            auto orderIn = [&] ( const Tri& a, const Tri& b )
            {
                Outcome res;
                if ( !preds.reset( a ) )
                    return res;
                const auto s1 = crossingSegment( a, h1 ), s2 = crossingSegment( a, h2 );
                bool defined = true;
                for ( int j = 0; j < 2; ++j )
                {
                    defined = preds.addPoint( VertId( j ), ( *cut )[j] ) && defined;
                    if ( s1 )
                        defined = preds.addPoint( VertId( 2 + j ), ( *s1 )[j] ) && defined;
                    if ( s2 )
                        defined = preds.addPoint( VertId( 4 + j ), ( *s2 )[j] ) && defined;
                }
                const auto lc = cutLine( b ), l1 = cutLine( h1 ), l2 = cutLine( h2 );
                auto crosses = [&] ( const PreciseCutLine& l, VertId v )
                {
                    return preds.leftOf( lc, false, v ) != preds.leftOf( lc, false, v + 1 )
                        && preds.leftOf( l, false, 0_v ) != preds.leftOf( l, false, 1_v );
                };
                if ( !defined || !s1 || !s2 || !crosses( l1, 2_v ) || !crosses( l2, 4_v ) )
                {
                    res.defined = defined;
                    return res;
                }
                res.defined = preds.addCrossing( 6_v, lc, l1, cIdentity ) && preds.addCrossing( 7_v, lc, l2, cIdentity );
                if ( !res.defined )
                    return res;
                EXPECT_FALSE( preds.failed() );
                // the sweep moves along the cut either way, so the order along it follows from the order of its ends
                res.order = preds.less( 6_v, 7_v ) == preds.less( 0_v, 1_v );
                return res;
            };
            const auto inF = orderIn( f, g );
            const auto inG = orderIn( g, f );
            if ( !inF.defined || !inG.defined )
                continue; // a degenerate triangle, or a point not defined without the perturbation
            EXPECT_EQ( inF.order.has_value(), inG.order.has_value() );
            if ( !inF.order || !inG.order )
                continue;
            EXPECT_EQ( *inF.order, *inG.order );
            ++numChecked;
            if ( halfSize > 3 )
            {
                const auto p = positionOf( ( *cut )[0] ), q = positionOf( ( *cut )[1] );
                const double t1 = dot( triplePoint( f, g, h1 ) - p, q - p ), t2 = dot( triplePoint( f, g, h2 ) - p, q - p );
                if ( std::abs( t1 - t2 ) > 1e-6 * ( q - p ).lengthSq() )
                {
                    EXPECT_EQ( *inF.order, t1 < t2 );
                }
            }
        }
        EXPECT_GT( numChecked, 100 );
    }
}

} // namespace MR
