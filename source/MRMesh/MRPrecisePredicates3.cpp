#include "MRPrecisePredicates3.h"
#include "MRHighPrecision.h"
#include "MRVector2.h"
#include "MRBox.h"
#include "MRGTest.h"

#if __has_include(<__msvc_int128.hpp>)
  #include <__msvc_int128.hpp>
  // this type is much faster than boost::multiprecision::checked_int128_t but lacks conversion in double and sqrt-function
  using int128_t = std::_Signed128;
#else
  using int128_t = __int128_t;
#endif

namespace
{
// INT_MAX in double for mapping in int range
constexpr double cRangeIntMax = 0.99 * std::numeric_limits<int>::max(); // 0.99 to be sure the no overflow will ever happen due to rounding errors
}

namespace MR
{

using Vector3hpFast = Vector3<int128_t>;

bool orient3d( const Vector3i & a, const Vector3i& b, const Vector3i& c )
{
    auto vhp = dot( Vector3hpFast{ a }, Vector3hpFast{ cross( Vector3ll{ b }, Vector3ll{ c } ) } );
    if ( vhp ) return vhp > 0;

    auto v = cross( Vector2ll{ b.x, b.y }, Vector2ll{ c.x, c.y } );
    if ( v ) return v > 0;

    v = -cross( Vector2ll{ b.x, b.z }, Vector2ll{ c.x, c.z } );
    if ( v ) return v > 0;

    v = cross( Vector2ll{ b.y, b.z }, Vector2ll{ c.y, c.z } );
    if ( v ) return v > 0;

    v = -cross( Vector2ll{ a.x, a.y }, Vector2ll{ c.x, c.y } );
    if ( v ) return v > 0;

    if ( c.x ) return c.x > 0;

    if ( c.y ) return c.y < 0;

    v = cross( Vector2ll{ a.x, a.z }, Vector2ll{ c.x, c.z } );
    if ( v ) return v > 0;

    if ( c.z ) return c.z > 0;

#ifndef NDEBUG
    v = -cross( Vector2ll{ a.y, a.z }, Vector2ll{ c.y, c.z } );
    assert( v == 0 );
    if ( v ) return v > 0;
#endif

    v = cross( Vector2ll{ a.x, a.y }, Vector2ll{ b.x, b.y } );
    if ( v ) return v > 0;

    if ( b.x ) return b.x < 0;

    if ( b.y ) return b.y > 0;

    if ( a.x ) return a.x > 0;

    return true;
}

bool orient3d( const PreciseVertCoords* vs )
{
    bool odd = false;
    std::array<int, 4> order = { 0, 1, 2, 3 };

    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = i + 1; j < 4; ++j )
        {
            assert( vs[order[i]].id != vs[order[j]].id );
            if ( vs[order[i]].id > vs[order[j]].id )
            {
                odd = !odd;
                std::swap( order[i], order[j] );
            }
        }
    }

    return odd != orient3d( vs[order[0]].pt, vs[order[1]].pt, vs[order[2]].pt, vs[order[3]].pt );
}

bool orient3d( const std::array<PreciseVertCoords, 4> & vs )
{
    return orient3d( vs.data() );
}

TriangleSegmentIntersectResult doTriangleSegmentIntersect( const std::array<PreciseVertCoords, 5> & vs )
{
    TriangleSegmentIntersectResult res;
    constexpr int a = 0;
    constexpr int b = 1;
    constexpr int c = 2;
    constexpr int d = 3;
    constexpr int e = 4;

    auto orient3d = [&]( int p, int q, int r, int s )
    {
        return MR::orient3d( { vs[p], vs[q], vs[r], vs[s] } );
    };

    const auto abcd = orient3d( a, b, c, d );
    res.dIsLeftFromABC = abcd;
    const auto abce = orient3d( a, b, c, e );
    if ( abcd == abce )
        return res; // segment DE is located at one side of the plane ABC

    const auto dabe = orient3d( a, b, d, e );
    const auto dbce = orient3d( b, c, d, e );
    if ( dabe != dbce )
        return res; // segment AC is located at one side of the plane DEB

    const auto dcae = !orient3d( a, c, d, e ); // '!' is due to inverted order of a and c
    if ( dbce != dcae )
        return res; // segment AB is located at one side of the plane DEC

    assert ( dcae == dabe ); // segment BC is crossed by the plane DEA

    res.doIntersect = true;
    return res;
}

ConvertToIntVector getToIntConverter( const Box3d& box )
{
    Vector3d center{ box.center() };
    auto bbSize = box.size();
    double maxDim = std::max( { bbSize[0],bbSize[1],bbSize[2] } );

    // range is selected so that after centering each integer point is within [-max/2; +max/2] range,
    // so the difference of any two points will be within [-max; +max] range
    double invRange = cRangeIntMax / maxDim;

    return [invRange, center] ( const Vector3f& v )
    {
        // perform intermediate operations in double for better precision
        return Vector3i( ( Vector3d{ v } - center ) * invRange );
    };
}

ConvertToFloatVector getToFloatConverter( const Box3d& box )
{
    Vector3d center{ box.center() };
    auto bbSize = box.size();
    double maxDim = std::max( { bbSize[0],bbSize[1],bbSize[2] } );

    // range is selected so that after centering each integer point is within [-max/2; +max/2] range,
    // so the difference of any two points will be within [-max; +max] range
    double range = maxDim / cRangeIntMax;

    return [range, center] ( const Vector3i& v )
    {
        return Vector3f( Vector3d{ v }*range + center );
    };
}

// ab - segment
// cd - segment
// if segments intersects - returns intersection point, nullopt otherwise
std::optional<Vector3i> findTwoSegmentsIntersection( const Vector3i& ai, const Vector3i& bi, const Vector3i& ci, const Vector3i& di )
{
    auto ab = Vector3hp{ bi - ai };
    auto ac = Vector3hp{ ci - ai };
    auto ad = Vector3hp{ di - ai };
    auto abc = cross( ab, ac );
    auto abd = cross( ab, ad );

    if ( dot( abc, abd ) > 0 )
        return std::nullopt; // CD is on one side of AB

    auto cd = Vector3hp{ di - ci };
    auto cb = Vector3hp{ bi - ci };
    auto cda = cross( cd, -ac );
    auto cdb = cross( cd, cb );
    if ( dot( cda, cdb ) > 0 )
        return std::nullopt; // AB is on one side of CD

    auto abcHSq = abc.lengthSq();
    auto abdHSq = abd.lengthSq();
    if ( ( abcHSq == 0 && abdHSq == 0 ) || ( cda.lengthSq() == 0 && cdb.lengthSq() == 0 ) ) // collinear
    {
        auto dAC = dot( ab, ac );
        auto dAD = dot( ab, ad );
        if ( dAC < 0 && dAD < 0 )
            return std::nullopt; // both C and D are lower than A (on the AB segment)

        auto dBC = dot( -ab, -cb );
        auto dBD = dot( -ab, Vector3hp{ di - bi } );
        if ( dBC < 0 && dBD < 0 )
            return std::nullopt; // both C and D are greater than B (on the AB segment)

        // have common points
        auto onePoint = dAC < 0 ? ai : ci; // find point that is closer to B
        auto otherPoint = dBD < 0 ? bi : di; // find point that is closer to A
        return Vector3i( ( onePoint + otherPoint ) / 2 ); // return middle point of overlapping segment
    }

    // common intersection - non-collinear
    auto abcS = boost::multiprecision::sqrt( abcHSq );
    auto abdS = boost::multiprecision::sqrt( abdHSq );
    return Vector3i( Vector3d( abdS * Vector3hp{ ci } + abcS * Vector3hp{ di } ) / double( abcS + abdS ) );
}

/// https://stackoverflow.com/a/18067292/7325599
template <class T>
T divRoundClosest( T n, T d )
{
    return ((n < 0) == (d < 0)) ? ((n + d/2)/d) : ((n - d/2)/d);
}

template <class T>
Vector3<T> divRoundClosest( const Vector3<T>& n, T d )
{
    return
    {
        divRoundClosest( n.x, d ),
        divRoundClosest( n.y, d ),
        divRoundClosest( n.z, d )
    };
}

Vector3f findTriangleSegmentIntersectionPrecise( 
    const Vector3f& a, const Vector3f& b, const Vector3f& c, 
    const Vector3f& d, const Vector3f& e, 
    CoordinateConverters converters )
{
    auto ai = converters.toInt( a );
    auto bi = converters.toInt( b );
    auto ci = converters.toInt( c );
    auto di = converters.toInt( d );
    auto ei = converters.toInt( e );
    auto abcd = dot( Vector3hpFast{ ai - di }, Vector3hpFast{ cross( Vector3ll{ bi - di }, Vector3ll{ ci - di } ) } );
    if ( abcd < 0 )
        abcd = -abcd;
    auto abce = dot( Vector3hpFast{ ai - ei }, Vector3hpFast{ cross( Vector3ll{ bi - ei }, Vector3ll{ ci - ei } ) } );
    if ( abce < 0 )
        abce = -abce;
    auto sum = abcd + abce;
    if ( sum != 0 )
        return converters.toFloat( Vector3i{ divRoundClosest( abcd * Vector3hpFast{ ei } + abce * Vector3hpFast{ di }, sum ) } );
    // rare case when `sum == 0` 
    // suggest finding middle point of edge segment laying inside triangle
    Vector3ll sumVec;
    int numSum = 0;
    if ( auto iABDE = findTwoSegmentsIntersection( ai, bi, di, ei ) )
    {
        sumVec += Vector3ll{ *iABDE };
        ++numSum;
    }
    if ( auto iBCDE = findTwoSegmentsIntersection( bi, ci, di, ei ) )
    {
        sumVec += Vector3ll{ *iBCDE };
        ++numSum;
    }
    if ( auto iCADE = findTwoSegmentsIntersection( ci, ai, di, ei ) )
    {
        sumVec += Vector3ll{ *iCADE };
        ++numSum;
    }
    if ( numSum > 0 )
        return converters.toFloat( Vector3i{ Vector3d( sumVec ) / double( numSum ) } );

    // rare case when `numSum == 0` - segment is fully inside face
    return Vector3f( ( Vector3d( d ) + Vector3d( e ) ) * 0.5 );
}

TEST( MRMesh, PrecisePredicates3 )
{
    const std::array<PreciseVertCoords, 5> vs = 
    { 
        PreciseVertCoords{ 0_v, Vector3i(  2,  1, 0 ) }, //a
        PreciseVertCoords{ 1_v, Vector3i{ -2,  1, 0 } }, //b
        PreciseVertCoords{ 2_v, Vector3i{  0, -2, 0 } }, //c

        PreciseVertCoords{ 3_v, Vector3i{  0, 0, -1 } }, //d
        PreciseVertCoords{ 4_v, Vector3i{  0, 0,  1 } }  //e
    };

    auto res = doTriangleSegmentIntersect( vs );

    EXPECT_TRUE( res.doIntersect );
    EXPECT_TRUE( res.dIsLeftFromABC );
}

} //namespace MR
