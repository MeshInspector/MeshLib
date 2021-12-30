#include "MRPrecisePredicates3.h"
#include "MRHighPrecision.h"
#include "MRVector2.h"
#include "MRGTest.h"

namespace MR
{

bool orient3d( const Vector3i & a, const Vector3i & b, const Vector3i & c )
{
    auto vhp = mixed( Vector3hp{ a }, Vector3hp{ b }, Vector3hp{ c } );
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

    v = -cross( Vector2ll{ a.y, a.z }, Vector2ll{ c.y, c.z } );
    if ( v ) return v > 0;

    v = cross( Vector2ll{ a.x, a.y }, Vector2ll{ b.x, b.y } );
    if ( v ) return v > 0;

    if ( b.x ) return b.x < 0;

    if ( b.y ) return b.y > 0;

    if ( a.x ) return a.x > 0;

    return true;
}

bool orient3d( const std::array<PreciseVertCoords, 4> & vs )
{
    return orient3d( vs.data() );
}

bool orient3d( const PreciseVertCoords* vs )
{
    bool odd = false;
    std::array<int, 4> order = {0, 1, 2, 3};

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
    auto abcd = mixed( Vector3hp{ ai - di }, Vector3hp{ bi - di }, Vector3hp{ ci - di } );
    if ( abcd < 0 )
        abcd = -abcd;
    auto abce = mixed( Vector3hp{ ai - ei }, Vector3hp{ bi - ei }, Vector3hp{ ci - ei } );
    if ( abce < 0 )
        abce = -abce;
    auto sum = abcd + abce;
    return converters.toFloat( Vector3i{ Vector3d( abcd * Vector3hp{ ei } + abce * Vector3hp{ di } ) / double( sum ) } );
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
