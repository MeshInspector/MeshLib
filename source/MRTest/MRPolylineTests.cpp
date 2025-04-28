#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, Polyline2 )
{
    Contour2f cont;
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );
    cont.push_back( Vector2f( 1.f, 1.f ) );

    Contour2f cont2;
    cont2.push_back( Vector2f( 2.f, 0.f ) );
    cont2.push_back( Vector2f( 3.f, 0.f ) );
    cont2.push_back( Vector2f( 2.f, 1.f ) );
    cont2.push_back( Vector2f( 3.f, 1.f ) );

    Contours2f conts{ cont,cont2 };

    Polyline2 pl( conts );
    auto conts2 = pl.contours();

    for ( auto i = 0; i < conts.size(); i++ )
{
        auto& c1 = conts[i];
        auto& c2 = conts2[i];
        for ( auto j = 0; j < c1.size(); j++ )
{
            auto v1 = c1[j];
            auto v2 = c2[j];
            EXPECT_NEAR( v1[0], v2[0], 1e-8 );
            EXPECT_NEAR( v1[1], v2[1], 1e-8 );
        }
    }
}

TEST( MRMesh, Polyline2LoopDir )
{
    Contour2f cont;
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 1.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );

    Polyline2 plNotClosed( { cont } );
    EXPECT_TRUE( plNotClosed.loopDirArea( 0_e ).z == FLT_MAX );

    cont.push_back( Vector2f( 0.f, 0.f ) );

    Polyline2 plClosed( { cont } );
    EXPECT_TRUE( plClosed.loopDirArea( 0_e ).z > 0.0f );
    EXPECT_TRUE( plClosed.loopDirArea( 1_e ).z < 0.0f );
}

TEST( MRMesh, Polyline3 )
{
    Contour3f cont;
    cont.push_back( Vector3f( 0.f, 0.f, 0.f ) );
    cont.push_back( Vector3f( 1.f, 0.f, 0.f ) );
    cont.push_back( Vector3f( 0.f, 1.f, 0.f ) );
    cont.push_back( Vector3f( 1.f, 1.f, 0.f ) );

    Contour3f cont2;
    cont2.push_back( Vector3f( 2.f, 0.f, 0.f ) );
    cont2.push_back( Vector3f( 3.f, 0.f, 0.f ) );
    cont2.push_back( Vector3f( 2.f, 1.f, 0.f ) );
    cont2.push_back( Vector3f( 3.f, 1.f, 0.f ) );

    Contours3f conts{ cont,cont2 };

    Polyline3 pl( conts );
    auto conts2 = pl.contours();

    for ( auto i = 0; i < conts.size(); i++ )
    {
        auto& c1 = conts[i];
        auto& c2 = conts2[i];
        for ( auto j = 0; j < c1.size(); j++ )
        {
            auto v1 = c1[j];
            auto v2 = c2[j];
            EXPECT_NEAR( v1[0], v2[0], 1e-8 );
            EXPECT_NEAR( v1[1], v2[1], 1e-8 );
        }
    }
}

TEST( MRMesh, PolylineSplitEdge )
{
    Contour2f cont;
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );
    Polyline2 polyline( { cont } );

    EXPECT_EQ( polyline.topology.numValidVerts(), 2 );
    EXPECT_EQ( polyline.points.size(), 2 );
    EXPECT_EQ( polyline.topology.lastNotLoneEdge(), EdgeId(1) ); // 1*2 = 2 half-edges in total

    auto e01 = polyline.topology.findEdge( 0_v, 1_v );
    EXPECT_TRUE( e01.valid() );
    auto ex = polyline.splitEdge( e01 );
    VertId v01 = polyline.topology.org( e01 );
    EXPECT_EQ( polyline.topology.dest( ex ), v01 );
    EXPECT_EQ( polyline.topology.numValidVerts(), 3 );
    EXPECT_EQ( polyline.points.size(), 3 );
    EXPECT_EQ( polyline.topology.lastNotLoneEdge(), EdgeId(3) ); // 2*2 = 4 half-edges in total
    EXPECT_EQ( polyline.points[v01], ( Vector2f(.5f, 0.f) ) );
}

} //namespace MR
