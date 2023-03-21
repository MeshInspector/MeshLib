#include "MRDistanceMap.h"
#include "MRDistanceMapParams.h"
#include "MRGTest.h"
#include "MRMeshLoad.h"
#include "MRMeshSave.h"
#include "MRVector2.h"
#include "MRUVSphere.h"
#include "MRTimer.h"
#include "MRPolyline.h"
#include "MRVector.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"

namespace MR
{

Contours2f getTestCont1()
{
    Contours2f c;
    // simple case
    c.push_back( { {2.f,1.f},{2.f,4.f},{3.f,4.f},{3.f,1.f},{2.f,1.f} } );

    // a bit more complicated...
    // should be clockwise
    //c.push_back( { {0.f,0.f},{1.f,2.f},{2.5f,2.5f},{3.f,4.f},{5.f,5.f} } );
    //c.push_back( { {5.f,5.f},{4.f,3.f},{2.5f,2.5f},{2.f,1.f},{0.f,0.f} } );
    return c;
}
Contours2f getTestCont2()
{
    Contours2f c;
    //simple case
    c.push_back( { {1.f,2.f},{1.f,3.f} ,{4.f,3.f},{4.f,2.f},{1.f,2.f} } );

    // a bit more complicated...
    // should be clockwise
    //c.push_back( { {0.f,5.f},{1.f,1.f},{5.f,0.f} } );
    //c.push_back( { {4.f,0.f},{4.f,4.f},{0.f,5.f} } );
    return c;
}
TEST( MRMesh, DistanceMapBoolean2D )
{
    auto c1 = getTestCont1();
    auto c2 = getTestCont2();
    Vector2f middlePoint{ 2.5f,2.5f };
    const ContourToDistanceMapParams params( { 16,16 }, Vector2f( 0.5f, 0.5f ), Vector2f( 4.f, 4.f ), true );

    auto unionContours = contourUnion( Polyline2( c1 ), Polyline2( c2 ), params ).contours();
    for ( const auto& cont : unionContours )
    {
        for ( const auto& p : cont )
        {
            EXPECT_GE( ( middlePoint - p ).lengthSq(), 0.5f );
        }
    }

    auto intersectContours = contourIntersection( Polyline2( c1 ), Polyline2( c2 ), params ).contours();
    for ( const auto& cont : intersectContours )
    {
        for ( const auto& p : cont )
        {
            EXPECT_LE( ( middlePoint - p ).lengthSq(), 0.5f );
        }
    }

    auto subContours = contourSubtract( Polyline2( c1 ), Polyline2( c2 ), params ).contours();
    EXPECT_EQ( subContours.size(), 2 );
}

TEST( MRMesh, DistanceMapContours )
{
    Contours2f conts;
    conts.push_back( { {0.0f,0.0f},{0.0f,500.0f},{500.0f,500.0f},{500.0f,0.0f},{0.0f,0.0f} } );
    float pixelSize = 1.0f;
    float offset = 50.0f;
    auto params = ContourToDistanceMapParams( pixelSize, conts, offset, true );
    auto orgDistMap = distanceMapFromContours( Polyline2( conts ), params );
    auto genConts = distanceMapTo2DIsoPolyline( orgDistMap, pixelSize, 0.0f ).contours();
    // fix offset of new contours (as far as [0,0] pixel interpreted as (0,0) coord)
    for( auto& gCon : genConts )
        for( auto& p : gCon )
            p -= Vector2f::diagonal( offset );

    auto genDistMap = distanceMapFromContours( Polyline2( genConts ), params );

    // for visual checks in case of test failure
    //saveDistanceMapToImage( orgDistMap, "org.png" );
    //saveDistanceMapToImage( genDistMap, "gen.png" );

    auto orgXOrg = orgDistMap.resX();
    auto orgYOrg = orgDistMap.resY();
    auto genXOrg = genDistMap.resX();
    auto genYOrg = genDistMap.resY();
    EXPECT_EQ( orgXOrg, genXOrg );
    EXPECT_EQ( orgYOrg, genYOrg );
    for( int x = 0; x < std::min( orgXOrg, genXOrg ); ++x )
    {
        for( int y = 0; y < std::min( orgYOrg, genYOrg ); ++y )
        {
            const auto orgV = orgDistMap.get( x, y );
            const auto genV = genDistMap.get( x, y );
            EXPECT_TRUE( genV && orgV );
            EXPECT_TRUE( (*genV) * (*orgV) >= 0.0f );
        }
    }
}

TEST( MRMesh, DistanceMapSphere )
{
    float pixSize = 0.1f;
    Mesh sphere = makeUVSphere( 1, 100, 100 );
    AffineXf3f xf1 = AffineXf3f(
        Matrix3f(
            Vector3f( 1.f, 0.f, 0.f ),
            Vector3f( 0.f, 1.f, 0.f ),
            Vector3f( 0.f, 0.f, 1.f ) ),
        Vector3f( -1.f, -1.f, -1.f )
    );
    AffineXf3f xf2 = AffineXf3f(
        Matrix3f(
            Vector3f( -1.f, 0.f, 0.f ),
            Vector3f( 0.f, -1.f, 0.f ),
            Vector3f( 0.f, 0.f, -1.f ) ),
        Vector3f( 1.f, 1.f, 1.f )
    );

    MeshToDistanceMapParams params1( xf1, Vector2f{ pixSize,pixSize }, Vector2i{ 10,10 } );
    MeshToDistanceMapParams params2( xf2, Vector2f{ pixSize,pixSize }, Vector2i{ 10,10 } );
    auto dm1 = computeDistanceMap( sphere, params1 );
    auto dm2 = computeDistanceMap( sphere, params2 );
    int count = 0;
    for( int x = 0; x < dm1.resX(); x++ )
    {
        for( int y = 0; y < dm1.resY(); y++ )
        {
            auto v1 = dm1.get( x, y );
            auto v2 = dm2.get( x, y );
            EXPECT_TRUE( bool( v1 ) == bool( v2 ) );
            if ( v1 && v2 && fabs( ( *v1 ) - ( *v2 ) ) > 1e-5 )
            {
                count++;
            }
        }
    }
    EXPECT_EQ( count, 0 );
}

TEST( MRMesh, DistanceMapWatertight )
{
    float pixelSize = 0.1f;
    Mesh sphere = makeUVSphere( 1, 13, 17 );
    AffineXf3f xf = AffineXf3f(
        Matrix3f(
            Vector3f( 1.f, 0.f, 0.f ),
            Vector3f( 0.f, 1.f, 0.f ),
            Vector3f( 0.f, 0.f, 1.f ) ),
        Vector3f( -1.f, -1.f, -1.f )
    );

    MeshToDistanceMapParams params( xf, Vector2f{ pixelSize,pixelSize }, Vector2i{ 10,10 } );

    auto dm1 = computeDistanceMapD( sphere, params );
    auto meshFromDm1 = distanceMapToMesh( dm1, params );

    auto dm2 = computeDistanceMapD( meshFromDm1, params );
    auto meshFromDm2 = distanceMapToMesh( dm2, params );
    int count = 0;

    EXPECT_EQ( dm1.resX(), dm2.resX() );
    EXPECT_EQ( dm1.resY(), dm2.resY() );

    for( int x = 0; x < dm2.resX(); x++ )
    {
        for( int y = 0; y < dm2.resY(); y++ )
        {
            const auto v1 = dm1.get( x, y );
            const auto v2 = dm2.get( x, y );
            if ( v1 && v2 )
            {
                if ( std::abs( (*v1) - (*v2) ) > 1e-6 )
                {
                    count++;
                }
            }
            else
            {
                if ( dm1.isValid( x, y ) || dm2.isValid( x, y ) )
                {
                    count++;
                }
            }
        }
    }
    // Number of distance map pixels with big differ between dm1 and dm2
    const int numberOfMisses = 25;
    EXPECT_EQ( count, numberOfMisses ); //for watertight

    //debug line
    const bool saveMesh = false;
    if ( saveMesh )
    {
        MeshSave::toMrmesh( sphere, std::filesystem::path( "c:/temp/sphere.mrmesh" ) );
        MeshSave::toMrmesh( meshFromDm1, std::filesystem::path( "c:/temp/dm1.mrmesh" ) );
        MeshSave::toMrmesh( meshFromDm2, std::filesystem::path( "c:/temp/dm2.mrmesh" ) );
    }
}

TEST( MRMesh, DistanceMapCompare )
{
    MR_TIMER

    float pixSize = 0.1f;
    Mesh mesh = makeUVSphere( 1, 100, 100 );
    auto meshBox = mesh.computeBoundingBox();

    AffineXf3f xf;
    xf.b = Vector3f( meshBox.min.x, meshBox.min.y, -1.f );
    xf.A.x = Vector3f( 1.f, 0.f, 0.f );
    xf.A.y = Vector3f( 0.f, 1.f, 0.f );
    xf.A.z = Vector3f( 0.f, 0.f, 1.f );

    // build our tree
    PointOnFace pofRes;
    {
        MR_NAMED_TIMER( "intersectRay" );
        if ( auto mir = rayMeshIntersect( mesh, { xf.b, xf.A.z } ) )
            pofRes = mir->proj;
    }

    MeshToDistanceMapParams params( xf, Vector2f{ pixSize,pixSize }, Vector2i{ 10,10 } );
    auto resD = computeDistanceMapD( mesh, params );
    auto resF = computeDistanceMap( mesh, params );

    //debug line
    const bool saveMesh = false;
    if ( saveMesh )
    {
        MeshSave::toMrmesh( distanceMapToMesh( resD, params ), std::filesystem::path( "c:/temp/Double.mrmesh" ) );
        MeshSave::toMrmesh( distanceMapToMesh( resF, params ), std::filesystem::path( "c:/temp/Float.mrmesh" ) );
    }
}

TEST( MRMesh, DistanceMapNegativeValue )
{
    float pixSize = 0.1f;
    Mesh mesh = makeUVSphere( 1, 100, 100 );
    auto meshBox = mesh.computeBoundingBox();

    AffineXf3f xf;
    xf.b = Vector3f( meshBox.min.x, meshBox.min.y, 1.f );
    xf.A.x = Vector3f( 1.f, 0.f, 0.f );
    xf.A.y = Vector3f( 0.f, 1.f, 0.f );
    xf.A.z = Vector3f( 0.f, 0.f, 1.f );

    AffineXf3f xf2 = xf;
    xf2.b.z = 0.f;

    MeshToDistanceMapParams params( xf, Vector2f{ pixSize,pixSize }, Vector2i{ 10,10 } );
    params.allowNegativeValues = true;
    MeshToDistanceMapParams params2( xf2, Vector2f{ pixSize,pixSize }, Vector2i{ 10,10 } );
    params2.allowNegativeValues = true;
    const auto dm = computeDistanceMap( mesh, params );
    const auto dm2 = computeDistanceMap( mesh, params2 );

    EXPECT_EQ( dm.resX(), dm2.resX() );
    EXPECT_EQ( dm.resY(), dm2.resY() );

    int count = 0;
    for ( int x = 0; x < dm2.resX(); x++ )
    {
        for ( int y = 0; y < dm2.resY(); y++ )
        {
            EXPECT_TRUE( dm2.isValid( x, y ) == dm.isValid( x, y ) );
            const auto v1 = dm.get( x, y );
            const auto v2 = dm2.get( x, y );
            if ( v1 && v2 )
            {
                if ( std::abs( ( *v1 ) - ( *v2 ) + 1.f ) > 1e-6 )
                {
                    count++;
                }
            }
        }
    }
    const int numberOfMisses = 0;
    EXPECT_EQ( count, numberOfMisses );

    //debug line
    const bool saveMesh = true;
    if ( saveMesh )
    {
        MeshSave::toMrmesh( distanceMapToMesh( dm, params ), std::filesystem::path( "c:/temp/dm.mrmesh" ) );
        MeshSave::toMrmesh( distanceMapToMesh( dm2, params2 ), std::filesystem::path( "c:/temp/dm2.mrmesh" ) );
    }
}

TEST( MRMesh, DistanceMapOffsetMap )
{
    Contours2f conts;
    conts.push_back( { {0.0f,0.0f},{0.0f,300.0f},{300.0f,300.0f},{300.0f,0.0f},{0.0f,0.0f} } );
    Polyline2 p2( conts );

    float pixelSize = 1.0f;
    float offset = 150.0f;
    auto params = ContourToDistanceMapParams( pixelSize, conts, offset, true );
    Vector<float, UndirectedEdgeId> perEdgeOffset( p2.topology.undirectedEdgeSize() );
    int offsetCounter = 0;
    for ( auto& off : perEdgeOffset )
        off = 20.0f * ( offsetCounter++ );

    
    ContoursDistanceMapOffset offsetParams{ perEdgeOffset, ContoursDistanceMapOffset::OffsetType::Shell };
    ContoursDistanceMapOptions options;
    options.offsetParameters = &offsetParams;
    auto distMap = distanceMapFromContours( p2, params, options );
    //saveDistanceMapToImage( distMap, "org.png" );

    int counter = 0;
    for ( int i = 0; i < distMap.resX() * distMap.resY(); ++i )
    {
        if ( distMap.getValue( i ) < 0.0f )
            ++counter;
    }
    // this is correct number of negative pixels (area inside offset)
    ASSERT_EQ( counter, 80275 );
}

TEST( MRMesh, DistanceMapInterpolation )
{
    DistanceMap dm( 2, 2 );
    dm.set( 0, 0, 2.f );
    dm.set( 0, 1, 3.f );
    dm.set( 1, 0, 4.f );
    dm.set( 1, 1, 5.f );
    {
        auto val = dm.getInterpolated( 1.f, 1.f );
        EXPECT_NEAR( *val, 3.5f, 1e-6 );
    }
    {
        auto val = dm.getInterpolated( 0.f, 0.f );
        EXPECT_NEAR( *val, 2.0f, 1e-6 );
    }
    {
        auto val = dm.getInterpolated( 0.8f, 1.3f );
        EXPECT_NEAR( *val, 3.4f, 1e-6 );
    }
    dm.unset( 1, 0 );
    {
        auto val = dm.getInterpolated( 0.5f, 0.5f );
        EXPECT_TRUE( !val );
    }
}

}
