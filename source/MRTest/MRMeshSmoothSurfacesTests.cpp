#include <MRMesh/MRMeshSmoothSurfaces.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRFittedBSplineSurface.h>
#include <MRMesh/MRMakePlane.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshComponents.h>
#include <MRMesh/MRMeshSubdivide.h>
#include <MRMesh/MRTorus.h>
#include <gtest/gtest.h>
#include <cmath>

namespace MR
{

TEST( MRMesh, FitSmoothSurfaces )
{
    // regions separated by the sharp edges
    const auto cyl = makeCylinder( 1.0f, 3.0f, 32 );
    const auto [cylRegions, numCylRegions] = MeshComponents::getAllComponentsMapBySharpEdges( cyl );
    EXPECT_EQ( numCylRegions, 3 );
    auto res = fitSmoothSurfaces( cyl, cylRegions );
    ASSERT_TRUE( res.has_value() );
    ASSERT_EQ( res->surfaces.size(), 3 );
    int planes = 0, cylinders = 0;
    for ( const auto& s : res->surfaces )
    {
        EXPECT_LT( s.maxDist, 1e-5f );
        if ( std::holds_alternative<Plane3d>( s.surface ) )
            ++planes;
        else if ( const auto* c = std::get_if<Cylinder3d>( &s.surface ) )
        {
            ++cylinders;
            EXPECT_NEAR( c->radius, 1, 1e-5 );
            EXPECT_NEAR( c->length, 3, 1e-5 );
            EXPECT_NEAR( std::abs( c->direction().z ), 1, 1e-6 );
        }
    }
    EXPECT_EQ( planes, 2 );
    EXPECT_EQ( cylinders, 1 );

    // a single region gets the same surface
    for ( RegionId r( 0 ); r < res->surfaces.endId(); ++r )
    {
        FaceBitSet region( cyl.topology.faceSize() );
        for ( FaceId f : cyl.topology.getValidFaces() )
            region.set( f, cylRegions[f] == r );
        const auto single = fitSmoothSurface( { cyl, &region } );
        ASSERT_TRUE( single.has_value() );
        EXPECT_EQ( *single, res->surfaces[r] );
    }
    const FaceBitSet empty;
    EXPECT_FALSE( fitSmoothSurface( { cyl, &empty } ).has_value() );

    // all vertices are noisy, as in a scanned mesh
    auto sphere = makeUVSphere( 1.0f, 32, 32 );
    for ( VertId v : sphere.topology.getValidVerts() )
        sphere.points[v] += 0.01f * Vector3f( std::sin( 12.9898f * float( v ) ), std::sin( 78.233f * float( v ) ), std::sin( 37.719f * float( v ) ) );
    res = fitSmoothSurfaces( sphere, Face2RegionMap( sphere.topology.faceSize(), RegionId( 0 ) ) );
    ASSERT_TRUE( res.has_value() );
    ASSERT_EQ( res->surfaces.size(), 1 );
    const auto* s = std::get_if<Sphere3d>( &res->surfaces.front().surface );
    ASSERT_TRUE( s );
    EXPECT_NEAR( s->radius, 1, 0.01 );
    EXPECT_LT( s->center.length(), 0.01 );
    EXPECT_LT( res->surfaces.front().rms, 0.02f );

    const auto torusMesh = makeTorus( 2.0f, 0.5f, 32, 32 );
    res = fitSmoothSurfaces( torusMesh, Face2RegionMap( torusMesh.topology.faceSize(), RegionId( 0 ) ) );
    ASSERT_TRUE( res.has_value() );
    const auto* t = std::get_if<Torus3d>( &res->surfaces.front().surface );
    ASSERT_TRUE( t );
    EXPECT_NEAR( t->majorRadius, 2, 0.01 );
    EXPECT_NEAR( t->minorRadius, 0.5, 0.01 );

    // freeform surface: no analytic surface is close, so a B-spline is fitted with given threshold
    auto wave = makePlane();
    subdivideMesh( wave, { .maxEdgeLen = 0.05f, .maxEdgeSplits = 100000 } );
    for ( auto& p : wave.points )
        p.z = 0.1f * std::sin( 4 * p.x ) * std::cos( 3 * p.y );
    const Face2RegionMap waveRegions( wave.topology.faceSize(), RegionId( 0 ) );
    res = fitSmoothSurfaces( wave, waveRegions, { .bsplineThreshold = 1e-3f } );
    ASSERT_TRUE( res.has_value() );
    const auto* bs = std::get_if<std::shared_ptr<const FittedBSplineSurface>>( &res->surfaces.front().surface );
    ASSERT_TRUE( bs && *bs );
    EXPECT_LT( res->surfaces.front().maxDist, 1e-3f );
    EXPECT_NEAR( (*bs)->maxError, res->surfaces.front().maxDist, 1e-6 );

    res = fitSmoothSurfaces( wave, waveRegions, { .bsplineThreshold = 1e-3f, .allowBSplines = false } );
    ASSERT_TRUE( res.has_value() );
    EXPECT_FALSE( std::holds_alternative<std::shared_ptr<const FittedBSplineSurface>>( res->surfaces.front().surface ) );

    // single region: B-spline only with given threshold
    const auto single = fitSmoothSurface( wave, { .bsplineThreshold = 1e-3f } );
    ASSERT_TRUE( single.has_value() );
    EXPECT_TRUE( std::holds_alternative<std::shared_ptr<const FittedBSplineSurface>>( single->surface ) );
    EXPECT_LT( single->maxDist, 1e-3f );
    const auto analytic = fitSmoothSurface( wave );
    ASSERT_TRUE( analytic.has_value() );
    EXPECT_FALSE( std::holds_alternative<std::shared_ptr<const FittedBSplineSurface>>( analytic->surface ) );
}

} //namespace MR
