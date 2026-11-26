#include <MRMesh/MRMeshAnalyticRegions.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRVector2.h>
#include <gtest/gtest.h>
#include <array>
#include <cmath>

namespace MR
{

namespace
{

/// the numbers of regions on planes, cylinders, cones, spheres and tori, with the check that every valid face is in a region
std::array<int, 5> countRegions( const Mesh& mesh, const MeshAnalyticRegions& regions )
{
    std::array<int, 5> res{};
    for ( const auto& s : regions.surfaces )
        ++res[s.index()];
    Vector<int, RegionId> numFaces( regions.surfaces.size(), 0 );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        const auto r = regions.face2region[f];
        EXPECT_TRUE( r.valid() && r < regions.surfaces.size() );
        if ( r.valid() && r < regions.surfaces.size() )
            ++numFaces[r];
    }
    for ( int n : numFaces )
        EXPECT_GT( n, 0 );
    return res;
}

} //anonymous namespace

TEST( MRMesh, FindAnalyticRegions )
{
    const auto cube = makeCube();
    auto res = findAnalyticRegions( cube ).value();
    EXPECT_EQ( countRegions( cube, res ), ( std::array<int, 5>{ 6, 0, 0, 0, 0 } ) );
    EXPECT_GT( res.tolerance, 0 );
    for ( const auto& s : res.surfaces )
    {
        // the planes of the faces of the unit cube centered in the origin
        const auto& plane = std::get<Plane3d>( s );
        EXPECT_NEAR( plane.n.length(), 1, 1e-12 );
        EXPECT_NEAR( std::abs( plane.d ), 0.5, 1e-6 );
        EXPECT_NEAR( std::max( { std::abs( plane.n.x ), std::abs( plane.n.y ), std::abs( plane.n.z ) } ), 1, 1e-12 );
    }

    const auto cylMesh = makeCylinder( 1.0f, 3.0f, 32 );
    res = findAnalyticRegions( cylMesh ).value();
    EXPECT_EQ( countRegions( cylMesh, res ), ( std::array<int, 5>{ 2, 1, 0, 0, 0 } ) );
    for ( const auto& s : res.surfaces )
        if ( const auto* cyl = std::get_if<Cylinder3d>( &s ) )
        {
            EXPECT_NEAR( cyl->radius, 1, 1e-5 );
            EXPECT_NEAR( cyl->length, 3, 1e-5 );
            EXPECT_NEAR( std::abs( cyl->direction().z ), 1, 1e-9 );
            EXPECT_NEAR( ( cyl->center() - Vector3d( 0, 0, 1.5 ) ).length(), 0, 1e-5 );
        }

    // as if loaded from a text file with 4 decimal digits: the default tolerance is larger
    auto rounded = cylMesh;
    for ( auto& p : rounded.points )
        for ( int i = 0; i < 3; ++i )
            p[i] = std::round( p[i] * 1e4f ) / 1e4f;
    const auto roundedRes = findAnalyticRegions( rounded ).value();
    EXPECT_EQ( countRegions( rounded, roundedRes ), ( std::array<int, 5>{ 2, 1, 0, 0, 0 } ) );
    EXPECT_GT( roundedRes.tolerance, res.tolerance );

    const auto sphereMesh = makeUVSphere( 1.0f, 32, 32 );
    res = findAnalyticRegions( sphereMesh ).value();
    EXPECT_EQ( countRegions( sphereMesh, res ), ( std::array<int, 5>{ 0, 0, 0, 1, 0 } ) );
    EXPECT_NEAR( std::get<Sphere3d>( res.surfaces.front() ).radius, 1, 1e-5 );
    EXPECT_NEAR( std::get<Sphere3d>( res.surfaces.front() ).center.length(), 0, 1e-5 );

    const auto torusMesh = makeTorus( 2.0f, 0.5f, 32, 32 );
    res = findAnalyticRegions( torusMesh ).value();
    // the regions are not united: a ring of quadrangles of the torus is also on a cone exactly, so a part of the torus is found as cones
    const auto torusCounts = countRegions( torusMesh, res );
    EXPECT_EQ( torusCounts[0] + torusCounts[1] + torusCounts[3], 0 );
    EXPECT_GT( torusCounts[4], 0 );
    for ( const auto& s : res.surfaces )
    {
        if ( const auto* torus = std::get_if<Torus3d>( &s ) )
        {
            EXPECT_NEAR( torus->majorRadius, 2, 1e-5 );
            EXPECT_NEAR( torus->minorRadius, 0.5, 1e-5 );
            EXPECT_NEAR( std::abs( torus->axis.z ), 1, 1e-9 );
            EXPECT_NEAR( torus->center.length(), 0, 1e-5 );
        }
        else
        {
            // coaxial with the torus
            const auto& cone = std::get<Cone3d>( s );
            EXPECT_NEAR( std::abs( cone.direction().z ), 1, 1e-9 );
            EXPECT_NEAR( Vector2d( cone.apex().x, cone.apex().y ).length(), 0, 1e-3 ); // the apex of a narrow cone is far from its ring
        }
    }

    // the edges of one triangle of the cube given as boundaries separate it from its neighbour in the same plane
    UndirectedEdgeBitSet boundaries( cube.topology.undirectedEdgeSize() );
    const FaceId f0 = cube.topology.getValidFaces().find_first();
    for ( EdgeId e : leftRing( cube.topology, f0 ) )
        boundaries.set( e.undirected() );
    res = findAnalyticRegions( cube, { .faceBoundaries = &boundaries } ).value();
    EXPECT_EQ( countRegions( cube, res ), ( std::array<int, 5>{ 7, 0, 0, 0, 0 } ) );

    // errors: canceled, no triangles
    EXPECT_FALSE( findAnalyticRegions( cylMesh, {}, []( float ) { return false; } ).has_value() );
    EXPECT_FALSE( findAnalyticRegions( Mesh{} ).has_value() );
}

} //namespace MR
