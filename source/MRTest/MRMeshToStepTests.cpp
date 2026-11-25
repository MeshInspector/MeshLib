#include <MRMesh/MRMeshToStep.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRTorus.h>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>

namespace MR
{

TEST( MRMesh, MeshToStep )
{
    auto convert = []( const Mesh& mesh )
    {
        std::ostringstream out;
        auto res = saveMeshAsStep( mesh, out );
        EXPECT_TRUE( res.has_value() );
        EXPECT_NE( out.str().find( "END-ISO-10303-21;" ), std::string::npos );
        EXPECT_NE( out.str().find( "CLOSED_SHELL" ), std::string::npos );
        return res ? *res : MeshToStepStats{};
    };

    auto stats = convert( makeCube() );
    EXPECT_EQ( stats.planes, 6 );
    EXPECT_EQ( stats.cylinders + stats.cones + stats.spheres + stats.tori + stats.bsplines, 0 );

    stats = convert( makeCylinder( 1.0f, 3.0f, 32 ) );
    EXPECT_EQ( stats.planes, 2 );
    EXPECT_EQ( stats.cylinders, 1 );
    EXPECT_EQ( stats.cones + stats.spheres + stats.tori + stats.bsplines, 0 );

    // as if loaded from a text file with 4 decimal digits
    auto rounded = makeCylinder( 1.0f, 3.0f, 32 );
    for ( auto& p : rounded.points )
        for ( int i = 0; i < 3; ++i )
            p[i] = std::round( p[i] * 1e4f ) / 1e4f;
    stats = convert( rounded );
    EXPECT_EQ( stats.planes, 2 );
    EXPECT_EQ( stats.cylinders, 1 );
    EXPECT_EQ( stats.cones + stats.spheres + stats.tori + stats.bsplines, 0 );

    stats = convert( makeUVSphere( 1.0f, 32, 32 ) );
    EXPECT_EQ( stats.spheres, 1 );
    EXPECT_EQ( stats.planes + stats.cylinders + stats.cones + stats.tori + stats.bsplines, 0 );

    stats = convert( makeTorus( 2.0f, 0.5f, 32, 32 ) );
    EXPECT_EQ( stats.tori, 1 );
    EXPECT_EQ( stats.planes + stats.cylinders + stats.cones + stats.spheres + stats.bsplines, 0 );
}

} //namespace MR
