#include <MRMesh/MRMeshToStep.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshMath.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMeshSubdivide.h>
#include <MRMesh/MRTorus.h>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>

namespace MR
{

TEST( MRMesh, MeshToStep )
{
    // the faces of the result are separated by given edges, by default by the sharp edges of the mesh
    auto convert = []( const Mesh& mesh, const UndirectedEdgeBitSet* boundaries = nullptr )
    {
        std::ostringstream out;
        auto res = saveMeshAsStep( mesh, boundaries ? *boundaries : findCreaseEdges( mesh.topology, mesh.points, 0.5f ), out );
        EXPECT_TRUE( res.has_value() );
        EXPECT_NE( out.str().find( "END-ISO-10303-21;" ), std::string::npos );
        EXPECT_NE( out.str().find( "CLOSED_SHELL" ), std::string::npos );
        return res ? *res : MeshToStepStats{};
    };

    auto stats = convert( makeCube() );
    EXPECT_EQ( stats.planes, 6 );
    EXPECT_EQ( stats.cylinders + stats.cones + stats.spheres + stats.tori + stats.bsplines, 0 );
    EXPECT_LT( stats.maxDeviation, 1e-5f );

    stats = convert( makeCylinder( 1.0f, 3.0f, 32 ) );
    EXPECT_EQ( stats.planes, 2 );
    EXPECT_EQ( stats.cylinders, 1 );
    EXPECT_EQ( stats.cones + stats.spheres + stats.tori + stats.bsplines, 0 );
    EXPECT_LT( stats.maxDeviation, 1e-5f );

    // all vertices are noisy, as in a scanned mesh
    auto addNoise = []( Mesh& mesh )
    {
        for ( VertId v : mesh.topology.getValidVerts() )
            mesh.points[v] += 0.01f * Vector3f( std::sin( 12.9898f * float( v ) ), std::sin( 78.233f * float( v ) ), std::sin( 37.719f * float( v ) ) );
    };
    auto noisyCube = makeCube();
    subdivideMesh( noisyCube, { .maxEdgeLen = 0.1f, .maxEdgeSplits = 100000, .maxDeviationAfterFlip = 0 } );
    const auto cubeEdges = findCreaseEdges( noisyCube.topology, noisyCube.points, 0.5f );
    addNoise( noisyCube );
    stats = convert( noisyCube, &cubeEdges );
    EXPECT_EQ( stats.planes, 6 );
    EXPECT_EQ( stats.cylinders + stats.cones + stats.spheres + stats.tori + stats.bsplines, 0 );
    EXPECT_LT( stats.maxDeviation, 0.03f );

    auto noisySphere = makeUVSphere( 1.0f, 32, 32 );
    addNoise( noisySphere );
    const UndirectedEdgeBitSet noEdges;
    stats = convert( noisySphere, &noEdges );
    EXPECT_EQ( stats.spheres, 1 );
    EXPECT_EQ( stats.planes + stats.cylinders + stats.cones + stats.tori + stats.bsplines, 0 );
    EXPECT_LT( stats.maxDeviation, 0.03f );

    stats = convert( makeUVSphere( 1.0f, 32, 32 ) );
    EXPECT_EQ( stats.spheres, 1 );
    EXPECT_EQ( stats.planes + stats.cylinders + stats.cones + stats.tori + stats.bsplines, 0 );

    stats = convert( makeTorus( 2.0f, 0.5f, 32, 32 ) );
    EXPECT_EQ( stats.tori, 1 );
    EXPECT_EQ( stats.planes + stats.cylinders + stats.cones + stats.spheres + stats.bsplines, 0 );
}

} //namespace MR
