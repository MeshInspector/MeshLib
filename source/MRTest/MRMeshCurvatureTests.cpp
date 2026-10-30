#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRRingIterator.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, DiscreteCurvatures )
{
    auto sphere = makeUVSphere( 1.0f, 32, 32 );
    auto torus = makeTorus( 2.0f, 1.0f, 32, 32 );
    for ( const Mesh * mesh : { &sphere, &torus } )
    {
        const auto mean = mesh->discreteMeanCurvatures();
        const auto gauss = mesh->discreteGaussianCurvatures();
        ASSERT_EQ( mean.size(), mesh->topology.vertSize() );
        ASSERT_EQ( gauss.size(), mesh->topology.vertSize() );
        double sumGauss = 0;
        for ( auto v : mesh->topology.getValidVerts() )
        {
            EXPECT_EQ( mean[v], mesh->discreteMeanCurvature( v ) );
            EXPECT_EQ( gauss[v], mesh->discreteGaussianCurvature( v ) );
            sumGauss += gauss[v];
        }
        // Gauss-Bonnet: 2*pi*(Euler characteristic), which is 2 for the sphere and 0 for the torus
        EXPECT_NEAR( sumGauss, mesh == &sphere ? 4 * PI : 0, 1e-3 );
    }

    // only the vertices of the region are computed
    VertBitSet region( sphere.topology.vertSize() );
    for ( int i = 0; i < (int)region.size(); i += 2 )
        region.set( VertId( i ) );
    const auto meanInRegion = sphere.discreteMeanCurvatures( &region );
    const auto gaussInRegion = sphere.discreteGaussianCurvatures( &region );
    const auto mean = sphere.discreteMeanCurvatures();
    const auto gauss = sphere.discreteGaussianCurvatures();
    for ( auto v : sphere.topology.getValidVerts() )
    {
        EXPECT_EQ( meanInRegion[v], region.test( v ) ? mean[v] : 0.0f );
        EXPECT_EQ( gaussInRegion[v], region.test( v ) ? gauss[v] : 0.0f );
    }

    // invalid vertices get 0
    const VertId v0 = sphere.topology.getValidVerts().find_first();
    FaceBitSet ring;
    for ( EdgeId e : orgRing( sphere.topology, v0 ) )
        if ( auto f = sphere.topology.left( e ) )
            ring.autoResizeSet( f );
    sphere.deleteFaces( ring );
    ASSERT_FALSE( sphere.topology.hasVert( v0 ) );
    EXPECT_EQ( sphere.discreteMeanCurvatures()[v0], 0.0f );
    EXPECT_EQ( sphere.discreteGaussianCurvatures()[v0], 0.0f );
}

} //namespace MR
