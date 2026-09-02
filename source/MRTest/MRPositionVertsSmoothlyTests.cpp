#include <MRMesh/MRPositionVertsSmoothly.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRRingIterator.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, InterpolateScalarsSmoothly )
{
    const Mesh sphere = makeUVSphere( 1, 16, 16 );
    const auto & topology = sphere.topology;

    // values are fixed near the poles (1 at the top, 0 at the bottom) and interpolated in the middle band
    VertBitSet region( topology.vertSize() );
    VertScalars field( topology.vertSize(), 0 );
    for ( auto v : topology.getValidVerts() )
    {
        const auto z = sphere.points[v].z;
        if ( std::abs( z ) < 0.5f )
            region.set( v );
        else if ( z > 0 )
            field[v] = 1;
    }
    interpolateScalarsSmoothly( topology, field, { .region = &region } );

    for ( auto v : region )
    {
        // every free value is exactly the mean of its neighbors, so it never leaves the range of fixed values
        double sum = 0;
        int cnt = 0;
        for ( auto e : orgRing( topology, v ) )
        {
            sum += field[topology.dest( e )];
            ++cnt;
        }
        EXPECT_NEAR( field[v], sum / cnt, 1e-5f );
        EXPECT_GE( field[v], 0 );
        EXPECT_LE( field[v], 1 );
    }

    // by symmetry the values on the equator are in the middle
    for ( auto v : region )
    {
        if ( std::abs( sphere.points[v].z ) < 1e-5f )
            EXPECT_NEAR( field[v], 0.5f, 1e-5f );
    }
}

} //namespace MR
