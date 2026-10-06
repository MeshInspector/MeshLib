#include <MRMesh/MRICP.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRAffineXf3.h>
#include <gtest/gtest.h>
#include <iostream>

namespace MR
{

TEST( MRMesh, ICPTorus )
{
    auto torusRef = makeTorus( 2.5f, 0.7f, 48, 48 );
    auto torusMove = torusRef;

    auto axis = Vector3f( 1, 0, 0 );
    auto trans = Vector3f( 0, 0.2f, 0.105f );

    const auto xf = AffineXf3f( Matrix3f::rotation( axis, 0.2f ), trans );

    auto run = [&] ( ICPMethod method, float eps )
    {
        ICP icp(
            torusMove, torusRef, xf, AffineXf3f(), torusMove.topology.getValidVerts(), torusRef.topology.getValidVerts()
        );
        ICPProperties props
        {
            .method = method,
            .iterLimit = 20
        };
        icp.setParams( props );
        auto newXf = icp.calculateTransformation();
        std::cout << icp.getStatusInfo() << '\n';

        EXPECT_LT( ( newXf.A - Matrix3f::identity() ).norm(), eps );
        EXPECT_LT( newXf.b.length(), eps );
    };

    std::cout << "running Point-to-Plane method\n";
    run( ICPMethod::PointToPlane, 1e-6f );

    std::cout << "running Point-to-Point method\n";
    run( ICPMethod::PointToPoint, 1e-3f );

    std::cout << "running Combined method\n";
    run( ICPMethod::Combined, 1e-6f );
}

TEST( MRMesh, ICPTorusWeightedSamples )
{
    const auto torus = makeTorus( 2.5f, 0.7f, 48, 48 );
    const auto xf = AffineXf3f( Matrix3f::rotation( Vector3f( 1, 0, 0 ), 0.2f ), Vector3f( 0, 0.2f, 0.105f ) );
    const ICPProperties props{ .method = ICPMethod::PointToPlane, .iterLimit = 20 };

    // with dblArea weights the result must match the constructor taking VertBitSet
    const auto & verts = torus.topology.getValidVerts();
    std::vector<WeightedVertexf> samples;
    for ( auto v : verts )
        samples.push_back( { v, torus.dblArea( v ) } );

    ICP icpBits( torus, torus, xf, AffineXf3f(), verts, verts );
    icpBits.setParams( props );
    const auto xfBits = icpBits.calculateTransformation();

    ICP icp( torus, torus, xf, AffineXf3f(), samples, samples );
    icp.setParams( props );
    const auto newXf = icp.calculateTransformation();
    EXPECT_EQ( newXf, xfBits );
    EXPECT_LT( ( newXf.A - Matrix3f::identity() ).norm(), 1e-6f );
    EXPECT_LT( newXf.b.length(), 1e-6f );

    // the weights are not changed by the algorithm
    const auto & pairs = icp.getFlt2RefPairs().vec;
    ASSERT_EQ( pairs.size(), samples.size() );
    for ( size_t i = 0; i < samples.size(); ++i )
    {
        EXPECT_EQ( pairs[i].srcVertId, samples[i].v );
        EXPECT_EQ( pairs[i].weight, samples[i].weight );
    }
}

} //namespace MR
