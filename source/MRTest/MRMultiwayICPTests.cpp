#include <MRMesh/MRMultiwayICP.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRGTest.h>
#include <iostream>

namespace MR
{

TEST( MRMesh, MultiwayICPTorus )
{
    // all objects have same shape but different transformations
    const auto torus = makeTorus( 2.5f, 0.7f, 48, 48 );

    auto axis = Vector3f( 1, 0, 0 );
    auto trans = Vector3f( 0, 0.2f, 0.105f );

    const auto xf = AffineXf3f( Matrix3f::rotation( axis, 0.2f ), trans );

    auto run = [&] ( ICPMethod method, int maxGroupSize, float eps )
    {
        ICPObjects objs;
        objs.push_back( { torus, xf } );
        objs.push_back( { torus, xf.inverse() } );
        objs.push_back( { torus, AffineXf3f{} } );
        MultiwayICP icp( objs, MultiwayICPSamplingParameters{ .maxGroupSize = maxGroupSize } );

        ICPProperties props
        {
            .method = method,
            .iterLimit = 30
        };
        icp.setParams( props );
        auto newXfs = icp.calculateTransformations();
        EXPECT_EQ( newXfs[ObjId( 2 )], AffineXf3f{} );
        std::cout << icp.getStatusInfo() << '\n';

        auto newXf0 = newXfs[ObjId( 0 )];
        EXPECT_LT( ( newXf0.A - Matrix3f::identity() ).norm(), eps );
        EXPECT_LT( newXf0.b.length(), eps );

        auto newXf1 = newXfs[ObjId( 1 )];
        EXPECT_LT( ( newXf1.A - Matrix3f::identity() ).norm(), eps );
        EXPECT_LT( newXf1.b.length(), eps );
    };

    for ( int maxGroupSize : { 0, 1 } )
    {
        std::cout << "running Point-to-Plane method, maxGroupSize=" << maxGroupSize << "\n";
        run( ICPMethod::PointToPlane, maxGroupSize, 1e-6f );

        std::cout << "running Point-to-Point method, maxGroupSize=" << maxGroupSize << "\n";
        run( ICPMethod::PointToPoint, maxGroupSize, 1e-3f );

        std::cout << "running Combined method, maxGroupSize=" << maxGroupSize << "\n";
        run( ICPMethod::Combined, maxGroupSize, 1e-6f );
    }
}

} //namespace MR
