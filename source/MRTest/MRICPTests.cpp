#include <MRMesh/MRICP.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRGTest.h>
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

} //namespace MR
