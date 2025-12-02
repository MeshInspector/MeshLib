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
    auto torusRef = makeTorus( 2.5f, 0.7f, 48, 48 );
    auto torusMove = torusRef;

    auto axis = Vector3f( 1, 0, 0 );
    auto trans = Vector3f( 0, 0.2f, 0.105f );

    auto xf = AffineXf3f( Matrix3f::rotation( axis, 0.2f ), trans );

    ICPObjects objs;
    objs.emplace_back( torusMove, xf );
    objs.emplace_back( torusRef, AffineXf3f{} );
    MultiwayICP icp( objs, MultiwayICPSamplingParameters{} );

    ICPProperties props
    {
        .iterLimit = 20
    };
    icp.setParams( props );
    auto newXfs = icp.calculateTransformations();
    EXPECT_EQ( newXfs[ObjId( 1 )], AffineXf3f{} );
    std::cout << icp.getStatusInfo() << '\n';

    auto newXf = newXfs[ObjId( 0 )];
    constexpr float eps = 1e-6f;
    EXPECT_NEAR( ( newXf.A - Matrix3f::identity() ).norm(), 0., eps );
    EXPECT_NEAR( newXf.b.length(), 0., eps );
}

} //namespace MR
