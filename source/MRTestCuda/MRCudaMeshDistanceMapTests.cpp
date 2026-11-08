#include <MRCuda/MRCudaMeshDistanceMap.h>
#include <MRMesh/MRDistanceMap.h>
#include <MRMesh/MRDistanceMapParams.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTorus.h>

#include <gtest/gtest.h>

using namespace MR;

// the pixels whose rays miss the mesh (e.g. in the hole of the torus) must stay invalid as in the CPU version
TEST( MRCuda, MeshDistanceMap )
{
    const auto torus = makeTorus( 1.0f, 0.3f, 32, 16 );
    const MeshToDistanceMapParams params( Vector3f::plusZ(), Vector2i{ 64, 64 }, torus );

    const auto cpuMap = computeDistanceMap( torus, params );
    const auto gpuMap = Cuda::computeDistanceMap( torus, params );
    ASSERT_TRUE( gpuMap ) << gpuMap.error();
    ASSERT_EQ( gpuMap->numPoints(), cpuMap.numPoints() );

    size_t numValid = 0;
    for ( size_t i = 0; i < cpuMap.numPoints(); ++i )
    {
        const auto cpuVal = cpuMap.get( i );
        const auto gpuVal = gpuMap->get( i );
        ASSERT_EQ( bool( gpuVal ), bool( cpuVal ) ) << "pixel " << i;
        if ( cpuVal )
        {
            ++numValid;
            ASSERT_NEAR( *gpuVal, *cpuVal, 1e-4f ) << "pixel " << i;
        }
    }
    EXPECT_GT( numValid, 0 );
    EXPECT_LT( numValid, cpuMap.numPoints() );
}
