#include <MRMesh/MRIterativeSampling.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshToPointCloud.h>
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, IterativeSampling )
{
    auto cloud = meshToPointCloud( makeTorus() );
    auto numSamples = (int)cloud.validPoints.count() / 2;
    auto optSamples = pointIterativeSampling( cloud, numSamples );
    EXPECT_EQ( numSamples, optSamples->count() );
}

} //namespace MR
