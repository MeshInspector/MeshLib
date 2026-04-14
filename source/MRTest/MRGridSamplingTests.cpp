#include <MRMesh/MRGridSampling.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, GridSampling )
{
    auto sphereMesh = makeUVSphere();
    auto numVerts = sphereMesh.topology.numValidVerts();
    auto samples = verticesGridSampling( sphereMesh, 0.5f );
    auto sampleCount = samples->count();
    EXPECT_LE( sampleCount, numVerts );
}

} // namespace MR
