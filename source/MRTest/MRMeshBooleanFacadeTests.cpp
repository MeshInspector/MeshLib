#include <MRMesh/MRMeshBooleanFacade.h>
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, MeshBooleanFacade )
{
    Mesh gingivaCopy = makeCube();
    Mesh combinedTooth = makeUVSphere( 1.1f );
    MeshMeshConverter convert;

    auto gingivaGrid = convert( gingivaCopy );
    auto toothGrid = convert( combinedTooth );
    toothGrid -= gingivaGrid;
    auto tooth = std::make_shared<MR::Mesh>( convert( toothGrid ) );
}

} //namespace MR
