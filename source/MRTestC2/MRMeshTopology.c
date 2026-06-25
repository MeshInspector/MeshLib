#include "TestMacros.h"
#include "TestFunctions.h"
#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRVector.h>
#include <MRCMisc/std_vector_MR_EdgeId.h>
#include <stdio.h>


void testMrMeshTopologyPack(void)
{
    MR_Mesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MR_MeshTopology* topology = MR_Mesh_GetMutable_topology(mesh);
    MR_MeshTopology_pack_4( topology, NULL, NULL, NULL, NULL);
    MR_Mesh_Destroy(mesh);
}

void testMrMeshTopologyGetValidVerts(void)
{
    MR_Mesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MR_MeshTopology* topology = MR_Mesh_GetMutable_topology(mesh);
    const MR_VertBitSet* verts = MR_MeshTopology_getValidVerts(topology);
    TEST_ASSERT(verts != NULL);
    MR_Mesh_Destroy(mesh);
}

void testMrMeshTopologyGetValidFaces(void)
{
    MR_Mesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MR_MeshTopology* topology = MR_Mesh_GetMutable_topology(mesh);
    const MR_FaceBitSet* faces = MR_MeshTopology_getValidFaces(topology);
    TEST_ASSERT(faces != NULL);
    MR_Mesh_Destroy(mesh);
}

void testMrMeshTopologyFindHoleRepresentiveEdges( void ) {
    MR_Mesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MR_MeshTopology* topology = MR_Mesh_GetMutable_topology(mesh);
    MR_std_vector_MR_EdgeId* holes = MR_MeshTopology_findHoleRepresentiveEdges(topology, NULL);
    MR_std_vector_MR_EdgeId_Destroy(holes);
    MR_Mesh_Destroy(mesh);
}

void testMrMeshTopologyGetLeftTriVerts( void ) {
    MR_Mesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MR_MeshTopology* topology = MR_Mesh_GetMutable_topology(mesh);

    MR_EdgeId edge = { 0 };
    MR_VertId v0, v1, v2;
    MR_MeshTopology_getLeftTriVerts_4(topology, edge, &v0, &v1, &v2);
    MR_Mesh_Destroy(mesh);
}

void testMrMeshTopologyFindNumHoles( void ) {
    MR_Mesh* mesh = createCube();
    MR_MeshTopology* topology = MR_Mesh_GetMutable_topology(mesh);
    TEST_ASSERT(mesh != NULL);

    int numHoles = MR_MeshTopology_findNumHoles(topology, NULL);
    TEST_ASSERT_INT_EQUAL(0, numHoles);
    MR_Mesh_Destroy(mesh);
}

void testMrMeshTopologyFaceSize(void) {
    MR_Mesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MR_MeshTopology* topology = MR_Mesh_GetMutable_topology(mesh);

    size_t faceSize = MR_MeshTopology_faceSize(topology);
    TEST_ASSERT_INT_EQUAL(12, (int)faceSize);
    MR_Mesh_Destroy(mesh);
}

void testMrMeshTopologyGetTriangulation(void) {
    MR_Mesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MR_MeshTopology* topology = MR_Mesh_GetMutable_topology(mesh);

    MR_Triangulation* triangulation = MR_MeshTopology_getTriangulation(topology);
    TEST_ASSERT(triangulation != NULL);
    TEST_ASSERT_INT_EQUAL(12, (int)MR_Triangulation_size(triangulation));
    MR_Triangulation_Destroy(triangulation);
    MR_Mesh_Destroy(mesh);
}
