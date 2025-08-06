#include "TestMacros.h"
#include "TestFunctions.h"
#include <MRMeshC/MRMeshTopology.h>
#include <MRMeshC/MRCube.h>
#include <MRMeshC/MRMesh.h>
#include <stdio.h>


void testMrMeshTopologyPack(void)
{
    MRMesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    mrMeshTopologyPack(topology);
    mrMeshFree(mesh);
}

void testMrMeshTopologyGetValidVerts(void)
{
    MRMesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    const MRVertBitSet* verts = mrMeshTopologyGetValidVerts(topology);
    TEST_ASSERT(verts != NULL);
    mrMeshFree(mesh);
}

void testMrMeshTopologyGetValidFaces(void)
{
    MRMesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    const MRFaceBitSet* faces = mrMeshTopologyGetValidFaces(topology);
    TEST_ASSERT(faces != NULL);
    mrMeshFree(mesh);
}

void testMrMeshTopologyFindHoleRepresentiveEdges( void ) {
    MRMesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    MREdgePath* holes = mrMeshTopologyFindHoleRepresentiveEdges(topology);
    mrEdgePathFree(holes);
    mrMeshFree(mesh);
}

void testMrMeshTopologyGetLeftTriVerts( void ) {
    MRMesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);

    MREdgeId edge = { 0 };
    MRVertId v0, v1, v2;
    mrMeshTopologyGetLeftTriVerts(topology, edge, &v0, &v1, &v2);
    mrMeshFree(mesh);
}

void testMrMeshTopologyFindNumHoles( void ) {
    MRMesh* mesh = createCube();
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    TEST_ASSERT(mesh != NULL);

    int numHoles = mrMeshTopologyFindNumHoles(topology, NULL);
    TEST_ASSERT_INT_EQUAL(0, numHoles);
    mrMeshFree(mesh);
}

void testMrMeshTopologyFaceSize(void) {
    MRMesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);

    size_t faceSize = mrMeshTopologyFaceSize(topology);
    TEST_ASSERT_INT_EQUAL(12, (int)faceSize);
    mrMeshFree(mesh);
}

void testMrMeshTopologyGetTriangulation(void) {
    MRMesh* mesh = createCube();
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);

    MRTriangulation* triangulation = mrMeshTopologyGetTriangulation(topology);
    TEST_ASSERT(triangulation != NULL);
    TEST_ASSERT_INT_EQUAL(12, (int)triangulation->size);
    mrTriangulationFree(triangulation);
    mrMeshFree(mesh);
}
