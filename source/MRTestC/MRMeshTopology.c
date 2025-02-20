#include "TestMacros.h"
#include <MRMeshC/MRMeshTopology.h>
#include <MRMeshC/MRCube.h>
#include <MRMeshC/MRMesh.h>
#include <stdio.h>


void test_mrMeshTopologyPack(void)
{
    MRVector3f size = mrVector3fDiagonal(1.0f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    mrMeshTopologyPack(topology);
    mrMeshFree(mesh);
}

void test_mrMeshTopologyGetValidVerts(void)
{
    MRVector3f size = mrVector3fDiagonal(1.0f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    const MRVertBitSet* verts = mrMeshTopologyGetValidVerts(topology);
    TEST_ASSERT(verts != NULL);
    mrMeshFree(mesh);
}

void test_mrMeshTopologyGetValidFaces(void)
{
    MRVector3f size = mrVector3fDiagonal(1.0f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    const MRFaceBitSet* faces = mrMeshTopologyGetValidFaces(topology);
    TEST_ASSERT(faces != NULL);
    mrMeshFree(mesh);
}

void test_mrMeshTopologyFindHoleRepresentiveEdges( void ) {
    MRVector3f size = mrVector3fDiagonal(1.0f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    MREdgePath* holes = mrMeshTopologyFindHoleRepresentiveEdges(topology);
    mrEdgePathFree(holes);
    mrMeshFree(mesh);
}

void test_mrMeshTopologyGetLeftTriVerts( void ) {
    MRVector3f size = mrVector3fDiagonal(1.0f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);

    MREdgeId edge = { 0 };
    MRVertId v0, v1, v2;
    mrMeshTopologyGetLeftTriVerts(topology, edge, &v0, &v1, &v2);
    mrMeshFree(mesh);
}

void test_mrMeshTopologyFindNumHoles( void ) {
    MRVector3f size = mrVector3fDiagonal(1.0f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);
    TEST_ASSERT(mesh != NULL);

    int numHoles = mrMeshTopologyFindNumHoles(topology, NULL);
    TEST_ASSERT_INT_EQUAL(0, numHoles);
    mrMeshFree(mesh);
}

void test_mrMeshTopologyFaceSize(void) {
    MRVector3f size = mrVector3fDiagonal(1.0f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);

    size_t faceSize = mrMeshTopologyFaceSize(topology);
    TEST_ASSERT_INT_EQUAL(12, (int)faceSize);
    mrMeshFree(mesh);
}

void test_mrMeshTopologyGetTriangulation(void) {
    MRVector3f size = mrVector3fDiagonal(1.0f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    TEST_ASSERT(mesh != NULL);
    MRMeshTopology* topology = mrMeshTopologyRef(mesh);

    MRTriangulation* triangulation = mrMeshTopologyGetTriangulation(topology);
    TEST_ASSERT(triangulation != NULL);
    TEST_ASSERT_INT_EQUAL(12, (int)triangulation->size);
    mrTriangulationFree(triangulation);
    mrMeshFree(mesh);
}