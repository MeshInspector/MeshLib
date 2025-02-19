#include "TestMacros.h"
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRCube.h>
#include <MRMeshC/MRVector3.h>
#include <MRMeshC/MRBox.h>
#include <MRMeshC/MRAffineXf.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MROffset.h>
#include <MRMeshC/MRString.h>


void testOffsetMesh(void)
{
    MRString* errorString = NULL;
    MRVector3f size = mrVector3fDiagonal(1.f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    MRMeshPart inputMeshPart = { .mesh = mesh, .region = NULL };
    MROffsetParameters params = mrOffsetParametersNew();
    params.voxelSize = mrSuggestVoxelSize(inputMeshPart, 10000.f);
    float offset = 0.1f;
    MRMesh* outputMesh = mrOffsetMesh(inputMeshPart, offset, &params, &errorString);
    TEST_ASSERT(errorString == NULL);
    mrMeshFree(outputMesh);
    mrMeshFree(mesh);
}

void testDoubleOffsetMesh(void)
{
    MRString* errorString = NULL;
    MRVector3f size = mrVector3fDiagonal(1.f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    MRMeshPart inputMeshPart = { .mesh = mesh, .region = NULL };
    MROffsetParameters params = mrOffsetParametersNew();
    params.voxelSize = mrSuggestVoxelSize(inputMeshPart, 10000.f);
    float offset = 0.1f;
    float offset2 = -0.2f;
    MRMesh* outputMesh = mrDoubleOffsetMesh(inputMeshPart, offset, offset2, &params, &errorString);
    TEST_ASSERT(errorString == NULL);
    mrMeshFree(outputMesh);
    mrMeshFree(mesh);
}

void testMcOffsetMesh(void)
{
    MRString* errorString = NULL;
    MRVector3f size = mrVector3fDiagonal(1.f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    MRMeshPart inputMeshPart = { .mesh = mesh, .region = NULL };
    MROffsetParameters params = mrOffsetParametersNew();
    params.voxelSize = mrSuggestVoxelSize(inputMeshPart, 10000.f);
    float offset = 0.1f;
    MRMesh* outputMesh = mrMcOffsetMesh(inputMeshPart, offset, &params, &errorString);
    TEST_ASSERT(errorString == NULL);
    mrMeshFree(outputMesh);
    mrMeshFree(mesh);
}

void testSharpOffsetMesh(void)
{
    MRString* errorString = NULL;
    MRVector3f size = mrVector3fDiagonal(1.f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    MRMeshPart inputMeshPart = { .mesh = mesh, .region = NULL };
    MROffsetParameters params = mrOffsetParametersNew();
    params.voxelSize = mrSuggestVoxelSize(inputMeshPart, 10000.f);
    MRGeneralOffsetParameters generalParams = mrGeneralOffsetParametersNew();
    generalParams.mode = MRGeneralOffsetParametersModeStandard;
    float offset = 0.1f;
    MRMesh* outputMesh = mrSharpOffsetMesh(inputMeshPart, offset, &params, &generalParams, &errorString);
    TEST_ASSERT(errorString == NULL);
    mrMeshFree(outputMesh);
    mrMeshFree(mesh);
}

void testGeneralOffsetMesh(void)
{
    MRString* errorString = NULL;
    MRVector3f size = mrVector3fDiagonal(1.f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    MRMeshPart inputMeshPart = { .mesh = mesh, .region = NULL };
    MROffsetParameters params = mrOffsetParametersNew();
    params.voxelSize = mrSuggestVoxelSize(inputMeshPart, 10000.f);
    MRGeneralOffsetParameters generalParams = mrGeneralOffsetParametersNew();
    generalParams.mode = MRGeneralOffsetParametersModeStandard;
    float offset = 0.1f;
    MRMesh* outputMesh = mrGeneralOffsetMesh(inputMeshPart, offset, &params, &generalParams, &errorString);
    TEST_ASSERT(errorString == NULL);
    mrMeshFree(outputMesh);
    mrMeshFree(mesh);
}

void testThickenMesh(void)
{
    MRString* errorString = NULL;
    MRVector3f size = mrVector3fDiagonal(1.f);
    MRVector3f base = mrVector3fDiagonal(-0.5f);
    MRMesh* mesh = mrMakeCube(&size, &base);
    MROffsetParameters params = mrOffsetParametersNew();
    params.voxelSize = mrSuggestVoxelSize((MRMeshPart) { .mesh = mesh, .region = NULL }, 10000000.f);
    MRGeneralOffsetParameters generalParams = mrGeneralOffsetParametersNew();
    generalParams.mode = MRGeneralOffsetParametersModeStandard;
    float offset = 0.1f;
    MRMesh* outputMesh = mrThickenMesh(mesh, offset, &params, &generalParams, &errorString);
    TEST_ASSERT(errorString == NULL);
    mrMeshFree(outputMesh);
    mrMeshFree(mesh);
}
