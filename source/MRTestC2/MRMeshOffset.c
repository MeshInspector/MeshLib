#include "TestMacros.h"
#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRBox.h>
#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRPartMapping.h>
#include <MRCMesh/MRString.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCVoxels/MROffset.h>


void testOffsetMesh(void)
{
    MR_Vector3f size = MR_Vector3f_diagonal(1.f);
    MR_Vector3f base = MR_Vector3f_diagonal(-0.5f);
    MR_Mesh* mesh = MR_makeCube(&size, &base);
    MR_MeshPart* inputMeshPart = MR_MeshPart_Construct(mesh, NULL);
    MR_OffsetParameters* params = MR_OffsetParameters_DefaultConstruct();
    MR_BaseShellParameters_Set_voxelSize(MR_OffsetParameters_MutableUpcastTo_MR_BaseShellParameters(params), MR_suggestVoxelSize(inputMeshPart, 10000.f));
    float offset = 0.1f;
    MR_expected_MR_Mesh_std_string* outputMesh = MR_offsetMesh(inputMeshPart, offset, params);
    MR_OffsetParameters_Destroy(params);
    MR_MeshPart_Destroy(inputMeshPart);
    TEST_ASSERT(MR_expected_MR_Mesh_std_string_GetValue(outputMesh));
    MR_expected_MR_Mesh_std_string_Destroy(outputMesh);
    MR_Mesh_Destroy(mesh);
}

void testDoubleOffsetMesh(void)
{
    MR_Vector3f size = MR_Vector3f_diagonal(1.f);
    MR_Vector3f base = MR_Vector3f_diagonal(-0.5f);
    MR_Mesh* mesh = MR_makeCube(&size, &base);
    MR_MeshPart* inputMeshPart = MR_MeshPart_Construct(mesh, NULL);
    MR_OffsetParameters* params = MR_OffsetParameters_DefaultConstruct();
    MR_BaseShellParameters_Set_voxelSize(MR_OffsetParameters_MutableUpcastTo_MR_BaseShellParameters(params), MR_suggestVoxelSize(inputMeshPart, 10000.f));
    float offset = 0.1f;
    float offset2 = -0.2f;
    MR_expected_MR_Mesh_std_string* outputMesh = MR_doubleOffsetMesh(inputMeshPart, offset, offset2, params);
    MR_OffsetParameters_Destroy(params);
    MR_MeshPart_Destroy(inputMeshPart);
    TEST_ASSERT(MR_expected_MR_Mesh_std_string_GetValue(outputMesh));
    MR_expected_MR_Mesh_std_string_Destroy(outputMesh);
    MR_Mesh_Destroy(mesh);
}

void testMcOffsetMesh(void)
{
    MR_Vector3f size = MR_Vector3f_diagonal(1.f);
    MR_Vector3f base = MR_Vector3f_diagonal(-0.5f);
    MR_Mesh* mesh = MR_makeCube(&size, &base);
    MR_MeshPart* inputMeshPart = MR_MeshPart_Construct(mesh, NULL);
    MR_OffsetParameters* params = MR_OffsetParameters_DefaultConstruct();
    MR_BaseShellParameters_Set_voxelSize(MR_OffsetParameters_MutableUpcastTo_MR_BaseShellParameters(params), MR_suggestVoxelSize(inputMeshPart, 10000.f));
    float offset = 0.1f;
    MR_expected_MR_Mesh_std_string* outputMesh = MR_mcOffsetMesh(inputMeshPart, offset, params, NULL);
    MR_OffsetParameters_Destroy(params);
    MR_MeshPart_Destroy(inputMeshPart);
    TEST_ASSERT(MR_expected_MR_Mesh_std_string_GetValue(outputMesh));
    MR_expected_MR_Mesh_std_string_Destroy(outputMesh);
    MR_Mesh_Destroy(mesh);
}

void testSharpOffsetMesh(void)
{
    MR_Vector3f size = MR_Vector3f_diagonal(1.f);
    MR_Vector3f base = MR_Vector3f_diagonal(-0.5f);
    MR_Mesh* mesh = MR_makeCube(&size, &base);
    MR_MeshPart* inputMeshPart = MR_MeshPart_Construct(mesh, NULL);
    MR_SharpOffsetParameters* params = MR_SharpOffsetParameters_DefaultConstruct();
    MR_BaseShellParameters_Set_voxelSize(MR_SharpOffsetParameters_MutableUpcastTo_MR_BaseShellParameters(params), MR_suggestVoxelSize(inputMeshPart, 10000.f));
    float offset = 0.1f;
    MR_expected_MR_Mesh_std_string* outputMesh = MR_sharpOffsetMesh(inputMeshPart, offset, params);
    MR_SharpOffsetParameters_Destroy(params);
    MR_MeshPart_Destroy(inputMeshPart);
    TEST_ASSERT(MR_expected_MR_Mesh_std_string_GetValue(outputMesh));
    MR_expected_MR_Mesh_std_string_Destroy(outputMesh);
    MR_Mesh_Destroy(mesh);
}

void testGeneralOffsetMesh(void)
{
    MR_Vector3f size = MR_Vector3f_diagonal(1.f);
    MR_Vector3f base = MR_Vector3f_diagonal(-0.5f);
    MR_Mesh* mesh = MR_makeCube(&size, &base);
    MR_MeshPart* inputMeshPart = MR_MeshPart_Construct(mesh, NULL);
    MR_GeneralOffsetParameters* params = MR_GeneralOffsetParameters_DefaultConstruct();
    MR_BaseShellParameters_Set_voxelSize(MR_GeneralOffsetParameters_MutableUpcastTo_MR_BaseShellParameters(params), MR_suggestVoxelSize(inputMeshPart, 10000.f));
    MR_GeneralOffsetParameters_Set_mode(params, MR_OffsetMode_Standard);
    float offset = 0.1f;
    MR_expected_MR_Mesh_std_string* outputMesh = MR_generalOffsetMesh(inputMeshPart, offset, params);
    MR_GeneralOffsetParameters_Destroy(params);
    MR_MeshPart_Destroy(inputMeshPart);
    TEST_ASSERT(MR_expected_MR_Mesh_std_string_GetValue(outputMesh));
    MR_expected_MR_Mesh_std_string_Destroy(outputMesh);
    MR_Mesh_Destroy(mesh);
}

void testThickenMesh(void)
{
    MR_Vector3f size = MR_Vector3f_diagonal(1.f);
    MR_Vector3f base = MR_Vector3f_diagonal(-0.5f);
    MR_Mesh* mesh = MR_makeCube(&size, &base);
    MR_MeshPart* inputMeshPart = MR_MeshPart_Construct(mesh, NULL);
    MR_GeneralOffsetParameters* params = MR_GeneralOffsetParameters_DefaultConstruct();
    MR_BaseShellParameters_Set_voxelSize(MR_GeneralOffsetParameters_MutableUpcastTo_MR_BaseShellParameters(params), MR_suggestVoxelSize(inputMeshPart, 10000000.f));
    MR_GeneralOffsetParameters_Set_mode(params, MR_OffsetMode_Standard);
    float offset = 0.1f;
    MR_PartMapping* map = MR_PartMapping_DefaultConstruct();
    MR_expected_MR_Mesh_std_string* outputMesh = MR_thickenMesh(mesh, offset, params, map);
    MR_PartMapping_Destroy(map);
    MR_GeneralOffsetParameters_Destroy(params);
    MR_MeshPart_Destroy(inputMeshPart);
    TEST_ASSERT(MR_expected_MR_Mesh_std_string_GetValue(outputMesh));
    MR_expected_MR_Mesh_std_string_Destroy(outputMesh);
    MR_Mesh_Destroy(mesh);
}
