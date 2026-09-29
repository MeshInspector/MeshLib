#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRBooleanOperation.h>
#include <MRCMesh/MRId.h>
#include <MRCMesh/MRMakeSphereMesh.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshBoolean.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRVector.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // First, create a unit sphere.
    float radius = 1.f; // Set radius for the test
    int horizontalResolution = 64; // Increase horizontal resolution
    int verticalResolution = 64; // Increase vertical resolution

    MR_Mesh* sphere1 = MR_makeUVSphere( &radius, &horizontalResolution, &verticalResolution );

    // Create a copy of this sphere and offset it.
    MR_Mesh* sphere2 = MR_Mesh_ConstructFromAnother( MR_PassBy_Copy, sphere1 );
    MR_Vector3f xfTranslation = {.x = 0.7f};
    MR_AffineXf3f xf = MR_AffineXf3f_translation( &xfTranslation );
    MR_Mesh_transform( sphere2, &xf, NULL );

    // Ask for an optional mapper relating the primitives of the input meshes to the primitives of the result.
    MR_BooleanResultMapper* mapper = MR_BooleanResultMapper_DefaultConstruct();
    MR_BooleanParameters* params = MR_BooleanParameters_DefaultConstruct();
    MR_BooleanParameters_Set_mapper( params, mapper );

    // Perform the boolean operation.
    MR_BooleanResult* result = MR_boolean_4_const_MR_Mesh_ref( sphere1, sphere2, MR_BooleanOperation_Intersection, params );
    MR_BooleanParameters_Destroy( params );
    if ( !MR_BooleanResult_valid( result ) )
    {
        fprintf( stderr, "Failed to perform boolean: %s\n", MR_std_string_data( MR_BooleanResult_Get_errorString( result ) ) );
        goto fail;
    }

    // Find the faces of the result produced by each input sphere, and the faces the cut created.
    MR_FaceBitSet* facesOfSphere1 = MR_BooleanResultMapper_map_MR_FaceBitSet(
        mapper, MR_MeshTopology_getValidFaces( MR_Mesh_Get_topology( sphere1 ) ), MR_BooleanResultMapper_MapObject_A );
    MR_FaceBitSet* facesOfSphere2 = MR_BooleanResultMapper_map_MR_FaceBitSet(
        mapper, MR_MeshTopology_getValidFaces( MR_Mesh_Get_topology( sphere2 ) ), MR_BooleanResultMapper_MapObject_B );
    MR_FaceBitSet* newFaces = MR_BooleanResultMapper_newFaces( mapper );

    printf( "faces from sphere1: %zu\n", MR_FaceBitSet_count( facesOfSphere1 ) );
    printf( "faces from sphere2: %zu\n", MR_FaceBitSet_count( facesOfSphere2 ) );
    printf( "faces created by the cut: %zu\n", MR_FaceBitSet_count( newFaces ) );

    MR_FaceBitSet_Destroy( newFaces );
    MR_FaceBitSet_Destroy( facesOfSphere2 );
    MR_FaceBitSet_Destroy( facesOfSphere1 );

    // Map one particular face of sphere1 forward: the cut can split it in several faces of the
    // result, or drop it completely if that part of sphere1 is not in the result.
    MR_FaceId faceOfSphere1 = { 793 };
    MR_FaceBitSet* oneFace = MR_FaceBitSet_DefaultConstruct();
    MR_FaceBitSet_autoResizeSet_2( oneFace, faceOfSphere1, NULL );
    MR_FaceBitSet* producedFaces = MR_BooleanResultMapper_map_MR_FaceBitSet( mapper, oneFace, MR_BooleanResultMapper_MapObject_A );
    printf( "face %d of sphere1 produced %zu faces of the result\n", faceOfSphere1.id_, MR_FaceBitSet_count( producedFaces ) );

    // And backward: the face of sphere1 each face of the result came from
    // (invalid id for the faces that came from sphere2).
    MR_FaceMap* new2OldFaces = MR_BooleanResultMapper_getNew2OldFaceMap( mapper, MR_BooleanResultMapper_MapObject_A );
    MR_FaceId resultFace = MR_FaceBitSet_find_first( producedFaces );
    printf( "face %d of the result came from face %d of sphere1\n",
        resultFace.id_, MR_FaceMap_index( new2OldFaces, resultFace )->id_ );

    MR_FaceMap_Destroy( new2OldFaces );
    MR_FaceBitSet_Destroy( producedFaces );
    MR_FaceBitSet_Destroy( oneFace );

    // Save result to an STL file.
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( MR_BooleanResult_Get_mesh( result ), "out_boolean.stl", NULL, NULL);
    if ( MR_expected_void_std_string_error( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_data( MR_expected_void_std_string_error( saveEx ) ) );
        goto fail;
    }

    rc = EXIT_SUCCESS;
fail:
    MR_BooleanResult_Destroy( result );
    MR_BooleanResultMapper_Destroy( mapper );
    MR_Mesh_Destroy( sphere2 );
    MR_Mesh_Destroy( sphere1 );
    return rc;
}
