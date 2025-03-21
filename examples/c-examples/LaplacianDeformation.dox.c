#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRExpandShrink.h>
#include <MRMeshC/MRLaplacian.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>

#include <stdlib.h>

int main( int argc, char* argv[] )
{
    // Load mesh
    MRMesh* mesh = mrMeshLoadFromAnySupportedFormat( "mesh.stl", NULL );

    // Construct deformer on the mesh vertices
    MRLaplacian* lDeformer = mrLaplacianNew( mesh );

    // Find an area for the deformation anchor points
    const MRVertBitSet* verts = mrMeshTopologyGetValidVerts( mrMeshTopology( mesh ) );
    MRVertId ancV0 = { mrBitSetFindFirst( (MRBitSet*)verts ) };
    MRVertId ancV1 = { mrBitSetFindLast( (MRBitSet*)verts ) };
    // Mark the anchor points in the free area
    MRVertBitSet* freeVerts = mrVertBitSetNew( mrBitSetSize( (MRBitSet*)verts ), false );
    mrBitSetSet( (MRBitSet*)verts, ancV0.id, true );
    mrBitSetSet( (MRBitSet*)verts, ancV1.id, true );
    // Expand the free area
    mrExpandVertRegion( mrMeshTopology( mesh ), freeVerts, 5 );

    // Initialize laplacian
    mrLaplacianInit( lDeformer, freeVerts, MREdgeWeightsCotanWithAreaEqWeight, MRLaplacianRememberShapeYes );

    MRBox3f bbox = mrMeshComputeBoundingBox( mesh, NULL );
    float shiftAmount = mrBox3fDiagonal( &bbox ) * 0.01f;
    // Fix the anchor vertices in the required position
    const MRVector3f* points = mrMeshPoints( mesh );
    MRVector3f posV0 = mrMeshNormalFromVert( mesh, ancV0 );
    posV0 = mrVector3fMulScalar( &posV0, shiftAmount );
    posV0 = mrVector3fAdd( &points[ancV0.id], &posV0 );
    mrLaplacianFixVertex( lDeformer, ancV0, &posV0, true );
    MRVector3f posV1 = mrMeshNormalFromVert( mesh, ancV1 );
    posV1 = mrVector3fMulScalar( &posV1, shiftAmount );
    posV1 = mrVector3fAdd( &points[ancV1.id], &posV1 );
    mrLaplacianFixVertex( lDeformer, ancV1, &posV1, true );

    // Move the free vertices according to the anchor ones
    mrLaplacianApply( lDeformer );

    // Invalidate the mesh because of the external vertex changes
    mrMeshInvalidateCaches( mesh, true );

    // Save the deformed mesh
    mrMeshSaveToAnySupportedFormat( mesh, "deformed_mesh.stl", NULL, NULL );

    mrVertBitSetFree( freeVerts );
    mrLaplacianFree( lDeformer );
    mrMeshFree( mesh );
    return EXIT_SUCCESS;
}
