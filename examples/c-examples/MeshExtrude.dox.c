#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshExtrude.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRRegionBoundary.h>

#include <stdlib.h>

int main( int argc, char* argv[] )
{
    // Load mesh
    MRMesh* mesh = mrMeshLoadFromAnySupportedFormat( "mesh.stl", NULL );

    // Select faces to extrude
    MRFaceBitSet* facesToExtrude = mrFaceBitSetNew( 0, false );
    mrBitSetAutoResizeSet( (MRBitSet*)facesToExtrude, 1, true );
    mrBitSetAutoResizeSet( (MRBitSet*)facesToExtrude, 2, true );

    // Create duplicated verts on region boundary
    mrMakeDegenerateBandAroundRegion( mesh, facesToExtrude );

    // Find vertices that will be moved
    MRVertBitSet* vertsToMove = mrGetIncidentVertsFromFaces( mrMeshTopology( mesh ), facesToExtrude );
    MRVector3f* points = mrMeshPointsRef( mesh );
    MRVector3f shift = mrVector3fPlusZ();
    for ( int i = 0; i < mrMeshPointsNum( mesh ); ++i )
        if ( mrBitSetTest( (MRBitSet*)vertsToMove, i ) )
            points[i] = mrVector3fAdd( &points[i], &shift );

    // Invalidate internal caches after manual changing
    mrMeshInvalidateCaches( mesh, true );

    // Save mesh
    mrMeshSaveToAnySupportedFormat( mesh, "extrudedMesh.stl", NULL, NULL );

    mrVertBitSetFree( vertsToMove );
    mrFaceBitSetFree( facesToExtrude );
    mrMeshFree( mesh );
    return EXIT_SUCCESS;
}
