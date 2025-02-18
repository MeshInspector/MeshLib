#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshFillHole.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>

#include <stdlib.h>

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;

    // Load meshes
    MRMesh* meshARes = mrMeshLoadFromAnySupportedFormat( "meshAwithHole.stl", NULL );
    MRMesh* meshBRes = mrMeshLoadFromAnySupportedFormat( "meshBwithHole.stl", NULL );

    // Unite meshes
    MRMesh* mesh = meshARes;
    mrMeshAddMesh( mesh, meshBRes );

    // Find holes (expect that there are exactly 2 holes)
    MREdgePath* edges = mrMeshTopologyFindHoleRepresentiveEdges( mrMeshTopology( mesh ) );
    if ( edges->size != 2 )
        goto out_edges;

    // Connect two holes
    MRFillHoleMetric* metric = mrGetUniversalMetric( mesh );
    MRStitchHolesParams params = {
        .metric = metric,
        .outNewFaces = NULL,
    };
    mrBuildCylinderBetweenTwoHoles( mesh, edges->data[0], edges->data[1], &params );

    // Save result
    mrMeshSaveToAnySupportedFormat( mesh, "stitchedMesh.stl", NULL, NULL );

    rc = EXIT_SUCCESS;
out_metric:
    mrFillHoleMetricFree( metric );
out_edges:
    mrEdgePathFree( edges );
out_mesh:
    mrMeshFree( meshBRes );
    mrMeshFree( meshARes );
    return rc;
}
