#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MROffset.h>
#include <MRMeshC/MRPointCloud.h>
#include <MRMeshC/MRPointCloudTriangulation.h>
#include <MRMeshC/MRUniformSampling.h>

#include <math.h>

int main( int argc, char* argv[] )
{
    // Generate point cloud
    MRVector3f points[10000];
    for ( int i = 0; i < 100; ++i )
    {
        float u = M_PI_2 * (float)i / ( 100.f - 1.f );
        for ( int j = 0; j < 100; ++j )
        {
            float v = M_PI * (float)j / ( 100.f - 1.f );

            points[i * 100 + j] = (MRVector3f){
                cos( u ) * sin( v ),
                sin( u ) * sin( v ),
                cos( v )
            };
        }
    }
    MRPointCloud* pc = mrPointCloudFromPoints( points, 10000 );
    // Remove duplicate points
    MRUniformSamplingSettings samplingSettings = mrUniformSamplingSettingsNew();
    samplingSettings.distance = 1e-3f;
    MRVertBitSet* vs = mrPointUniformSampling( pc, &samplingSettings );
    mrPointCloudSetValidPoints( pc, vs );
    mrPointCloudInvalidateCaches( pc );

    // Triangulate it
    MRMesh* triangulated = mrTriangulatePointCloud( pc, NULL );

    // Fix possible issues
    MROffsetParameters offsetParams = mrOffsetParametersNew();
    offsetParams.voxelSize = mrSuggestVoxelSize( (MRMeshPart){ triangulated, NULL }, 5e+6f );
    MRMesh* mesh = mrOffsetMesh( (MRMeshPart){ triangulated, NULL }, 0.f, &offsetParams, NULL );

    mrMeshFree( mesh );
    mrMeshFree( triangulated );
    mrVertBitSetFree( vs );
    mrPointCloudFree( pc );
}
