#include "MRCMesh/MRMeshSave.h"
#include "MRCMisc/expected_MR_Mesh_std_string.h"
#include "MRCMisc/expected_void_std_string.h"
#include "MRCMisc/std_string.h"
#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRPointCloud.h>
#include <MRCMesh/MRPointCloudTriangulation.h>
#include <MRCMesh/MRUniformSampling.h>
#include <MRCMesh/MRVector.h>
#include <MRCMisc/std_optional_MR_Mesh.h>
#include <MRCMisc/std_optional_MR_VertBitSet.h>
#include <MRCVoxels/MROffset.h>

#include <math.h>
#include <stdio.h>

#define PI 3.14159265358979323846

int main( void )
{
    // Generate point cloud
    MR_VertCoords* points = MR_VertCoords_Construct_1_uint64_t( 10000 );
    for ( int i = 0; i < 100; ++i )
    {
        float u = PI*2 * (float)i / ( 100.f - 1.f );
        for ( int j = 0; j < 100; ++j )
        {
            float v = PI * (float)j / ( 100.f - 1.f );

            *MR_VertCoords_index( points, (MR_VertId){ i * 100 + j } ) = (MR_Vector3f){
                cos( u ) * sin( v ),
                sin( u ) * sin( v ),
                cos( v )
            };
        }
    }
    MR_PointCloud* pc = MR_PointCloud_DefaultConstruct();
    MR_PointCloud_Set_points( pc, MR_PassBy_Move, points );
    // Fill the bitset of valid points in `pc` to all ones.
    MR_BitSet_resize( MR_VertBitSet_MutableUpcastTo_MR_BitSet( MR_PointCloud_GetMutable_validPoints( pc ) ), 10000, &(bool){true} );
    MR_VertCoords_Destroy( points );

    // Remove duplicate points
    MR_UniformSamplingSettings* samplingSettings = MR_UniformSamplingSettings_DefaultConstruct();
    MR_UniformSamplingSettings_Set_distance( samplingSettings, 1e-3f );
    MR_std_optional_MR_VertBitSet* vs = MR_pointUniformSampling( pc, samplingSettings );
    MR_UniformSamplingSettings_Destroy( samplingSettings );

    if ( !vs )
    {
        fprintf( stderr, "Removing duplicate points failed\n" );
        MR_std_optional_MR_VertBitSet_Destroy( vs );
        return 1;
    }

    MR_PointCloud_Set_validPoints( pc, MR_PassBy_Move, MR_std_optional_MR_VertBitSet_MutableValue( vs ) );
    MR_std_optional_MR_VertBitSet_Destroy( vs );

    MR_PointCloud_invalidateCaches( pc );

    // Triangulate the point cloud
    MR_std_optional_MR_Mesh* triangulatedOpt = MR_triangulatePointCloud( pc, NULL, MR_PassBy_DefaultArgument, NULL );
    MR_PointCloud_Destroy( pc );

    MR_Mesh* triangulated = MR_std_optional_MR_Mesh_MutableValue( triangulatedOpt );
    if ( !triangulated )
    {
        fprintf( stderr, "Triangulation failed\n" );
        MR_std_optional_MR_Mesh_Destroy( triangulatedOpt );
        return 1;
    }

    // Fix possible issues
    MR_OffsetParameters* offsetParams = MR_OffsetParameters_DefaultConstruct();
    MR_MeshPart* mp = MR_MeshPart_Construct( triangulated, NULL );
    MR_BaseShellParameters_Set_voxelSize( MR_OffsetParameters_MutableUpcastTo_MR_BaseShellParameters( offsetParams ), MR_suggestVoxelSize( mp, 5e+4f ) );
    MR_expected_MR_Mesh_std_string *meshEx = MR_offsetMesh( mp, 0.f, offsetParams );
    MR_MeshPart_Destroy( mp );
    MR_std_optional_MR_Mesh_Destroy( triangulatedOpt );

    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );
    if ( !mesh )
    {
        fprintf( stderr, "Offset failed: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        MR_expected_MR_Mesh_std_string_Destroy( meshEx );
        return 1;
    }

    // Save the result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "result.stl", NULL, NULL);
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );

    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        MR_expected_void_std_string_Destroy( saveEx );
        return 1;
    }

    MR_expected_void_std_string_Destroy( saveEx );
    return 0;
}
