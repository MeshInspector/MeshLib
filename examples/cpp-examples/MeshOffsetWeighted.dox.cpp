#include "MRMesh/MRTorus.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRParallelFor.h"
#include "MRVoxels/MRWeightedPointsShell.h"
#include "MRVoxels/MROffset.h"
#include "MRMesh/MRMeshSave.h"
#include <iostream>

int main()
{
    // Create some mesh
    auto mesh = MR::makeTorus();

    // Create VertScalars obj with weights for every vertex
    auto vertSize = mesh.topology.vertSize();
    MR::VertScalars scalars( vertSize );
    MR::ParallelFor( scalars, [&] ( MR::VertId v )
    {
        scalars[v] = std::abs( mesh.points[v].x / 5.0f ); // Individual extra offset sizes for points
    } );

    auto params = MR::WeightedShell::ParametersMetric();
    // Algorithm is voxel based, voxel size affects performance and form of result mesh
    params.voxelSize = MR::suggestVoxelSize( mesh, 1000 );
    // common basic offset applied for all point
    // Vertex-specific weighted offsets applied after the basic one
    params.offset = 0.2f;
    params.dist.maxWeight = *std::max_element( MR::begin( scalars ), MR::end( scalars ) ); // should always have maximum between weights provided
 
    auto res = MR::WeightedShell::meshShell( mesh, scalars, params );
    if ( !res.has_value() )
    {
        std::cerr << res.error();
        return 1;
    }
 
    auto saveRes = MR::MeshSave::toAnySupportedFormat( *res, "offset_weighted.ctm" );
    if ( !saveRes.has_value() )
    {
        std::cerr << saveRes.error();
        return 1;
    }
    return 0;
}
