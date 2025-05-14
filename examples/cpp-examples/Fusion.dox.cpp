#include <MRMesh/MRPointsLoad.h>
#include <MRMesh/MRPointCloud.h>
#include "MRMesh/MRBox.h"
#include "MRMesh/MRPointCloudRadius.h"
#include "MRVoxels/MRPointsToMeshFusion.h"
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRMesh.h>
#include <iostream>

int main()
{
    // load points
    auto loadRes = MR::PointsLoad::fromAnySupportedFormat( "NefertitiPoints.ply" );
    if ( !loadRes.has_value() )
    {
        std::cerr << loadRes.error() << "\n";
        return 1; // error while loading file
    }

    MR::PointsToMeshParameters params;
    params.voxelSize = loadRes->computeBoundingBox().diagonal() * 1e-3f;
    params.sigma = std::max( params.voxelSize, MR::findAvgPointsRadius( *loadRes, 40 ) );
    params.minWeight = 1.0f;

    auto fusionRes = MR::pointsToMeshFusion( *loadRes, params );
    if ( !fusionRes.has_value() )
    {
        std::cerr << fusionRes.error() << "\n";
        return 1; // error while saving file
    }

    auto saveRes = MR::MeshSave::toAnySupportedFormat( *fusionRes, "NefertitiMesh.ply" );
    if ( !saveRes.has_value() )
    {
        std::cerr << saveRes.error() << "\n";
        return 1; // error while saving file
    }
    return 0;
}