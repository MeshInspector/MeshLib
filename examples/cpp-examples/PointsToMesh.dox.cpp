#include <MRMesh/MRPointsLoad.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRPointCloudTriangulation.h>
#include <MRMesh/MRMeshSave.h>
#include <iostream>

int main()
{
    // load points
    auto loadRes = MR::PointsLoad::fromAnySupportedFormat( "Points.ply" );
    if ( !loadRes.has_value() )
    {
        std::cerr << loadRes.error() << "\n";
        return 1; // error while loading file
    }
    auto triangulationRes = MR::triangulatePointCloud( *loadRes );
    assert( triangulationRes ); // can be nullopt only if canceled by progress callback

    auto saveRes = MR::MeshSave::toAnySupportedFormat( *triangulationRes, "Mesh.ctm" );
    if ( !saveRes.has_value() )
    {
        std::cerr << saveRes.error() << "\n";
        return 1; // error while saving file
    }
    return 0;
}
