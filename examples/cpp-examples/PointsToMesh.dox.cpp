#include <MRMesh/MRPointsLoad.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRPointCloudTriangulation.h>
#include <MRMesh/MRMeshSave.h>
#include <MRIOExtras/MRIOExtras.h>
#include <iostream>

int main()
{
    // CTM, JPEG and other formats beyond the basic ones come with the MRIOExtras library
    MR::loadIOExtras();

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
