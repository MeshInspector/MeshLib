#include <MRMesh/MRPointsLoad.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRDelaunayTriangulationXY.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRColor.h>
#include <MRIOExtras/MRIOExtras.h>
#include <iostream>

int main()
{
    // CTM, JPEG and other formats beyond the basic ones come with the MRIOExtras library
    MR::loadIOExtras();

    // load points
    MR::VertColors colors;
    MR::PointsLoadSettings pls;
    pls.colors = &colors;
    auto loadRes = MR::PointsLoad::fromAnySupportedFormat( "TerrainPoints.ply", pls );
    if ( !loadRes.has_value() )
    {
        std::cerr << loadRes.error() << "\n";
        return 1; // error while loading file
    }
    auto triangulationRes = MR::delaunayTriangulationXY( *loadRes );
    if ( !triangulationRes.has_value() )
    {
        std::cerr << triangulationRes.error() << "\n";
        return 1; // error while triangulating
    }

    MR::SaveSettings ss;
    if ( loadRes->points.size() == colors.size() )
        ss.colors = &colors;
    auto saveRes = MR::MeshSave::toAnySupportedFormat( *triangulationRes, "TerrainMesh.ctm", ss );
    if ( !saveRes.has_value() )
    {
        std::cerr << saveRes.error() << "\n";
        return 1; // error while saving file
    }
    return 0;
}
