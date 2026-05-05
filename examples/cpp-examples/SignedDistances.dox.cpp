#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRMesh.h"
#include <iostream>

int main()
{
    auto refMesh = MR::MeshLoad::fromAnySupportedFormat( "mesh1.ctm" );
    if ( !refMesh.has_value() )
    {
        std::cerr << refMesh.error();
        return 1;
    }
    auto mesh = MR::MeshLoad::fromAnySupportedFormat( "mesh2.ctm" );
    if ( !mesh.has_value() )
    {
        std::cerr << mesh.error();
        return 1;
    }
    // get object of VertScalars - set of distances between points of target mesh and reference mesh
    auto vertDistances = MR::findSignedDistances( *refMesh, *mesh );
    auto minmax = std::minmax_element( begin( vertDistances ), end( vertDistances ) );
    std::cout << "Distance between reference mesh and the closest point of target mesh is " << *minmax.first << "\n";
    std::cout << "Distance between reference mesh and the farthest point of target mesh is " << *minmax.second << "\n";

    return 0;
}
