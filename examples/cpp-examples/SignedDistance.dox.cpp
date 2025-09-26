#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshMeshDistance.h"
#include <iostream>

int main()
{
    auto mesh1Res = MR::MeshLoad::fromAnySupportedFormat( "mesh1.ctm" );
    if ( !mesh1Res.has_value() )
    {
        std::cerr << mesh1Res.error();
        return 1;
    }
    auto mesh2Res = MR::MeshLoad::fromAnySupportedFormat( "mesh2.ctm" );
    if ( !mesh2Res.has_value() )
    {
        std::cerr << mesh2Res.error();
        return 1;
    }

    auto dist = MR::findSignedDistance( *mesh1Res, *mesh2Res );
    std::cout << "Signed distance between meshes is " << dist.signedDist << "\n";
    return 0;
}
