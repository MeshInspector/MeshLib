#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshProject.h"
#include "MRMesh/MRMesh.h"
#include <iostream>

int main()
{
    auto mesh1Res = MR::MeshLoad::fromAnySupportedFormat( "mesh1.ctm" );
    if ( !mesh1Res.has_value() )
    {
        std::cerr << mesh1Res.error();
        return 1;
    }
    auto point = MR::Vector3f( 1.5f, 1.0f, 0.5f );

    auto dist = MR::findSignedDistance( point, *mesh1Res );
    if ( dist )
        std::cout << "Signed distance from point to mesh " << dist->dist << "\n";
    return 0;
}
