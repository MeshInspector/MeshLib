#ifndef MESHLIB_NO_VOXELS

#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMesh.h>
#include <MRVoxels/MRMinimalSurface.h>
#include "MRMesh/MRMeshFillHole.h"


namespace MR
{

TEST( MRMesh, CellularFillingSurface )
{
    Vector3f size{ 1.f, 1.f, 1.f };

    for ( float T = 0.1f; T < 1.f; T += 0.1f )
    {
        for ( float R = 0.f; R < T / 2.f; R += 0.1f )
        {
            for ( float W = 0.1f; W < T; W += 0.1f )
            {
                auto res = FillingSurface::CellularSurface::build( size,
                               { .period = Vector3f::diagonal( T ), .width = Vector3f::diagonal( W ), .r = R } );
                ASSERT_TRUE( res );
            }
        }
    }
}

}

#endif