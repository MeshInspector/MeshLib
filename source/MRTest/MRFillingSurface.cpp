#ifndef MESHLIB_NO_VOXELS

#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMesh.h>
#include <MRVoxels/MRMinimalSurface.h>
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRBox.h"


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

TEST( MRMesh, CellularFillingSurfaceDensity )
{
    Vector3f size{ 1.f, 1.f, 1.f };

    for ( float T = 0.1f; T < 1.f; T += 0.1f )
    {
        for ( float R = 0.f; R < T / 2.f; R += 0.1f )
        {
            for ( float W = 0.1f; W < T; W += 0.1f )
            {
                auto res = FillingSurface::CellularSurface::build( size,
                               { .period = Vector3f::diagonal( T ), .width = Vector3f::diagonal( W ), .r = R, .highRes = true } );
                ASSERT_TRUE( res );
                fillHoles( *res, res->topology.findHoleRepresentiveEdges() );
                ASSERT_TRUE( res->topology.findNumHoles() == 0 );

                const auto realDensity = res->volume() / res->getBoundingBox().volume();
                const auto predictedDensity = FillingSurface::CellularSurface::estimateDensity( T, W, R );
                ASSERT_NEAR( realDensity, predictedDensity, 0.01f );

                const auto predictedWidth = FillingSurface::CellularSurface::estimateWidth( T, R, realDensity );

                // sometimes more than one solution exist, so we should not compare directly with W, but rather we have to rebuild the surface
                auto res2 = FillingSurface::CellularSurface::build( size,
                                { .period = Vector3f::diagonal( T ), .width = Vector3f::diagonal( predictedWidth ), .r = R, .highRes = true } );
                const auto predictedDensity2 = res->volume() / res->getBoundingBox().volume();
                ASSERT_NEAR( realDensity, predictedDensity2, 0.01f );
            }
        }
    }
}

}

#endif