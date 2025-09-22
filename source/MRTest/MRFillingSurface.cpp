#ifndef MESHLIB_NO_VOXELS

#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MRBox.h>
#include <MRVoxels/MRFillingSurface.h>

namespace MR
{

#ifdef NDEBUG
const float cStepSize = 0.2f;
#else
const float cStepSize = 0.5f;
#endif

TEST( MRMesh, CellularFillingSurface )
{
    Vector3f size{ 1.f, 1.f, 1.f };

    for ( float T = 0.1f; T < 1.f; T += cStepSize )
    {
        for ( float R = 0.f; R < T / 2.f; R += cStepSize )
        {
            for ( float W = 0.1f; W < T; W += cStepSize )
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
    int noSolutions = 0;

    for ( float T = 0.1f; T < 1.f; T += cStepSize )
    {
        for ( float R = 0.f; R < T / 2.f; R += cStepSize )
        {
            for ( float W = 0.1f; W < T; W += cStepSize )
            {
                auto res = FillingSurface::CellularSurface::build( size,
                               { .period = Vector3f::diagonal( T ), .width = Vector3f::diagonal( W ), .r = R, .highRes = true } );
                ASSERT_TRUE( res );
                fillHoles( *res, res->topology.findHoleRepresentiveEdges() );
                ASSERT_TRUE( res->topology.findNumHoles() == 0 );

                const auto realDensity = static_cast<float>( res->volume() / res->getBoundingBox().volume() );
                const auto predictedDensity = FillingSurface::CellularSurface::estimateDensity( T, W, R );
                ASSERT_NEAR( realDensity, predictedDensity, 0.01f );

                float R2use = R;
                auto maybePredictedWidth = FillingSurface::CellularSurface::estimateWidth( T, R2use, realDensity );

                // sometimes no solutions exist, then the first retry with reduced sphere size must succeed
                if ( !maybePredictedWidth )
                {
                    R2use /= 2.f;
                    maybePredictedWidth = FillingSurface::CellularSurface::estimateWidth( T, R2use, realDensity );
                    ++noSolutions;
                }
                ASSERT_TRUE( maybePredictedWidth );

                // sometimes more than one solution exist, so we should not compare directly with W, but rather we have to rebuild the surface
                auto res2 = FillingSurface::CellularSurface::build( size,
                                { .period = Vector3f::diagonal( T ), .width = Vector3f::diagonal( *maybePredictedWidth ), .r = R2use, .highRes = true } );
                ASSERT_TRUE( res2 );
                const auto predictedDensity2 = res2->volume() / res2->getBoundingBox().volume();
                ASSERT_NEAR( realDensity, predictedDensity2, 0.01f );
            }
        }
    }

    ASSERT_LE( noSolutions, 1 );
}

}

#endif