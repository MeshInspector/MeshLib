#include "MRSolarRadiation.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRIntersectionPrecomputes.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

std::vector<Vector3f> createSkyPatches()
{
    constexpr int numberOfRows = 7;
    constexpr int patchesPerRow[numberOfRows] = { 30, 30, 24, 24, 18, 12, 6 };
    constexpr int resolutionMultiplier = 1;

    float patchAngleIncrement = PI2_F / ( numberOfRows * resolutionMultiplier + 0.5f );

    int totalPatches = 0;
    for ( int patchesInRow : patchesPerRow )
        totalPatches += patchesInRow;

    totalPatches *= resolutionMultiplier * resolutionMultiplier;
    totalPatches += 1; // Add zenith 

    std::vector<Vector3f> patches;
    patches.reserve( totalPatches );

    for ( int i = 0; i < numberOfRows * resolutionMultiplier; i++ )
    {
        float currentAltitude = i * patchAngleIncrement; 
        int patchesInCurrentRow = patchesPerRow[i / resolutionMultiplier] * resolutionMultiplier;

        for (int j = 0; j < patchesInCurrentRow; j++)
        {
            // Start from North and proceed in a clockwise direction
            float currentAzimuth = PI2_F - ( 2 * PI_F * j / patchesInCurrentRow );
            patches.push_back( unitVector3( currentAltitude + patchAngleIncrement / 2, currentAzimuth ) );
        }
    }

    // Add zenith patch
    patches.push_back( unitVector3( PI2_F, 0.0f ) );

    return patches;
}

VertScalars computeSolarRadiation( const Mesh & terrain, const VertCoords & samples, const VertBitSet & validSamples )
{
    MR_TIMER
    VertScalars res( samples.size(), 0.0f );

    const auto skyPatches = createSkyPatches();
    const auto rPatches = 1 / float( skyPatches.size() );
    std::vector<IntersectionPrecomputes<float>> precs;
    precs.reserve( skyPatches.size() );
    for ( const auto & sp : skyPatches )
        precs.emplace_back( sp );

    BitSetParallelFor( validSamples, [&]( VertId sampleVertId )
    {
        const auto samplePt = samples[sampleVertId];

        auto raysInSky = skyPatches.size();
        for ( int i = 0; i < skyPatches.size(); ++i )
        {
            if ( rayMeshIntersect( terrain, Line3f( samplePt, skyPatches[i] ), 0, FLT_MAX, &precs[i], false ) )
                --raysInSky;
        }
        res[sampleVertId] = rPatches * (float)raysInSky;
    } );

    return res;
}

} //namespace MR
