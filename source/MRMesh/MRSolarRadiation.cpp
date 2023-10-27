#include "MRSolarRadiation.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRIntersectionPrecomputes.h"
#include "MRLine3.h"
#include "MRMeshIntersect.h"
#include "MRTimer.h"
#include <cfloat>
#include <cstdlib>

namespace MR
{

std::vector<Vector3f> sampleHalfSphere()
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

VertScalars computeSkyViewFactor( const Mesh & terrain, const VertCoords & samples, const VertBitSet & validSamples,
    const std::vector<SkyPatch> & skyPatches, BitSet * outSkyRays )
{
    MR_TIMER
    VertScalars res( samples.size(), 0.0f );

    float maxRadiation = 0;
    for ( const auto & patch : skyPatches )
        maxRadiation += patch.radiation;
    const float rMaxRadiation = 1 / maxRadiation;

    if ( outSkyRays )
    {
        *outSkyRays = findSkyRays( terrain, samples, validSamples, skyPatches );
        BitSetParallelFor( validSamples, [&]( VertId sampleVertId )
        {
            float totalRadiation = 0;
            auto ray = size_t( sampleVertId ) * skyPatches.size();
            for ( int i = 0; i < skyPatches.size(); ++i, ++ray )
            {
                if ( outSkyRays->test( ray ) )
                    totalRadiation += skyPatches[i].radiation;
            }
            res[sampleVertId] = rMaxRadiation * totalRadiation;
        } );

        return res;
    }

    std::vector<IntersectionPrecomputes<float>> precs;
    precs.reserve( skyPatches.size() );
    for ( const auto & sp : skyPatches )
        precs.emplace_back( sp.dir );

    BitSetParallelFor( validSamples, [&]( VertId sampleVertId )
    {
        const auto samplePt = samples[sampleVertId];

        float totalRadiation = 0;
        for ( int i = 0; i < skyPatches.size(); ++i )
        {
            if ( !rayMeshIntersect( terrain, Line3f( samplePt, skyPatches[i].dir ), 0, FLT_MAX, &precs[i], false ) )
                totalRadiation += skyPatches[i].radiation;
        }
        res[sampleVertId] = rMaxRadiation * totalRadiation;
    } );

    return res;
}

BitSet findSkyRays( const Mesh & terrain,
    const VertCoords & samples, const VertBitSet & validSamples,
    const std::vector<SkyPatch> & skyPatches )
{
    MR_TIMER

    std::vector<IntersectionPrecomputes<float>> precs;
    precs.reserve( skyPatches.size() );
    for ( const auto & sp : skyPatches )
        precs.emplace_back( sp.dir );

    BitSet res( samples.size() * skyPatches.size() );
    BitSetParallelForAll( res, [&]( size_t ray )
    {
        const auto div = std::div( std::int64_t( ray ), std::int64_t( skyPatches.size() ) );
        const VertId sample( int( div.quot ) );
        if ( !validSamples.test( sample ) )
            return;
        const auto patch = div.rem;
        if ( !rayMeshIntersect( terrain, Line3f( samples[sample], skyPatches[patch].dir ), 0, FLT_MAX, &precs[patch], false ) )
            res.set( ray );
    } );

    return res;
}

} //namespace MR
