#include <MRCuda/MRCudaSolarRadiation.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshIntersect.h>
#include <MRMesh/MRSolarRadiation.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace MR;

namespace
{

// a square terrain with a hill in the middle
Mesh makeHillTerrain( int n )
{
    VertCoords pts;
    for ( int y = 0; y <= n; ++y )
        for ( int x = 0; x <= n; ++x )
        {
            const float fx = float( x ) / n - 0.5f, fy = float( y ) / n - 0.5f;
            pts.push_back( Vector3f( fx, fy, 0.4f * std::exp( -( fx * fx + fy * fy ) * 20 ) ) );
        }
    Triangulation t;
    auto id = [n] ( int x, int y ) { return VertId( y * ( n + 1 ) + x ); };
    for ( int y = 0; y < n; ++y )
        for ( int x = 0; x < n; ++x )
        {
            t.push_back( { id( x, y ), id( x + 1, y ), id( x + 1, y + 1 ) } );
            t.push_back( { id( x, y ), id( x + 1, y + 1 ), id( x, y + 1 ) } );
        }
    return Mesh::fromTriangles( std::move( pts ), t );
}

} // namespace

// the GPU versions must report the same sky rays and sky view factors as the CPU versions
TEST( MRCuda, SolarRadiation )
{
    constexpr int n = 40;
    const auto terrain = makeHillTerrain( n );

    std::vector<SkyPatch> patches;
    for ( const auto& dir : sampleHalfSphere() )
        patches.push_back( { dir, 1.0f } );

    VertCoords samples = terrain.points;
    for ( auto& p : samples )
        p.z += 1e-3f;
    VertBitSet validSamples( samples.size() );
    for ( int y = 0; y <= n; y += 2 )
        for ( int x = 0; x <= n; x += 2 )
            validSamples.set( VertId( y * ( n + 1 ) + x ) );

    const auto cpuRays = findSkyRays( terrain, samples, validSamples, patches );
    EXPECT_GT( cpuRays.count(), 0 );

    std::vector<MeshIntersectionResult> gpuInters;
    auto gpuRays = Cuda::findSkyRays( terrain, samples, validSamples, patches, &gpuInters );
    gpuRays.resize( cpuRays.size() );
    EXPECT_EQ( ( gpuRays ^ cpuRays ).count(), 0 );
    ASSERT_EQ( gpuInters.size(), cpuRays.size() );
    for ( size_t i = 0; i < cpuRays.size(); ++i )
    {
        const bool valid = validSamples.test( VertId( i / patches.size() ) );
        // an intersection is reported for every valid ray hitting the terrain, and only for them
        ASSERT_EQ( bool( gpuInters[i] ), valid && !cpuRays.test( i ) ) << "ray " << i;
    }

    BitSet cpuSkyRays;
    const auto cpuFactors = computeSkyViewFactor( terrain, samples, validSamples, patches, &cpuSkyRays );
    for ( int mode = 0; mode < 3; ++mode )
    {
        BitSet gpuSkyRays;
        std::vector<MeshIntersectionResult> inters;
        const auto gpuFactors = Cuda::computeSkyViewFactor( terrain, samples, validSamples, patches,
            mode == 1 ? &gpuSkyRays : nullptr, mode == 2 ? &inters : nullptr );
        for ( auto v : validSamples )
            ASSERT_NEAR( gpuFactors[v], cpuFactors[v], 1e-6f ) << "mode " << mode << ", vertex " << v;
        if ( mode == 1 )
        {
            gpuSkyRays.resize( cpuSkyRays.size() );
            EXPECT_EQ( ( gpuSkyRays ^ cpuSkyRays ).count(), 0 );
        }
    }
}
