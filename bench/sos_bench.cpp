// Throwaway benchmark: cost of one simulation-of-simplicity tie resolution in MR::inSphere.
// Every timed call places 4 lattice points exactly on the sphere with rSq == R^2, so the primary
// predicate always answers OnSphere and the whole cost is the SoS tie-break (the code under test).
#include "MRMesh/MRInSphere.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace MR;

namespace
{

// all lattice points with x^2+y^2+z^2 == r2 and |coord| <= lim
std::vector<Vector3i> onSphere( int64_t r2, int lim )
{
    std::vector<Vector3i> res;
    for ( int x = -lim; x <= lim; ++x )
        for ( int y = -lim; y <= lim; ++y )
            for ( int z = -lim; z <= lim; ++z )
                if ( int64_t( x ) * x + int64_t( y ) * y + int64_t( z ) * z == r2 )
                    res.push_back( Vector3i{ x, y, z } );
    return res;
}

using Quad = std::array<Vector3i, 4>;

// quadruples of distinct points on the sphere of squared radius r2 centered at the origin, ordered so
// that the center is on the positive side of the plane (a,b,c), and verified to make the primary
// predicate answer exactly OnSphere - i.e. every such quadruple forces the SoS path
std::vector<Quad> tieQuads( int64_t r2, int lim, size_t maxQuads )
{
    const auto pts = onSphere( r2, lim );
    std::vector<Quad> res;
    for ( size_t i = 0; i < pts.size() && res.size() < maxQuads; ++i )
        for ( size_t j = 0; j < pts.size() && res.size() < maxQuads; ++j )
            for ( size_t k = 0; k < pts.size() && res.size() < maxQuads; ++k )
                for ( size_t l = 0; l < pts.size() && res.size() < maxQuads; ++l )
                {
                    if ( i == j || i == k || i == l || j == k || j == l || k == l )
                        continue;
                    Quad q{ pts[i], pts[j], pts[k], pts[l] };
                    // the center (origin) must be on the positive side of the plane (a,b,c)
                    if ( dot( cross( Vector3i64{ q[1] - q[0] }, Vector3i64{ q[2] - q[0] } ), Vector3i64{ -q[0] } ) <= 0 )
                        std::swap( q[1], q[2] );
                    if ( inSphere( q[0], q[1], q[2], q[3], r2 ) != InSphereResult::OnSphere )
                        continue; // not a tie: skip, we only time the tie-break
                    res.push_back( q );
                }
    return res;
}

const int perms[24][4] = {
    {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
    {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
    {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
    {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0} };

struct Res { double usPerCall; uint64_t checksum; int64_t calls; };

Res timeMag( const std::vector<Quad> & base, int64_t r2, int scale, int64_t targetCalls, int runs )
{
    // scale the whole configuration: s*p lies on the sphere of squared radius s^2*r2
    std::vector<Quad> qs = base;
    const int64_t rSq = int64_t( scale ) * scale * r2;
    for ( auto & q : qs )
        for ( auto & p : q )
            p = Vector3i{ p.x * scale, p.y * scale, p.z * scale };

    double best = 1e300;
    uint64_t checksum = 0;
    int64_t calls = 0;
    for ( int r = 0; r < runs; ++r )
    {
        uint64_t sum = 0;
        int64_t n = 0;
        const auto t0 = std::chrono::steady_clock::now();
        while ( n < targetCalls )
            for ( const auto & q : qs )
                for ( const auto & pm : perms )
                {
                    const std::array<PreciseVertCoords, 4> vs{
                        PreciseVertCoords{ VertId( pm[0] ), q[0] }, PreciseVertCoords{ VertId( pm[1] ), q[1] },
                        PreciseVertCoords{ VertId( pm[2] ), q[2] }, PreciseVertCoords{ VertId( pm[3] ), q[3] } };
                    sum = sum * 3 + uint64_t( int( inSphere( vs, rSq ) ) );
                    ++n;
                }
        const double sec = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
        if ( sec * 1e6 / double( n ) < best )
            best = sec * 1e6 / double( n );
        checksum = sum;
        calls = n;
    }
    return { best, checksum, calls };
}

} // anonymous namespace

int main( int argc, char ** argv )
{
    const int runs = argc > 1 ? std::atoi( argv[1] ) : 3;
    const int64_t targetCalls = argc > 2 ? std::atoll( argv[2] ) : 150000;

    const int64_t r2 = 9; // sphere x^2+y^2+z^2 == 9: 30 lattice points, plenty of tie quadruples
    const std::vector<Quad> base = tieQuads( r2, 3, 4 );
    if ( base.empty() )
    {
        printf( "FAILED: no tie quadruples found\n" );
        return 1;
    }
    int maxCoord = 0;
    for ( const auto & q : base )
        for ( const auto & p : q )
            for ( int i = 0; i < 3; ++i )
                maxCoord = std::max( maxCoord, std::abs( p[i] ) );

    printf( "quads=%zu  base r2=%lld  base max coord=%d  runs=%d  calls/run>=%lld\n",
        base.size(), (long long)r2, maxCoord, runs, (long long)targetCalls );
    printf( "%-14s %-12s %-12s %s\n", "scale", "us/call", "calls", "checksum" );
    for ( int scale : { 1, 32, 1024, 32768, 1048576, 268435456 } )
    {
        const auto res = timeMag( base, r2, scale, targetCalls, runs );
        const double lg = std::log2( double( scale ) * maxCoord );
        printf( "2^%-11.1f %-12.4f %-12lld %llu\n", lg, res.usPerCall, (long long)res.calls, (unsigned long long)res.checksum );
        fflush( stdout );
    }
    return 0;
}
