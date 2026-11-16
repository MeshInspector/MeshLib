// Ray-mesh intersection benchmark: MeshLib (MRMesh) vs Intel Embree 4.
// The same generated meshes and rays on every machine (portable hash RNG, no <random> distributions).
// usage: MREmbreeBench [reps=5] [raysPerSet=1000000]
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRBitSet.h"
#include <embree4/rtcore.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <functional>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace MR;

namespace
{

using Clock = std::chrono::steady_clock;

double secondsSince( Clock::time_point t0 )
{
    return std::chrono::duration<double>( Clock::now() - t0 ).count();
}

// best of reps
double bestOf( int reps, const std::function<void()>& f )
{
    double best = std::numeric_limits<double>::infinity();
    for ( int i = 0; i < reps; ++i )
    {
        auto t0 = Clock::now();
        f();
        best = std::min( best, secondsSince( t0 ) );
    }
    return best;
}

// splitmix64 -> uniform float in [0,1): identical on every platform and compiler
struct Rng
{
    uint64_t s;
    uint64_t next()
    {
        uint64_t z = ( s += 0x9E3779B97F4A7C15ull );
        z = ( z ^ ( z >> 30 ) ) * 0xBF58476D1CE4E5B9ull;
        z = ( z ^ ( z >> 27 ) ) * 0x94D049BB133111EBull;
        return z ^ ( z >> 31 );
    }
    float uni() { return float( next() >> 40 ) * ( 1.0f / 16777216.0f ); }
};

struct Rays
{
    std::string name;
    std::vector<Vector3f> org, dir;
    bool parallel = false; // all directions equal
};

Vector3f randomUnit( Rng& r )
{
    for ( ;; )
    {
        Vector3f v{ 2 * r.uni() - 1, 2 * r.uni() - 1, 2 * r.uni() - 1 };
        float l2 = v.lengthSq();
        if ( l2 > 1e-4f && l2 <= 1 )
            return v / std::sqrt( l2 );
    }
}

// orthographic camera grid looking along (-1,-1.3,-0.7), covering the bounding sphere
Rays makeCoherent( const Box3f& box, int n )
{
    Rays rs{ "coherent", {}, {}, true };
    const Vector3f c = box.center();
    const float rad = 0.5f * box.diagonal();
    const Vector3f d = Vector3f{ -1.f, -1.3f, -0.7f }.normalized();
    const Vector3f u = cross( d, Vector3f{ 0, 0, 1 } ).normalized();
    const Vector3f v = cross( u, d );
    const int side = int( std::lround( std::sqrt( double( n ) ) ) );
    rs.org.reserve( size_t( side ) * side );
    for ( int j = 0; j < side; ++j )
        for ( int i = 0; i < side; ++i )
        {
            float x = ( ( i + 0.5f ) / side * 2 - 1 ) * rad;
            float y = ( ( j + 0.5f ) / side * 2 - 1 ) * rad;
            rs.org.push_back( c - 2 * rad * d + x * u + y * v );
        }
    rs.dir.assign( rs.org.size(), d );
    return rs;
}

// origins uniform in the bounding box enlarged twice, directions uniform on the sphere
Rays makeRandom( const Box3f& box, int n, uint64_t seed )
{
    Rays rs{ "random", {}, {}, false };
    Rng r{ seed };
    const Vector3f c = box.center(), h = box.size();
    rs.org.resize( n );
    rs.dir.resize( n );
    for ( int i = 0; i < n; ++i )
    {
        rs.org[i] = c + Vector3f{ ( r.uni() - 0.5f ) * 2 * h.x, ( r.uni() - 0.5f ) * 2 * h.y, ( r.uni() - 0.5f ) * 2 * h.z };
        rs.dir[i] = randomUnit( r );
    }
    return rs;
}

std::atomic<int64_t> gEmbreeBytes{ 0 };
bool embreeMemoryMonitor( void*, ssize_t bytes, bool )
{
    gEmbreeBytes += bytes;
    return true;
}

RTCScene buildEmbree( RTCDevice dev, const Mesh& mesh )
{
    RTCScene scene = rtcNewScene( dev );
    RTCGeometry g = rtcNewGeometry( dev, RTC_GEOMETRY_TYPE_TRIANGLE );
    const auto& pts = mesh.points;
    float* vb = (float*)rtcSetNewGeometryBuffer( g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof( float ), pts.size() );
    std::memcpy( vb, pts.data(), pts.size() * sizeof( Vector3f ) );
    const size_t nf = mesh.topology.faceSize();
    unsigned* ib = (unsigned*)rtcSetNewGeometryBuffer( g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof( unsigned ), nf );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, nf ), [&]( const tbb::blocked_range<size_t>& r )
    {
        for ( size_t f = r.begin(); f < r.end(); ++f )
        {
            auto t = mesh.topology.getTriVerts( FaceId( int( f ) ) );
            ib[3 * f] = t[0];
            ib[3 * f + 1] = t[1];
            ib[3 * f + 2] = t[2];
        }
    } );
    rtcCommitGeometry( g );
    rtcAttachGeometry( scene, g );
    rtcReleaseGeometry( g );
    rtcCommitScene( scene );
    return scene;
}

struct Hits
{
    std::vector<float> t; // +inf if no hit
    std::vector<int> face;
};

void initRayHit( RTCRayHit& rh, const Vector3f& o, const Vector3f& d )
{
    rh.ray.org_x = o.x; rh.ray.org_y = o.y; rh.ray.org_z = o.z;
    rh.ray.dir_x = d.x; rh.ray.dir_y = d.y; rh.ray.dir_z = d.z;
    rh.ray.tnear = 0; rh.ray.tfar = std::numeric_limits<float>::infinity();
    rh.ray.mask = ~0u; rh.ray.flags = 0; rh.ray.time = 0; rh.ray.id = 0;
    rh.hit.geomID = RTC_INVALID_GEOMETRY_ID; rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
}

constexpr size_t kGrain = 1024;

void mlClosest( const Mesh& mesh, const Rays& rs, Hits& h )
{
    const size_t n = rs.org.size();
    std::optional<IntersectionPrecomputes<float>> prec;
    if ( rs.parallel )
        prec.emplace( rs.dir[0] );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, n, kGrain ), [&]( const tbb::blocked_range<size_t>& r )
    {
        for ( size_t i = r.begin(); i < r.end(); ++i )
        {
            auto res = rayMeshIntersect( mesh, Line3f( rs.org[i], rs.dir[i] ), 0.0f, FLT_MAX, prec ? &*prec : nullptr, true );
            h.t[i] = res ? res.distanceAlongLine : std::numeric_limits<float>::infinity();
            h.face[i] = res ? int( res.proj.face ) : -1;
        }
    } );
}

void mlMulti( const Mesh& mesh, const Rays& rs, Hits& h )
{
    MultiRayMeshIntersectResult res;
    res.rayDistances = &h.t;
    multiRayMeshIntersect( mesh, rs.org, rs.dir, res );
}

void mlAny( const Mesh& mesh, const Rays& rs, std::vector<uint8_t>& hit )
{
    const size_t n = rs.org.size();
    std::optional<IntersectionPrecomputes<float>> prec;
    if ( rs.parallel )
        prec.emplace( rs.dir[0] );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, n, kGrain ), [&]( const tbb::blocked_range<size_t>& r )
    {
        for ( size_t i = r.begin(); i < r.end(); ++i )
            hit[i] = rayMeshIntersect( mesh, Line3f( rs.org[i], rs.dir[i] ), 0.0f, FLT_MAX, prec ? &*prec : nullptr, false ).valid();
    } );
}

void emClosest1( RTCScene scene, const Rays& rs, Hits& h )
{
    const size_t n = rs.org.size();
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, n, kGrain ), [&]( const tbb::blocked_range<size_t>& r )
    {
        for ( size_t i = r.begin(); i < r.end(); ++i )
        {
            RTCRayHit rh;
            initRayHit( rh, rs.org[i], rs.dir[i] );
            rtcIntersect1( scene, &rh );
            const bool hit = rh.hit.geomID != RTC_INVALID_GEOMETRY_ID;
            h.t[i] = hit ? rh.ray.tfar : std::numeric_limits<float>::infinity();
            h.face[i] = hit ? int( rh.hit.primID ) : -1;
        }
    } );
}

void emClosest8( RTCScene scene, const Rays& rs, Hits& h )
{
    const size_t n = rs.org.size();
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, n, kGrain ), [&]( const tbb::blocked_range<size_t>& r )
    {
        for ( size_t b = r.begin(); b < r.end(); b += 8 )
        {
            alignas( 32 ) int valid[8];
            alignas( 32 ) RTCRayHit8 rh;
            for ( int k = 0; k < 8; ++k )
            {
                const size_t i = b + k;
                valid[k] = i < r.end() ? -1 : 0;
                const size_t j = std::min( i, r.end() - 1 );
                rh.ray.org_x[k] = rs.org[j].x; rh.ray.org_y[k] = rs.org[j].y; rh.ray.org_z[k] = rs.org[j].z;
                rh.ray.dir_x[k] = rs.dir[j].x; rh.ray.dir_y[k] = rs.dir[j].y; rh.ray.dir_z[k] = rs.dir[j].z;
                rh.ray.tnear[k] = 0; rh.ray.tfar[k] = std::numeric_limits<float>::infinity();
                rh.ray.mask[k] = ~0u; rh.ray.flags[k] = 0; rh.ray.time[k] = 0; rh.ray.id[k] = 0;
                rh.hit.geomID[k] = RTC_INVALID_GEOMETRY_ID; rh.hit.instID[0][k] = RTC_INVALID_GEOMETRY_ID;
            }
            rtcIntersect8( valid, scene, &rh );
            for ( int k = 0; k < 8 && b + k < r.end(); ++k )
            {
                const bool hit = rh.hit.geomID[k] != RTC_INVALID_GEOMETRY_ID;
                h.t[b + k] = hit ? rh.ray.tfar[k] : std::numeric_limits<float>::infinity();
                h.face[b + k] = hit ? int( rh.hit.primID[k] ) : -1;
            }
        }
    } );
}

void emAny( RTCScene scene, const Rays& rs, std::vector<uint8_t>& hit )
{
    const size_t n = rs.org.size();
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, n, kGrain ), [&]( const tbb::blocked_range<size_t>& r )
    {
        for ( size_t i = r.begin(); i < r.end(); ++i )
        {
            RTCRayHit rh;
            initRayHit( rh, rs.org[i], rs.dir[i] );
            rtcOccluded1( scene, &rh.ray );
            hit[i] = rh.ray.tfar < 0;
        }
    } );
}

struct Agreement
{
    size_t hitsA = 0, hitsB = 0, onlyOne = 0, sameFace = 0, farApart = 0;
    float maxRelDiff = 0;
};

Agreement compare( const Hits& a, const Hits& b, float scale )
{
    Agreement g;
    for ( size_t i = 0; i < a.t.size(); ++i )
    {
        const bool ha = std::isfinite( a.t[i] ), hb = std::isfinite( b.t[i] );
        g.hitsA += ha;
        g.hitsB += hb;
        if ( ha != hb )
        {
            ++g.onlyOne;
            continue;
        }
        if ( !ha )
            continue;
        g.sameFace += a.face[i] == b.face[i];
        const float d = std::abs( a.t[i] - b.t[i] ) / scale;
        g.maxRelDiff = std::max( g.maxRelDiff, d );
        g.farApart += d > 1e-4f;
    }
    return g;
}

struct Case
{
    std::string name;
    Mesh mesh;
};

} // namespace

int main( int argc, char** argv )
{
    const int reps = argc > 1 ? std::atoi( argv[1] ) : 5;
    const int nRays = argc > 2 ? std::atoi( argv[2] ) : 1000000;
    const int hw = int( std::thread::hardware_concurrency() );

    std::vector<Case> cases;
    cases.push_back( { "sphere20k", makeSphere( { .radius = 1, .numMeshVertices = 10000 } ) } );
    cases.push_back( { "sphere1M", makeSphere( { .radius = 1, .numMeshVertices = 500000 } ) } );
    cases.push_back( { "torus1M", makeTorus( 1, 0.3f, 1000, 500 ) } );

    RTCDevice dev = rtcNewDevice( nullptr );
    if ( !dev )
    {
        std::printf( "rtcNewDevice failed: %d\n", int( rtcGetDeviceError( nullptr ) ) );
        return 1;
    }
    rtcSetDeviceMemoryMonitorFunction( dev, embreeMemoryMonitor, nullptr );
    std::printf( "Embree %d.%d.%d, hardware threads %d, reps %d, rays per set %d\n",
        int( rtcGetDeviceProperty( dev, RTC_DEVICE_PROPERTY_VERSION_MAJOR ) ),
        int( rtcGetDeviceProperty( dev, RTC_DEVICE_PROPERTY_VERSION_MINOR ) ),
        int( rtcGetDeviceProperty( dev, RTC_DEVICE_PROPERTY_VERSION_PATCH ) ), hw, reps, nRays );
    std::printf( "Embree native packet support: 4=%d 8=%d 16=%d\n",
        int( rtcGetDeviceProperty( dev, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED ) ),
        int( rtcGetDeviceProperty( dev, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED ) ),
        int( rtcGetDeviceProperty( dev, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED ) ) );

    // RESULT lines: RESULT;threads;mesh;faces;metric;meshlib;embree   (times in ms, rates in Mrays/s, memory in MB)
    for ( int threads : { 1, hw } )
    {
        tbb::global_control gc( tbb::global_control::max_allowed_parallelism, threads );
        const int qreps = threads == 1 ? std::max( 1, reps / 2 ) : reps;
        for ( auto& c : cases )
        {
            Mesh& mesh = c.mesh;
            const size_t nf = mesh.topology.numValidFaces();
            const Box3f box = mesh.computeBoundingBox();
            auto out = [&]( const char* metric, double ml, double em )
            {
                std::printf( "RESULT;%d;%s;%zu;%s;%.3f;%.3f\n", threads, c.name.c_str(), nf, metric, ml, em );
                std::fflush( stdout );
            };

            // tree build
            double mlBuild = bestOf( reps, [&] { mesh.invalidateCaches(); (void)mesh.getAABBTree(); } );
            const double mlMB = mesh.getAABBTree().heapBytes() / 1048576.0;
            RTCScene scene = nullptr;
            double emBuild = bestOf( reps, [&]
            {
                if ( scene )
                    rtcReleaseScene( scene );
                gEmbreeBytes = 0;
                scene = buildEmbree( dev, mesh );
            } );
            const double emMB = gEmbreeBytes.load() / 1048576.0;
            out( "build_ms", mlBuild * 1e3, emBuild * 1e3 );
            if ( threads == hw )
                out( "tree_MB", mlMB, emMB );

            for ( const Rays& rs : { makeCoherent( box, nRays ), makeRandom( box, nRays, 12345 ) } )
            {
                const size_t n = rs.org.size();
                const double mr = n / 1e6;
                Hits hm{ std::vector<float>( n ), std::vector<int>( n ) }, he1 = hm, he8 = hm;
                std::vector<uint8_t> am( n ), ae( n );

                const double tMl = bestOf( qreps, [&] { mlClosest( mesh, rs, hm ); } );
                const double tEm1 = bestOf( qreps, [&] { emClosest1( scene, rs, he1 ); } );
                const double tEm8 = bestOf( qreps, [&] { emClosest8( scene, rs, he8 ); } );
                const double tMlAny = bestOf( qreps, [&] { mlAny( mesh, rs, am ); } );
                const double tEmAny = bestOf( qreps, [&] { emAny( scene, rs, ae ); } );
                Hits hmm{ std::vector<float>( n ), {} };
                const double tMlMulti = threads == hw ? bestOf( qreps, [&] { mlMulti( mesh, rs, hmm ); } ) : 0;

                const std::string p = rs.name + "_";
                out( ( p + "closest1_Mrays_s" ).c_str(), mr / tMl, mr / tEm1 );
                out( ( p + "closest_packet8_Mrays_s" ).c_str(), mr / tMl, mr / tEm8 );
                out( ( p + "any_Mrays_s" ).c_str(), mr / tMlAny, mr / tEmAny );
                if ( threads == hw )
                    out( ( p + "multiRay_vs_closest1_Mrays_s" ).c_str(), mr / tMlMulti, mr / tEm1 );

                if ( threads == hw )
                {
                    auto g = compare( hm, he1, box.diagonal() );
                    size_t anyDiff = 0, anyHitsM = 0;
                    for ( size_t i = 0; i < n; ++i )
                    {
                        anyDiff += am[i] != ae[i];
                        anyHitsM += am[i];
                    }
                    auto g8 = compare( he1, he8, box.diagonal() );
                    std::printf( "CHECK;%s;%s;hits ml=%zu em=%zu;only-one-hits=%zu;same-face=%zu;max|dt|/diag=%.2g;dt>1e-4=%zu;"
                                 "any-hit ml=%zu mismatches=%zu;embree 1 vs 8 differ=%zu\n",
                        c.name.c_str(), rs.name.c_str(), g.hitsA, g.hitsB, g.onlyOne, g.sameFace, g.maxRelDiff, g.farApart,
                        anyHitsM, anyDiff, g8.onlyOne + g8.farApart );
                    std::fflush( stdout );
                }
            }
            rtcReleaseScene( scene );
        }
    }
    rtcReleaseDevice( dev );
    return 0;
}
