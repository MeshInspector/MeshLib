// Benchmark of MeshLib's mesh AABBTree and the algorithms using it, for comparing two builds of MRMesh.
// usage: treebench [reps=5] [threads...]   (default threads: 1 and all)
// output lines: RES;threads;mesh;metric;value;unit
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRMeshProject.h"
#include "MRMesh/MRDirMax.h"
#include "MRMesh/MRBall.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRMeshCollidePrecise.h"
#include "MRMesh/MRPrecisePredicates3.h"
#include "MRMesh/MRMeshMeshDistance.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRLine3.h"
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace MR;

namespace
{

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

Vector3f randomSurfacePoint( const Mesh& m, Rng& r )
{
    const int nf = int( m.topology.faceSize() );
    for ( ;; )
    {
        const FaceId f( int( r.next() % uint64_t( nf ) ) );
        if ( !m.topology.hasFace( f ) )
            continue;
        const auto tv = m.topology.getTriVerts( f );
        float a = r.uni(), b = r.uni();
        if ( a + b > 1 ) { a = 1 - a; b = 1 - b; }
        return m.points[tv[0]] + a * ( m.points[tv[1]] - m.points[tv[0]] ) + b * ( m.points[tv[2]] - m.points[tv[0]] );
    }
}

// rays from random points in the box enlarged twice: half in random directions, half aimed at random surface points
void makeRays( const Mesh& m, int n, uint64_t seed, std::vector<Vector3f>& org, std::vector<Vector3f>& dir )
{
    Rng r{ seed };
    const Box3f box = m.computeBoundingBox();
    const Vector3f c = box.center(), h = box.size();
    org.resize( n );
    dir.resize( n );
    for ( int i = 0; i < n; ++i )
    {
        org[i] = c + Vector3f{ ( r.uni() - 0.5f ) * 2 * h.x, ( r.uni() - 0.5f ) * 2 * h.y, ( r.uni() - 0.5f ) * 2 * h.z };
        dir[i] = ( i % 2 ) ? ( randomSurfacePoint( m, r ) - org[i] ).normalized() : randomUnit( r );
    }
}

// points near the surface: random surface points moved by up to 2% of the diagonal
std::vector<Vector3f> nearPoints( const Mesh& m, int n, uint64_t seed )
{
    Rng r{ seed };
    const float off = 0.02f * m.computeBoundingBox().diagonal();
    std::vector<Vector3f> pts( n );
    for ( auto& p : pts )
        p = randomSurfacePoint( m, r ) + off * Vector3f( 2 * r.uni() - 1, 2 * r.uni() - 1, 2 * r.uni() - 1 );
    return pts;
}

using Clock = std::chrono::steady_clock;
double bestSec( int reps, const std::function<void()>& f )
{
    double best = 1e30;
    for ( int k = 0; k < reps; ++k )
    {
        auto t0 = Clock::now();
        f();
        best = std::min( best, std::chrono::duration<double>( Clock::now() - t0 ).count() );
    }
    return best;
}

template<typename F>
void parFor( size_t n, size_t grain, F&& f )
{
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, n, grain ), [&]( const tbb::blocked_range<size_t>& r )
    {
        for ( size_t i = r.begin(); i < r.end(); ++i )
            f( i );
    } );
}

std::atomic<uint64_t> gSink{ 0 }; // keeps the results from being optimized away

bool want( const char * metric )
{
    static const char * only = std::getenv( "TB_ONLY" ); // ';'-separated metrics, all if not set
    if ( !only )
        return true;
    const std::string list = std::string( ";" ) + only + ";";
    return list.find( std::string( ";" ) + metric + ";" ) != std::string::npos;
}

void out( int threads, const char* mesh, const char* metric, double value, const char* unit )
{
    std::printf( "RES;%d;%s;%s;%.4f;%s\n", threads, mesh, metric, value, unit );
    std::fflush( stdout );
}

} // namespace

int main( int argc, char** argv )
{
    const int reps = argc > 1 ? std::atoi( argv[1] ) : 5;
    std::vector<int> threadCounts;
    for ( int i = 2; i < argc; ++i )
        threadCounts.push_back( std::atoi( argv[i] ) );
    const int hw = int( std::thread::hardware_concurrency() );
    if ( threadCounts.empty() )
        threadCounts = { 1, hw };
    std::printf( "hardware threads %d, reps %d\n", hw, reps );

    struct Case { const char* name; Mesh mesh; };
    std::vector<Case> cases;
    cases.push_back( { "sphere1M", makeSphere( { .radius = 1, .numMeshVertices = 500000 } ) } );
    cases.push_back( { "torus1M", makeTorus( 1, 0.3f, 1000, 500 ) } );

    // pairs for collisions and distances, and two overlapping spheres for self-collisions
    Mesh ball = makeSphere( { .radius = 0.5f, .numMeshVertices = 500000 } );
    ball.transform( AffineXf3f::translation( Vector3f( 1.0f, 0.1f, 0.05f ) ) );
    Mesh farBall = ball;
    farBall.transform( AffineXf3f::translation( Vector3f( 0, 0, 1.0f ) ) );
    const AffineXf3f rigid( Matrix3f::rotation( Vector3f( 1, 2, 3 ).normalized(), 0.2f ), Vector3f( 0.02f, -0.03f, 0.01f ) );
    Mesh two = makeSphere( { .radius = 1, .numMeshVertices = 500000 } );
    {
        Mesh shifted = two;
        shifted.transform( AffineXf3f::translation( Vector3f( 0.7f, 0.2f, 0.1f ) ) );
        two.addMesh( shifted );
    }

    for ( int threads : threadCounts )
    {
        tbb::global_control gc( tbb::global_control::max_allowed_parallelism, threads );
        const int nRays = threads == 1 ? 200000 : 2000000;
        const int nPts = threads == 1 ? 50000 : 500000;
        for ( auto& c : cases )
        {
            Mesh& m = c.mesh;
            const char* name = c.name;

            // tree construction and updates
            if ( want( "build" ) ) out( threads, name, "build", 1e3 * bestSec( reps, [&] { m.invalidateCaches(); (void)m.getAABBTree(); } ), "ms" );
            if ( threads == threadCounts.front() )
                if ( want( "memory" ) ) out( threads, name, "memory", m.getAABBTree().heapBytes() / 1048576.0, "MB" );
            {
                VertBitSet some( m.topology.vertSize() );
                for ( VertId v : m.topology.getValidVerts() )
                    if ( int( v ) % 100 == 0 )
                        some.set( v );
                if ( want( "refit 1% verts" ) ) out( threads, name, "refit 1% verts", 1e3 * bestSec( reps, [&] { m.updateCaches( &some ); } ), "ms" );
                if ( want( "refit all verts" ) ) out( threads, name, "refit all verts", 1e3 * bestSec( reps, [&] { m.updateCaches( nullptr ); } ), "ms" );
            }
            (void)m.getAABBTree();

            // rays
            std::vector<Vector3f> org, dir;
            makeRays( m, nRays, 7, org, dir );
            const double mr = nRays / 1e6;
            if ( want( "ray closest" ) ) out( threads, name, "ray closest", mr / bestSec( reps, [&] {
                parFor( org.size(), 1024, [&]( size_t i ) { if ( rayMeshIntersect( m, Line3f( org[i], dir[i] ) ) ) ++gSink; } ); } ), "Mrays/s" );
            if ( want( "ray any" ) ) out( threads, name, "ray any", mr / bestSec( reps, [&] {
                parFor( org.size(), 1024, [&]( size_t i ) { if ( rayMeshIntersect( m, Line3f( org[i], dir[i] ), 0.0f, FLT_MAX, nullptr, false ) ) ++gSink; } ); } ), "Mrays/s" );
            {
                const int nAll = nRays / 10;
                if ( want( "ray all hits" ) ) out( threads, name, "ray all hits", nAll / 1e6 / bestSec( reps, [&] {
                    parFor( size_t( nAll ), 256, [&]( size_t i )
                    {
                        int k = 0;
                        rayMeshIntersectAll( m, Line3f( org[i], dir[i] ), [&k]( const MeshIntersectionResult& ) { ++k; return true; } );
                        gSink += k;
                    } ); } ), "Mrays/s" );
            }

            // projections and searches
            const auto pts = nearPoints( m, nPts, 8 );
            const double mp = nPts / 1e6;
            if ( want( "projection" ) ) out( threads, name, "projection", mp / bestSec( reps, [&] {
                parFor( pts.size(), 256, [&]( size_t i ) { gSink += findProjection( pts[i], m ).proj.face.valid(); } ); } ), "Mpts/s" );
            if ( want( "signed distance" ) ) out( threads, name, "signed distance", mp / bestSec( reps, [&] {
                parFor( pts.size(), 256, [&]( size_t i ) { gSink += findSignedDistance( pts[i], m ).has_value(); } ); } ), "Mpts/s" );
            const float ballR = 0.01f * m.computeBoundingBox().diagonal();
            if ( want( "tris in ball" ) ) out( threads, name, "tris in ball", mp / bestSec( reps, [&] {
                parFor( pts.size(), 256, [&]( size_t i )
                {
                    int k = 0;
                    findTrisInBall( m, Ball3f( pts[i], ballR * ballR ), [&k]( const MeshProjectionResult&, Ball3f& ) { ++k; return Processing::Continue; } );
                    gSink += k;
                } ); } ), "Mpts/s" );
            {
                std::vector<Vector3f> dirs( nPts );
                Rng r{ 5 };
                for ( auto& d : dirs )
                    d = randomUnit( r );
                if ( want( "findDirMax" ) ) out( threads, name, "findDirMax", mp / bestSec( reps, [&] {
                    parFor( dirs.size(), 256, [&]( size_t i ) { gSink += int( findDirMax( dirs[i], m ) ); } ); } ), "Mdirs/s" );
            }
            {
                const int nPlanes = threads == 1 ? 50 : 500;
                std::vector<Plane3f> planes( nPlanes );
                Rng r{ 11 };
                const Box3f box = m.computeBoundingBox();
                for ( auto& p : planes )
                    p = Plane3f::fromDirAndPt( randomUnit( r ), box.center() + 0.3f * ( r.uni() - 0.5f ) * box.size() );
                if ( want( "plane sections" ) ) out( threads, name, "plane sections", nPlanes / bestSec( reps, [&] {
                    parFor( planes.size(), 1, [&]( size_t i )
                    {
                        std::vector<FaceId> fs;
                        planeMeshIntersect( m, planes[i], nullptr, nullptr, nullptr, &fs );
                        gSink += fs.size();
                    } ); } ), "planes/s" );
            }
            {
                // the winding number (dipoles are built once, as by the mesh)
                (void)m.getDipoles();
                const int nW = nPts;
                if ( want( "winding number" ) ) out( threads, name, "winding number", nW / 1e6 / bestSec( reps, [&] {
                    parFor( size_t( nW ), 256, [&]( size_t i ) { gSink += m.calcFastWindingNumber( pts[i] ) > 0.5f; } ); } ), "Mpts/s" );
            }
        }

        // two meshes
        (void)ball.getAABBTree();
        (void)farBall.getAABBTree();
        (void)two.getAABBTree();
        Mesh& torus = cases[1].mesh;
        if ( want( "collide" ) ) out( threads, "torus1M+sphere1M", "collide", 1e3 * bestSec( reps, [&] { gSink += findCollidingTriangles( torus, ball ).size(); } ), "ms" );
        if ( want( "collide rigid" ) ) out( threads, "torus1M+sphere1M", "collide rigid", 1e3 * bestSec( reps, [&] { gSink += findCollidingTriangles( torus, ball, &rigid ).size(); } ), "ms" );
        if ( want( "collide first" ) ) out( threads, "torus1M+sphere1M", "collide first", 1e3 * bestSec( reps, [&] { gSink += findCollidingTriangles( torus, ball, &rigid, true ).size(); } ), "ms" );
        const auto conv = getVectorConverters( MeshPart( torus ), MeshPart( ball ) );
        if ( want( "collide precise" ) ) out( threads, "torus1M+sphere1M", "collide precise", 1e3 * bestSec( reps, [&] { gSink += findCollidingEdgeTrisPrecise( torus, ball, conv.toInt ).size(); } ), "ms" );
        if ( want( "collide precise rigid" ) ) out( threads, "torus1M+sphere1M", "collide precise rigid", 1e3 * bestSec( reps, [&] { gSink += findCollidingEdgeTrisPrecise( torus, ball, conv.toInt, &rigid ).size(); } ), "ms" );
        if ( want( "distance" ) ) out( threads, "torus1M+sphere1M", "distance", 1e3 * bestSec( reps, [&] { gSink += findDistance( torus, farBall ).distSq > 0; } ), "ms" );
        if ( want( "distance rigid" ) ) out( threads, "torus1M+sphere1M", "distance rigid", 1e3 * bestSec( reps, [&] { gSink += findDistance( torus, farBall, &rigid ).distSq > 0; } ), "ms" );
        if ( want( "self-collide" ) ) out( threads, "2 spheres 2M", "self-collide", 1e3 * bestSec( reps, [&] { gSink += findSelfCollidingTriangles( two )->size(); } ), "ms" );
        const auto conv2 = getVectorConverters( MeshPart( two ) );
        if ( want( "self-collide precise" ) ) out( threads, "2 spheres 2M", "self-collide precise", 1e3 * bestSec( reps, [&] { gSink += findSelfCollidingEdgeTrisPrecise( two, conv2.toInt ).size(); } ), "ms" );
    }
    std::printf( "sink %llu\n", (unsigned long long)gSink.load() );
    return 0;
}
