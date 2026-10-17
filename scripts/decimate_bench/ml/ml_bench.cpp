// MeshLib side of the decimation comparison, run as in examples/cpp-examples/MeshDecimate.dox.cpp:
// packOptimally( false ) (no AABB tree, as in the MeshDecimate example), then decimateMesh with subdivideParts parts and a target face count.
//   ml_bench time <mesh> <parts> <ratio> <reps>       -> CSV: pack_ms,decimate_ms,faces   (minimum over reps)
//   ml_bench save <mesh> <parts> <ratio> <out>         -> decimates once and saves the result
//   ml_bench eval <orig> <result>                      -> CSV of quality metrics of result against orig
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRMeshDecimate.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRMeshProject.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRExpected.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace MR;

static Mesh load( const std::string& p )
{
    auto m = MeshLoad::fromAnySupportedFormat( p );
    if ( !m )
    {
        std::fprintf( stderr, "load %s: %s\n", p.c_str(), m.error().c_str() );
        std::exit( 1 );
    }
    return std::move( *m );
}

static double msSince( std::chrono::steady_clock::time_point t0 )
{
    return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
}

static DecimateSettings settingsFor( const Mesh& mesh, int parts, float ratio )
{
    DecimateSettings s;
    s.maxError = FLT_MAX; // stop only at the target face count, like trueform's ratio
    s.maxDeletedFaces = int( mesh.topology.numValidFaces() * ( 1 - ratio ) );
    s.subdivideParts = parts;
    return s;
}

// mean and max distance from valid vertices of a to the surface of b
static void oneSided( const Mesh& a, const Mesh& b, double& mean, float& maxD )
{
    std::vector<float> d( a.points.size(), 0.f );
    ParallelFor( a.points, [&]( VertId v )
    {
        if ( a.topology.hasVert( v ) )
            d[v] = std::sqrt( findProjection( a.points[v], b ).distSq );
    } );
    double s = 0;
    maxD = 0;
    for ( auto v : a.topology.getValidVerts() )
    {
        s += d[v];
        maxD = std::max( maxD, d[v] );
    }
    mean = s / a.topology.numValidVerts();
}

static int eval( const Mesh& orig, const Mesh& m )
{
    const float diag = orig.computeBoundingBox().diagonal();
    (void)orig.getAABBTree();
    (void)m.getAABBTree();
    const auto& faces = m.topology.getValidFaces();
    std::atomic<int> ar20{ 0 }, ar100{ 0 }, ar1000{ 0 }, zeroArea{ 0 }, flipped{ 0 }, off60{ 0 };
    Vector<float, FaceId> ar( m.topology.faceSize(), 0.f );
    const float tinyArea = 1e-12f * diag * diag;
    ParallelFor( ar, [&]( FaceId f )
    {
        if ( !faces.test( f ) )
            return;
        const float a = m.triangleAspectRatio( f );
        ar[f] = std::isfinite( a ) ? a : FLT_MAX;
        if ( a > 20 ) ++ar20;
        if ( a > 100 ) ++ar100;
        if ( a > 1000 ) ++ar1000;
        if ( m.area( f ) <= tinyArea ) ++zeroArea;
        // orientation against the original surface under the triangle's center
        const auto proj = findProjection( m.triCenter( f ), orig );
        const float c = dot( m.normal( f ), orig.normal( proj.proj.face ) );
        if ( c < 0 ) ++flipped;
        if ( c < 0.5f ) ++off60;
    } );
    float maxAr = 0;
    for ( auto f : faces )
        maxAr = std::max( maxAr, ar[f] );
    auto self = findSelfCollidingTriangles( m );
    double meanAB, meanBA;
    float maxAB, maxBA;
    oneSided( m, orig, meanAB, maxAB );
    oneSided( orig, m, meanBA, maxBA );
    std::printf( "faces,holes,self_intersecting_pairs,zero_area,aspect_gt20,aspect_gt100,aspect_gt1000,aspect_max,"
        "flipped,normal_off_gt60,dist_to_orig_mean,dist_to_orig_max,dist_from_orig_mean,dist_from_orig_max\n" );
    std::printf( "%d,%d,%lld,%d,%d,%d,%d,%.3g,%d,%d,%.3g,%.3g,%.3g,%.3g\n", m.topology.numValidFaces(), m.topology.findNumHoles(),
        self ? (long long)self->size() : -1LL, zeroArea.load(), ar20.load(), ar100.load(), ar1000.load(), maxAr,
        flipped.load(), off60.load(), meanAB / diag, maxAB / diag, meanBA / diag, maxBA / diag );
    return 0;
}

int main( int argc, char** argv )
{
    if ( argc < 4 )
    {
        std::fprintf( stderr, "usage: ml_bench time|save|eval ...\n" );
        return 1;
    }
    const std::string mode = argv[1];
    const Mesh src = load( argv[2] );
    if ( mode == "eval" )
        return eval( src, load( argv[3] ) );

    const int parts = std::atoi( argv[3] );
    const float ratio = float( std::atof( argv[4] ) );
    if ( mode == "save" )
    {
        Mesh m = src;
        if ( parts > 1 )
            m.packOptimally( false );
        auto s = settingsFor( m, parts, ratio );
        s.packMesh = true;
        decimateMesh( m, s );
        if ( auto r = MeshSave::toAnySupportedFormat( m, argv[5] ); !r )
        {
            std::fprintf( stderr, "save: %s\n", r.error().c_str() );
            return 1;
        }
        return 0;
    }

    const int reps = std::atoi( argv[5] );
    double bestPack = 1e30, bestDec = 1e30;
    int faces = 0;
    for ( int i = 0; i < reps; ++i )
    {
        Mesh m = src;
        auto t0 = std::chrono::steady_clock::now();
        if ( parts > 1 && !std::getenv( "ML_BENCH_NO_PACK" ) )
            m.packOptimally( false );
        bestPack = std::min( bestPack, msSince( t0 ) );
        const auto s = settingsFor( m, parts, ratio );
        t0 = std::chrono::steady_clock::now();
        decimateMesh( m, s );
        bestDec = std::min( bestDec, msSince( t0 ) );
        faces = m.topology.numValidFaces();
    }
    std::printf( "%.2f,%.2f,%d\n", parts > 1 ? bestPack : 0.0, bestDec, faces );
    return 0;
}
