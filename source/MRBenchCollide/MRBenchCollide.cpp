// Benchmark of findCollidingEdgeTrisPrecise, findSelfCollidingEdgeTrisPrecise, boolean and selfBoolean
// on procedural meshes, mirroring the measurements of MeshLib PR #7168.
// Prints lines "T <case> <ms>" (best time) and "H <case> <hash>" (result hash for bit-identity checks).

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRMeshBoolean.h"
#include "MRMesh/MRMeshCollidePrecise.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRBuffer.h"
#include <tbb/global_control.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
extern "C" __declspec( dllimport ) const char * mrBenchCollideVariant();
#else
extern "C" const char * mrBenchCollideVariant();
#endif

using namespace MR;

namespace
{

bool gQuick = false;
std::string gFilter;

std::uint64_t mix( std::uint64_t h, std::uint64_t v )
{
    v += 0x9e3779b97f4a7c15ull + ( h << 6 ) + ( h >> 2 );
    v ^= v >> 30; v *= 0xbf58476d1ce4e5b9ull;
    v ^= v >> 27; v *= 0x94d049bb133111ebull;
    v ^= v >> 31;
    return h ^ v;
}

std::uint64_t floatBits( float f )
{
    std::uint32_t u;
    std::memcpy( &u, &f, 4 );
    return u;
}

/// deterministic value in [-1, 1] for index i, the same on all platforms
double hashNoise( std::uint64_t i, std::uint64_t salt )
{
    const auto h = mix( salt, i );
    return double( h >> 11 ) / double( 1ull << 52 ) - 1.0;
}

std::uint64_t hashResult( PreciseCollisionResult r )
{
    std::sort( r.begin(), r.end(), []( const VarEdgeTri & a, const VarEdgeTri & b )
    {
        if ( a.edge != b.edge ) return a.edge < b.edge;
        if ( a.tri() != b.tri() ) return a.tri() < b.tri();
        return a.isEdgeATriB() < b.isEdgeATriB();
    } );
    std::uint64_t h = r.size();
    for ( const auto & x : r )
        h = mix( mix( mix( h, (int)x.edge ), (int)x.tri() ), x.isEdgeATriB() );
    return h;
}

std::uint64_t hashResult( std::vector<EdgeTri> r )
{
    std::sort( r.begin(), r.end(), []( const EdgeTri & a, const EdgeTri & b )
    {
        if ( a.edge != b.edge ) return a.edge < b.edge;
        return a.tri < b.tri;
    } );
    std::uint64_t h = r.size();
    for ( const auto & x : r )
        h = mix( mix( h, (int)x.edge ), (int)x.tri );
    return h;
}

std::uint64_t hashMesh( const Mesh & m )
{
    std::uint64_t h = mix( m.topology.numValidVerts(), m.topology.numValidFaces() );
    for ( auto v : m.topology.getValidVerts() )
    {
        const auto & p = m.points[v];
        h = mix( mix( mix( mix( h, (int)v ), floatBits( p.x ) ), floatBits( p.y ) ), floatBits( p.z ) );
    }
    for ( auto f : m.topology.getValidFaces() )
    {
        const auto vs = m.topology.getTriVerts( f );
        h = mix( mix( mix( mix( h, (int)f ), (int)vs[0] ), (int)vs[1] ), (int)vs[2] );
    }
    return h;
}

std::uint64_t hashBoolean( const BooleanResult & r )
{
    if ( !r.valid() )
        std::fprintf( stderr, "boolean failed: %s\n", r.errorString.c_str() );
    std::uint64_t h = std::hash<std::string>{}( r.errorString );
    return r.valid() ? mix( h, hashMesh( r.mesh ) ) : h;
}

double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>( steady_clock::now().time_since_epoch() ).count();
}

bool selected( const std::string & name )
{
    if ( gFilter.empty() )
        return true;
    // comma-separated substrings
    size_t pos = 0;
    while ( pos <= gFilter.size() )
    {
        const auto end = std::min( gFilter.find( ',', pos ), gFilter.size() );
        const auto part = gFilter.substr( pos, end - pos );
        if ( !part.empty() && name.find( part ) != std::string::npos )
            return true;
        pos = end + 1;
    }
    return false;
}

/// runs f() `runs` times, prints the best time and the hash returned by the first run
void measure( const std::string & name, int runs, const std::function<std::uint64_t()> & f )
{
    if ( !selected( name ) )
        return;
    double best = 1e300;
    std::uint64_t hash = 0;
    for ( int i = 0; i < runs; ++i )
    {
        const auto t0 = nowMs();
        const auto h = f();
        const auto t = nowMs() - t0;
        best = std::min( best, t );
        if ( i == 0 )
            hash = h;
        else if ( h != hash )
            std::printf( "W %s nondeterministic\n", name.c_str() );
    }
    std::printf( "T %s %.4f\nH %s %016llx\n", name.c_str(), best, name.c_str(), (unsigned long long)hash );
    std::fflush( stdout );
}

/// makes a torus with roughly square cells and `verts` vertices
Mesh makeTorusMesh( int primaryRes, int secondaryRes )
{
    auto m = makeTorus( 1.0f, 0.3f, primaryRes, secondaryRes );
    m.packOptimally( false );
    return m;
}

void addNoise( Mesh & m, float amplitude, std::uint64_t salt )
{
    const auto normals = [&]
    {
        std::vector<Vector3f> res( m.points.size() );
        BitSetParallelFor( m.topology.getValidVerts(), [&]( VertId v ) { res[(int)v] = m.normal( v ); } );
        return res;
    }();
    BitSetParallelFor( m.topology.getValidVerts(), [&]( VertId v )
    {
        m.points[v] += float( amplitude * hashNoise( (int)v, salt ) ) * normals[(int)v];
    } );
    m.invalidateCaches();
}

/// a rigid transformation: rotation by `angle` around `axis` passing through `center`
AffineXf3f rotationAround( const Vector3f & axis, float angle, const Vector3f & center )
{
    return AffineXf3f::xfAround( Matrix3f::rotation( axis.normalized(), angle ), center );
}

/// golden-spiral direction number i of n
Vector3f spiralDir( int i, int n )
{
    const double golden = 3.14159265358979323846 * ( 3.0 - std::sqrt( 5.0 ) );
    const double z = 1.0 - ( 2.0 * i + 1.0 ) / n;
    const double r = std::sqrt( std::max( 0.0, 1.0 - z * z ) );
    const double phi = golden * i;
    return Vector3f( float( r * std::cos( phi ) ), float( r * std::sin( phi ) ), float( z ) );
}

/// pose number i of n for the second sphere: a rotation and a shift so that the spheres intersect
AffineXf3f spherePose( int i, int n )
{
    const auto axis = spiralDir( ( i * 7 + 3 ) % n, n );
    const float angle = float( 2 * 3.14159265358979323846 * std::fmod( i * 0.6180339887498949, 1.0 ) );
    const auto shift = 0.6f * spiralDir( i, n );
    return AffineXf3f::translation( shift ) * AffineXf3f::linear( Matrix3f::rotation( axis, angle ) );
}

struct Pair
{
    std::string name;
    Mesh a, b;
};

/// two intersecting tori; b is already placed in the space of a
Pair makePair( const std::string & name, int aP, int aS, int bP, int bS )
{
    Pair res{ name, makeTorusMesh( aP, aS ), makeTorusMesh( bP, bS ) };
    res.b.transform( AffineXf3f::translation( Vector3f( 0.45f, 0.1f, 0.05f ) ) * AffineXf3f::linear( Matrix3f::rotation( Vector3f( 1, 0.2f, 0 ).normalized(), 0.6f ) ) );
    std::fprintf( stderr, "%s: %d + %d vertices, %d + %d triangles\n", name.c_str(),
        res.a.topology.numValidVerts(), res.b.topology.numValidVerts(), res.a.topology.numValidFaces(), res.b.topology.numValidFaces() );
    return res;
}

Mesh merged( const Pair & p )
{
    Mesh m = p.a;
    m.addMesh( p.b );
    m.packOptimally( false );
    return m;
}

void collide( const Pair & p, const AffineXf3f * xf, const std::string & suffix, int runsPar, int runsSingle )
{
    p.a.getAABBTree();
    p.b.getAABBTree();
    const auto conv = getVectorConverters( p.a, p.b, xf ).toInt;
    auto f = [&] { return hashResult( findCollidingEdgeTrisPrecise( p.a, p.b, conv, xf ) ); };
    measure( "collide/" + p.name + suffix + "/parallel", runsPar, f );
    tbb::global_control gc( tbb::global_control::max_allowed_parallelism, 1 );
    measure( "collide/" + p.name + suffix + "/single", runsSingle, f );
}

void selfCollide( const std::string & name, const Mesh & m, const AffineXf3f * xf, int runsPar, int runsSingle )
{
    m.getAABBTree();
    const auto conv = xf ? getVectorConverters( m, m, xf ).toInt : getVectorConverters( m ).toInt;
    auto f = [&] { return hashResult( findSelfCollidingEdgeTrisPrecise( m, conv, false, xf ) ); };
    measure( "self/" + name + "/parallel", runsPar, f );
    tbb::global_control gc( tbb::global_control::max_allowed_parallelism, 1 );
    measure( "self/" + name + "/single", runsSingle, f );
}

void booleans( const Pair & p, const AffineXf3f * xf, const std::string & suffix, const std::vector<std::pair<BooleanOperation, const char*>> & ops, int runs )
{
    p.a.getAABBTree();
    p.b.getAABBTree();
    for ( const auto & [op, opName] : ops )
    {
        measure( "boolean/" + p.name + suffix + "/" + opName, runs, [&]
        {
            BooleanParameters params;
            params.rigidB2A = xf;
            return hashBoolean( boolean( p.a, p.b, op, params ) );
        } );
    }
}

} // namespace

int main( int argc, char ** argv )
{
    for ( int i = 1; i < argc; ++i )
    {
        if ( !std::strcmp( argv[i], "--quick" ) )
            gQuick = true;
        else if ( !std::strcmp( argv[i], "--filter" ) && i + 1 < argc )
            gFilter = argv[++i];
    }
    std::printf( "V %s\n", mrBenchCollideVariant() );
    std::fflush( stdout );

    // sizes as in #7168: "dental" ~146K + 140K vertices, "big" ~1M + 1M vertices
    const int scale = gQuick ? 4 : 1;
    const int sphereN = gQuick ? 50 : 2000;
    const int sphereSingleN = gQuick ? 20 : 500;
    const int r = gQuick ? 1 : 0; // fewer repetitions in quick mode
    auto runs = [r]( int n ) { return r ? 1 : n; };

    const auto medium = makePair( "medium", 700 / scale, 209 / scale, 684 / scale, 205 / scale );
    const auto big = makePair( "big", 1826 / scale, 548 / scale, 1824 / scale, 548 / scale );
    auto noisy = big;
    noisy.name = "big-noise";
    addNoise( noisy.a, 0.25f * 0.0034f * scale, 1 );
    addNoise( noisy.b, 0.25f * 0.0034f * scale, 2 );

    const auto rotMedium = rotationAround( Vector3f( 0.3f, 0.5f, 1 ), 0.01f, medium.b.computeBoundingBox().center() );
    const auto rotBig = rotationAround( Vector3f( 0.3f, 0.5f, 1 ), 0.01f, big.b.computeBoundingBox().center() );

    const auto sphere = makeSphere( { .radius = 1.0f, .numMeshVertices = 3366 } );
    sphere.getAABBTree();
    std::fprintf( stderr, "sphere: %d vertices, %d triangles\n", sphere.topology.numValidVerts(), sphere.topology.numValidFaces() );

    // findCollidingEdgeTrisPrecise
    collide( medium, nullptr, "", runs( 10 ), runs( 5 ) );
    collide( big, nullptr, "", runs( 10 ), runs( 3 ) );
    collide( noisy, nullptr, "", runs( 5 ), runs( 2 ) );
    collide( medium, &rotMedium, "-rot", runs( 10 ), runs( 5 ) );
    collide( big, &rotBig, "-rot", runs( 10 ), runs( 3 ) );
    {
        auto spheres = [&]( int n )
        {
            std::uint64_t h = 0;
            for ( int i = 0; i < n; ++i )
            {
                const auto xf = spherePose( i, sphereN );
                const auto c = getVectorConverters( sphere, sphere, &xf ).toInt;
                h = mix( h, hashResult( findCollidingEdgeTrisPrecise( sphere, sphere, c, &xf ) ) );
            }
            return h;
        };
        measure( "collide/spheres-x" + std::to_string( sphereN ) + "/parallel", runs( 3 ), [&] { return spheres( sphereN ); } );
        tbb::global_control gc( tbb::global_control::max_allowed_parallelism, 1 );
        measure( "collide/spheres-x" + std::to_string( sphereSingleN ) + "/single", 1, [&] { return spheres( sphereSingleN ); } );
    }

    // whole boolean
    const std::vector<std::pair<BooleanOperation, const char*>> threeOps = {
        { BooleanOperation::Union, "Union" },
        { BooleanOperation::Intersection, "Intersection" },
        { BooleanOperation::DifferenceAB, "DifferenceAB" } };
    const std::vector<std::pair<BooleanOperation, const char*>> unionOnly = { { BooleanOperation::Union, "Union" } };
    booleans( medium, nullptr, "", threeOps, runs( 10 ) );
    booleans( big, nullptr, "", threeOps, runs( 5 ) );
    booleans( noisy, nullptr, "", unionOnly, runs( 5 ) );
    booleans( medium, &rotMedium, "-rot", unionOnly, runs( 10 ) );
    booleans( big, &rotBig, "-rot", unionOnly, runs( 5 ) );
    if ( selected( "boolean/spheres" ) )
    {
        // each pose once per process; the driver takes the best per pose over rounds
        std::vector<double> times( sphereN );
        std::uint64_t h = 0;
        for ( int i = 0; i < sphereN; ++i )
        {
            const auto xf = spherePose( i, sphereN );
            BooleanParameters params;
            params.rigidB2A = &xf;
            const auto t0 = nowMs();
            const auto res = boolean( sphere, sphere, BooleanOperation::DifferenceAB, params );
            times[i] = nowMs() - t0;
            h = mix( h, hashBoolean( res ) );
        }
        std::printf( "P boolean/spheres-x%d/DifferenceAB", sphereN );
        for ( auto t : times )
            std::printf( " %.4f", t );
        std::printf( "\nH boolean/spheres-x%d/DifferenceAB %016llx\n", sphereN, (unsigned long long)h );
        std::fflush( stdout );
    }

    // the poses of MeshInspector's BooleanBenchmarkTool: two coincident spheres, B rotated around Z by (i+1) * 0.1 degrees
    if ( selected( "mic-spheres" ) )
    {
        const int n = gQuick ? 50 : 2000;
        auto micPose = []( int i )
        {
            const float angleRad = float( double( i + 1 ) * 0.1 * 3.14159265358979323846 / 180.0 );
            return AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), angleRad ) );
        };
        auto collideAll = [&]
        {
            std::uint64_t h = 0;
            for ( int i = 0; i < n; ++i )
            {
                const auto xf = micPose( i );
                const auto c = getVectorConverters( sphere, sphere, &xf ).toInt;
                h = mix( h, hashResult( findCollidingEdgeTrisPrecise( sphere, sphere, c, &xf ) ) );
            }
            return h;
        };
        const auto name = "mic-spheres-x" + std::to_string( n );
        measure( "collide/" + name + "/parallel", runs( 3 ), collideAll );
        {
            tbb::global_control gc( tbb::global_control::max_allowed_parallelism, 1 );
            measure( "collide/" + name + "/single", 1, collideAll );
        }
        // the same poses, but B is transformed in advance and passed without rigidB2A (only the calls are timed)
        auto collideNoXf = [&]( const std::string & caseName, int nRuns )
        {
            if ( !selected( caseName ) )
                return;
            double best = 1e300;
            std::uint64_t hash = 0;
            for ( int r = 0; r < nRuns; ++r )
            {
                double total = 0;
                std::uint64_t h = 0;
                for ( int i = 0; i < n; ++i )
                {
                    Mesh b = sphere;
                    b.transform( micPose( i ) );
                    b.getAABBTree();
                    const auto c = getVectorConverters( sphere, b ).toInt;
                    const auto t0 = nowMs();
                    auto found = findCollidingEdgeTrisPrecise( sphere, b, c );
                    total += nowMs() - t0;
                    h = mix( h, hashResult( std::move( found ) ) );
                }
                best = std::min( best, total );
                hash = h;
            }
            std::printf( "T %s %.4f\nH %s %016llx\n", caseName.c_str(), best, caseName.c_str(), (unsigned long long)hash );
            std::fflush( stdout );
        };
        collideNoXf( "collide/" + name + "-noxf/parallel", runs( 3 ) );
        {
            tbb::global_control gc( tbb::global_control::max_allowed_parallelism, 1 );
            collideNoXf( "collide/" + name + "-noxf/single", 1 );
        }
        std::vector<double> times( n );
        std::uint64_t h = 0;
        for ( int i = 0; i < n; ++i )
        {
            const auto xf = micPose( i );
            BooleanParameters params;
            params.rigidB2A = &xf;
            const auto t0 = nowMs();
            const auto res = boolean( sphere, sphere, BooleanOperation::DifferenceAB, params );
            times[i] = nowMs() - t0;
            h = mix( h, hashBoolean( res ) );
        }
        std::printf( "P boolean/%s/DifferenceAB", name.c_str() );
        for ( auto t : times )
            std::printf( " %.4f", t );
        std::printf( "\nH boolean/%s/DifferenceAB %016llx\n", name.c_str(), (unsigned long long)h );
        std::fflush( stdout );
    }

    // findSelfCollidingEdgeTrisPrecise and selfBoolean
    const auto rotSphere = rotationAround( Vector3f( 0.3f, 0.5f, 1 ), 0.01f, Vector3f() );
    selfCollide( "sphere", sphere, nullptr, runs( 20 ), runs( 10 ) );
    selfCollide( "sphere-rot", sphere, &rotSphere, runs( 20 ), runs( 10 ) );
    const auto mergedMedium = merged( medium );
    const auto rotMergedMedium = rotationAround( Vector3f( 0.3f, 0.5f, 1 ), 0.01f, mergedMedium.computeBoundingBox().center() );
    selfCollide( "merged-medium", mergedMedium, nullptr, runs( 5 ), runs( 2 ) );
    selfCollide( "merged-medium-rot", mergedMedium, &rotMergedMedium, runs( 5 ), runs( 2 ) );
    {
        const auto mergedBig = merged( big );
        const auto rotMergedBig = rotationAround( Vector3f( 0.3f, 0.5f, 1 ), 0.01f, mergedBig.computeBoundingBox().center() );
        selfCollide( "merged-big", mergedBig, nullptr, runs( 3 ), 1 );
        selfCollide( "merged-big-rot", mergedBig, &rotMergedBig, runs( 3 ), 1 );
        measure( "selfBoolean/merged-medium", runs( 5 ), [&]
        {
            auto res = selfBoolean( mergedMedium );
            return res ? hashMesh( *res ) : std::hash<std::string>{}( res.error() );
        } );
        measure( "selfBoolean/merged-big", runs( 3 ), [&]
        {
            auto res = selfBoolean( mergedBig );
            return res ? hashMesh( *res ) : std::hash<std::string>{}( res.error() );
        } );
    }
    std::printf( "E done\n" );
    return 0;
}
