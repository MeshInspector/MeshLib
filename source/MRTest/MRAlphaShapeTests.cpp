#include <MRMesh/MRAlphaShape.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshComponents.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace MR
{

TEST( MRMesh, AlphaShape )
{
    PointCloud cloud;
    cloud.points.push_back( { 0.5f, 0.5f, 0.1f } ); //0_v
    cloud.points.push_back( { 0.5f, 0.5f, -.1f } ); //1_v
    cloud.points.push_back( { 0,    0,    0 } );    //2_v
    cloud.points.push_back( { 1,    0,    0 } );    //3_v
    cloud.points.push_back( { 0,    1,    0 } );    //4_v
    cloud.validPoints.autoResizeSet( 2_v, 3, true );

    Triangulation tris;
    std::vector<AlphaShapeNei> neis;
    AlphaShapeStats stats;

    auto data = getAlphaShapeData( cloud, 3, false );
    findAlphaShapeNeiTriangles( cloud, 3_v, data, tris, neis, true, &stats );
    EXPECT_EQ( tris.size(), 0 );
    findAlphaShapeNeiTriangles( cloud, 4_v, data, tris, neis, true, &stats );
    EXPECT_EQ( tris.size(), 0 );
    findAlphaShapeNeiTriangles( cloud, 2_v, data, tris, neis, true, &stats );
    EXPECT_EQ( tris.size(), 2 ); // two balls touching all three points from the opposite sides are empty

    // only the triangle 2-3-4 is considered, from point #2 with the two others having larger ids
    EXPECT_EQ( stats.consideredTris, 1 );
    EXPECT_EQ( stats.touchableTris, 1 );
    EXPECT_EQ( stats.inBallTests, 0 ); // no other points in the neighbourhood to test
    EXPECT_EQ( stats.shadowTests, 0 ); // and none behind the only triangle to be shadowed by it
    EXPECT_EQ( stats.exactShadowTests, 0 );
    EXPECT_EQ( stats.shadowedNeis, 0 );

    cloud.validPoints.set( 1_v );
    cloud.invalidateCaches();
    tris.clear();
    data = getAlphaShapeData( cloud, 3, false );
    findAlphaShapeNeiTriangles( cloud, 2_v, data, tris, neis, true );
    EXPECT_EQ( tris.size(), 1 ); // 1_v is inside one of the two balls

    cloud.validPoints.set( 0_v );
    cloud.invalidateCaches();
    tris.clear();
    data = getAlphaShapeData( cloud, 3, false );
    findAlphaShapeNeiTriangles( cloud, 2_v, data, tris, neis, true );
    EXPECT_EQ( tris.size(), 0 ); // 0_v and 1_v are inside the balls on the both sides

    const auto allTris = findAlphaShapeAllTriangles( cloud, 3 );
    EXPECT_EQ( allTris.size(), 6 );
}

// four points of a square are exactly on both balls passing via any three of them,
// so every ball emptiness test here is a tie resolved by simulation-of-simplicity
TEST( MRMesh, AlphaShapeSquare )
{
    PointCloud cloud;
    cloud.points.push_back( { 0, 0, 0 } ); //0_v
    cloud.points.push_back( { 1, 0, 0 } ); //1_v
    cloud.points.push_back( { 1, 1, 0 } ); //2_v
    cloud.points.push_back( { 0, 1, 0 } ); //3_v
    cloud.validPoints.autoResizeSet( 0_v, 4, true );

    const auto tris = findAlphaShapeAllTriangles( cloud, 0.8f );
    // the square is covered by two triangles from each side, and the sides take different diagonals;
    // in floating point all the four balls looked empty giving eight triangles
    const std::vector<ThreeVertIds> expectedTris{
        { 0_v, 1_v, 2_v }, { 0_v, 2_v, 3_v }, // the diagonal 0-2 from the positive side
        { 0_v, 3_v, 1_v }, { 1_v, 3_v, 2_v }  // the diagonal 1-3 from the negative side
    };
    EXPECT_EQ( tris.vec_, expectedTris );

    const auto mesh = findAlphaShape( cloud, 0.8f );
    EXPECT_EQ( mesh.topology.numValidFaces(), 4 );
    EXPECT_TRUE( mesh.topology.isClosed() );
}

// two grids crossing along a line: many junction fans where several continuation triangles exist;
// the ccwAroundLine-based selection of the best continuation gives one connected component here,
// while taking the first found continuation gave 468 vertices in 65 components
TEST( MRMesh, AlphaShapeCrossingGrids )
{
    PointCloud cloud;
    for ( int i = 0; i <= 10; ++i )
        for ( int j = 0; j <= 10; ++j )
            cloud.points.push_back( { i * 0.05f, j * 0.05f, 0 } );
    for ( int i = 0; i <= 10; ++i )
        for ( int k = 0; k <= 10; ++k )
            if ( k != 5 )
                cloud.points.push_back( { i * 0.05f, 0.25f, k * 0.05f - 0.25f } );
    cloud.validPoints.autoResizeSet( 0_v, (int)cloud.points.size(), true );

    AlphaShapeStats stats;
    const auto mesh = findAlphaShape( cloud, 0.1f, &stats );
    // the counters are about 99000, 47600 and 1324000 here, but not exactly the same on every
    // platform, because the neighbourhood of a point is searched in floating point
    EXPECT_GT( stats.consideredTris, stats.touchableTris );
    EXPECT_GT( stats.inBallTests, stats.touchableTris );
    EXPECT_EQ( mesh.topology.numValidFaces(), 584 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 322 );
    EXPECT_EQ( MeshComponents::getNumComponents( mesh ), 1 );
}

namespace
{

// the found triangles must be exactly the same on every branch and every platform,
// so a single number is enough to compare the runs below
std::uint64_t hashOf( const Triangulation & tris )
{
    std::uint64_t h = 1469598103934665603ull; // FNV-1a
    for ( const auto & t : tris )
        for ( const auto v : t )
            for ( int i = 0; i < 4; ++i )
                h = ( h ^ ( ( unsigned( int( v ) ) >> ( 8 * i ) ) & 0xff ) ) * 1099511628211ull;
    return h;
}

// a sphere of the given radius sampled by the Fibonacci spiral: every point is on the alpha-shape
PointCloud sphereCloud( int n, float radius )
{
    PointCloud res;
    res.points.reserve( n );
    constexpr float golden = 2.39996323f; // pi * ( 3 - sqrt( 5 ) )
    for ( int i = 0; i < n; ++i )
    {
        const float z = 1 - ( 2 * i + 1.f ) / n;
        const float r = std::sqrt( std::max( 0.f, 1 - z * z ) );
        const float a = golden * i;
        res.points.push_back( radius * Vector3f{ r * std::cos( a ), r * std::sin( a ), z } );
    }
    res.validPoints.autoResizeSet( 0_v, n, true );
    return res;
}

// a plane grid of the given step, the densest neighbourhood the filters below have to prune
PointCloud gridCloud( int n, float step )
{
    PointCloud res;
    res.points.reserve( n * n );
    for ( int i = 0; i < n; ++i )
        for ( int j = 0; j < n; ++j )
            res.points.push_back( { i * step, j * step, 0 } );
    res.validPoints.autoResizeSet( 0_v, n * n, true );
    return res;
}

// uniform noise in a cube: no structure, and the neighbourhoods are the largest of the three
PointCloud randomCloud( int n, float size )
{
    PointCloud res;
    res.points.reserve( n );
    std::mt19937 gen( 20260813 );
    std::uniform_real_distribution<float> d( 0, size );
    for ( int i = 0; i < n; ++i )
        res.points.push_back( { d( gen ), d( gen ), d( gen ) } );
    res.validPoints.autoResizeSet( 0_v, n, true );
    return res;
}

// the number of points of the largest cloud: a Release build on a hosted runner searches it in
// seconds, while MSVC iterator-debug, brew-llvm Debug and wasm are one to two orders slower and
// share the same 10-minute cap on the whole MRTest run, so they get a proportionally smaller cloud
#if defined( NDEBUG ) && !defined( __EMSCRIPTEN__ )
constexpr int cBenchPoints = 40000;
#else
constexpr int cBenchPoints = 4000;
#endif

// the reps of one variant, capped by the count and by the wall clock: five is enough for the
// minimum to be a stable estimator, and the budget keeps a slow runner from eating the job's cap
constexpr int cBenchReps = 5;
constexpr double cBenchBudgetMs = 20000;

struct BenchResult
{
    std::vector<double> ms; // sorted times of the completed reps
    std::uint64_t hash = 0;
    size_t numTris = 0;
    AlphaShapeStats stats; // of the last rep, so the counters are per-search and not accumulated
};

// searches the same cloud with the same prepared data cBenchReps times and keeps the times;
// the shadow filter is switched by data.shadowFilter, so the two variants a branch is compared
// by run in one process on one runner, on the same cloud and with the same tree
BenchResult benchVariant( const PointCloud & cloud, const AlphaShapeData & data )
{
    BenchResult res;
    double total = 0;
    for ( int i = 0; i < cBenchReps && total < cBenchBudgetMs; ++i )
    {
        AlphaShapeStats stats;
        const auto start = std::chrono::steady_clock::now();
        const auto tris = findAlphaShapeAllTriangles( cloud, data, {}, &stats );
        const auto ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - start ).count();
        EXPECT_TRUE( tris.has_value() ); // no progress callback is given, so the search cannot be cancelled
        if ( !tris )
            break;
        res.ms.push_back( ms );
        total += ms;
        res.hash = hashOf( *tris );
        res.numTris = tris->size();
        res.stats = stats;
    }
    std::sort( res.ms.begin(), res.ms.end() );
    return res;
}

// one greppable line per variant: 19 MRTest jobs are scraped for these
void printVariant( const char * name, const PointCloud & cloud, float radius, bool filter, const BenchResult & r )
{
    if ( r.ms.empty() )
        return;
    std::cout << "[alpha-bench] " << name << " points=" << cloud.points.size() << " radius=" << radius
        << " filter=" << ( filter ? "on" : "off" )
        << " reps=" << r.ms.size()
        << " min=" << r.ms.front() << " med=" << r.ms[r.ms.size() / 2] << " max=" << r.ms.back()
        << " tris=" << r.numTris << " hash=" << r.hash
        << " consideredTris=" << r.stats.consideredTris
        << " touchableTris=" << r.stats.touchableTris
        << " inBallTests=" << r.stats.inBallTests
        << " shadowTests=" << r.stats.shadowTests
        << " exactShadowTests=" << r.stats.exactShadowTests
        << " shadowedNeis=" << r.stats.shadowedNeis << std::endl;
}

void benchAlphaShape( const char * name, const PointCloud & cloud, float radius )
{
    auto data = getAlphaShapeData( cloud, radius, true );
    cloud.getAABBTree(); // built once here, so it is not timed with the first rep

    // filter off first: this branch is master plus the shadow filter, so that variant is the
    // master baseline, and its counters are what a master build must reproduce
    data.shadowFilter = false;
    const auto off = benchVariant( cloud, data );
    data.shadowFilter = true;
    const auto on = benchVariant( cloud, data );

    printVariant( name, cloud, radius, false, off );
    printVariant( name, cloud, radius, true, on );
    if ( !off.ms.empty() && !on.ms.empty() )
        std::cout << "[alpha-bench] " << name << " speedup=" << off.ms.front() / on.ms.front()
            << " inBallTestsRatio=" << double( off.stats.inBallTests ) / double( std::max<size_t>( 1, on.stats.inBallTests ) )
            << std::endl;

    // the filter only removes the neighbours that cannot change the outcome, so the whole point
    // of it is that these two agree - on every cloud, every runner and every branch
    EXPECT_EQ( off.hash, on.hash );
    EXPECT_EQ( off.numTris, on.numTris );
    EXPECT_LE( on.stats.inBallTests, off.stats.inBallTests );
    EXPECT_EQ( off.stats.shadowTests, 0 );
}

} // anonymous namespace

// benchmark of the alpha-shape search on three clouds of different structure, timing the shadow
// filter against the unfiltered search (= master) in one process: same runner, same cloud, same
// binary. The cloud sizes keep the neighbourhood of a point about the same in every config, so
// the filter-on/off ratio is comparable across the runners even where the absolute times are not
TEST( MRMesh, DISABLED_AlphaShapeBench )
{
    const int n = cBenchPoints;
    // the ball radius is scaled with the sampling step of each cloud, so a smaller cloud is not
    // also a sparser one: step ~ 1/sqrt(n) on the sphere and the grid, ~ 1/cbrt(n) in the cube
    benchAlphaShape( "sphere", sphereCloud( n, 1.f ), 0.02f * std::sqrt( 40000.f / n ) );
    benchAlphaShape( "grid",   gridCloud( int( std::lround( std::sqrt( float( n ) ) ) ), 0.01f ), 0.03f );
    benchAlphaShape( "random", randomCloud( n, 1.f ), 0.05f * std::cbrt( 40000.f / n ) );
}

} //namespace MR
