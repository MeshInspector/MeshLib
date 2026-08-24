#include <MRMesh/MRAlphaShape.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshComponents.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <random>

namespace MR
{

TEST( MRMesh, AlphaShapeDuplicates )
{
    // four corners of a cube forming a tetrahedron, with the last one duplicated
    PointCloud cloud;
    cloud.points.push_back( {  0.5f,  0.5f, -0.5f } ); //0_v
    cloud.points.push_back( { -0.5f,  0.5f,  0.5f } ); //1_v
    cloud.points.push_back( {  0.5f,  0.5f,  0.5f } ); //2_v
    cloud.points.push_back( {  0.5f, -0.5f,  0.5f } ); //3_v
    cloud.points.push_back( {  0.5f, -0.5f,  0.5f } ); //4_v
    cloud.validPoints.resize( cloud.points.size(), true );

    const auto tris = findAlphaShapeAllTriangles( cloud, 1 );
    EXPECT_EQ( tris.size(), 4 );

    // the duplicated position is merged: only its smallest id appears in the triangles,
    // and the tetrahedron surface is closed - every directed edge is balanced by its opposite
    std::map<std::pair<VertId, VertId>, int> edges;
    for ( const auto & t : tris )
        for ( int i = 0; i < 3; ++i )
        {
            EXPECT_NE( t[i], 4_v );
            ++edges[ { t[i], t[( i + 1 ) % 3] } ];
        }
    for ( const auto & [e, n] : edges )
    {
        auto it = edges.find( { e.second, e.first } );
        EXPECT_EQ( n, it == edges.end() ? 0 : it->second );
    }

    // the duplicate point with the larger id gets no triangles of its own
    Triangulation vTris;
    std::vector<AlphaShapeNei> neis;
    auto data = getAlphaShapeData( cloud, 1, false );
    findAlphaShapeNeiTriangles( cloud, 4_v, data, vTris, neis, false );
    EXPECT_TRUE( vTris.empty() );
    EXPECT_TRUE( neis.empty() );
    findAlphaShapeNeiTriangles( cloud, 3_v, data, vTris, neis, false );
    EXPECT_EQ( vTris.size(), 3 );
    EXPECT_EQ( neis.size(), 3 );
}

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

    // each of the three points has the two others as neighbours, and one pair of them to check
    EXPECT_EQ( stats.collectedNeis, 6 );
    EXPECT_EQ( stats.redundancyTests, 3 );
    EXPECT_EQ( stats.redundantNeis, 0 ); // no point here is behind another one
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

void benchAlphaShape( const char * name, const PointCloud & cloud, float radius )
{
    AlphaShapeStats stats;
    const auto start = std::chrono::steady_clock::now();
    const auto tris = findAlphaShapeAllTriangles( cloud, radius, &stats );
    const auto ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - start ).count();
    std::cout << name << ": " << cloud.points.size() << " points, radius " << radius << '\n'
        << "  time            " << ms << " ms\n"
        << "  triangles       " << tris.size() << " (hash " << hashOf( tris ) << ")\n"
        << "  collectedNeis   " << stats.collectedNeis << '\n'
        << "  redundancyTests " << stats.redundancyTests << '\n'
        << "  redundantNeis   " << stats.redundantNeis << '\n'
        << "  consideredTris  " << stats.consideredTris << '\n'
        << "  touchableTris   " << stats.touchableTris << '\n'
        << "  inBallTests     " << stats.inBallTests << '\n'
        << "  shadowTests     " << stats.shadowTests << '\n'
        << "  exactShadowTests " << stats.exactShadowTests << '\n'
        << "  shadowedNeis    " << stats.shadowedNeis << std::endl;
}

} // anonymous namespace

// opt-in benchmark of the alpha-shape search on three clouds of different structure, in the idiom
// of DISABLED_FastIntMulWordsBench: run it with --gtest_also_run_disabled_tests to compare the
// timings and the counters of two branches, and the triangle hashes to prove they agree
TEST( MRMesh, DISABLED_AlphaShapeBench )
{
    benchAlphaShape( "sphere", sphereCloud( 40000, 1.f ), 0.02f );
    benchAlphaShape( "grid",   gridCloud( 200, 0.01f ),   0.03f );
    benchAlphaShape( "random", randomCloud( 40000, 1.f ), 0.05f );
}

} //namespace MR
