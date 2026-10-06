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
#include <set>

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

TEST( MRMesh, BallPivotVertex )
{
    // the pivot edge is on the z-axis directed up, the other points are around it at mid-height,
    // and the balls via the edge and #vk are empty from both sides
    PointCloud cloud;
    cloud.points.push_back( {  0,       0,       0    } ); //0_v, vi
    cloud.points.push_back( {  0,       0,       1    } ); //1_v, vj
    cloud.points.push_back( {  0.8485f, -0.8485f, 0.5f } ); //2_v, vk, rotation 315 degrees
    cloud.points.push_back( {  0,       1,       0.5f } ); //3_v, rotation 90 degrees
    cloud.points.push_back( { -1,       0,       0.5f } ); //4_v, rotation 180 degrees
    cloud.points.push_back( {  0.2952f, 1.6742f, 0.5f } ); //5_v, rotation 80 degrees, far from the axis
    cloud.validPoints.resize( cloud.points.size(), true );

    auto data = getAlphaShapeData( cloud, 1, false );
    std::vector<BallPivotCandidate> cands;
    auto ids = [&cands]
    {
        std::vector<VertId> res;
        for ( const auto & c : cands )
            res.push_back( c.coords.id );
        std::sort( res.begin(), res.end() ); // the order of the candidates is unspecified
        return res;
    };
    const std::vector<VertId> allCands{ 3_v, 4_v, 5_v };

    // 5_v is the first counter-clockwise, but its ball contains 3_v, which is hit by the rolling ball first
    EXPECT_EQ( findBallPivotVertex( cloud, 0_v, 1_v, 2_v, data, cands ), 3_v );
    EXPECT_EQ( ids(), allCands );

    // the reversed edge rotates the other way
    EXPECT_EQ( findBallPivotVertex( cloud, 1_v, 0_v, 2_v, data, cands ), 4_v );
    EXPECT_EQ( ids(), allCands );

    cloud.validPoints.reset( 3_v );
    cloud.invalidateCaches();
    data = getAlphaShapeData( cloud, 1, false );
    EXPECT_EQ( findBallPivotVertex( cloud, 0_v, 1_v, 2_v, data, cands ), 5_v );

    // with no other point, the ball rotates to the other side of the same triangle
    PointCloud tri;
    for ( VertId v : { 0_v, 1_v, 2_v } )
        tri.points.push_back( cloud.points[v] );
    tri.validPoints.resize( tri.points.size(), true );
    data = getAlphaShapeData( tri, 1, false );
    EXPECT_EQ( findBallPivotVertex( tri, 0_v, 1_v, 2_v, data, cands ), 2_v );
    EXPECT_TRUE( cands.empty() );
    EXPECT_EQ( findBallPivotVertex( tri, 1_v, 0_v, 2_v, data, cands ), 2_v );
}

// the ball pivoted over any edge of an alpha-shape triangle must stop at another alpha-shape triangle;
// the thin shell has most triangles in both orientations, where the ball pivoted from one side of a triangle
// could touch the neighbour behind the starting ball, if #vk were not tested for being inside
TEST( MRMesh, BallPivotAlphaShapeTriangles )
{
    PointCloud cloud;
    const int n = 2000;
    for ( int i = 0; i < n; ++i ) // Fibonacci sphere
    {
        const float z = 1 - ( 2 * i + 1 ) / float( n );
        const float rho = std::sqrt( 1 - z * z );
        const float phi = 2.39996323f * i;
        cloud.points.push_back( { rho * std::cos( phi ), rho * std::sin( phi ), z } );
    }
    cloud.validPoints.resize( cloud.points.size(), true );

    auto cyclic = []( VertId a, VertId b, VertId c ) // the rotation starting from the smallest id
    {
        if ( b < a && b < c )
            return std::array<VertId, 3>{ b, c, a };
        if ( c < a && c < b )
            return std::array<VertId, 3>{ c, a, b };
        return std::array<VertId, 3>{ a, b, c };
    };
    const auto data = getAlphaShapeData( cloud, 0.2f, true );
    const auto tris = findAlphaShapeAllTriangles( cloud, data );
    std::set<std::array<VertId, 3>> triSet;
    for ( const auto & t : tris )
        triSet.insert( cyclic( t[0], t[1], t[2] ) );

    std::vector<BallPivotCandidate> cands;
    for ( const auto & t : tris )
        for ( int e = 0; e < 3; ++e )
        {
            const VertId a = t[e], b = t[( e + 1 ) % 3], c = t[( e + 2 ) % 3];
            const auto x = findBallPivotVertex( cloud, a, b, c, data, cands );
            EXPECT_TRUE( triSet.contains( cyclic( b, a, x ) ) );
        }
    EXPECT_GT( tris.size(), size_t( 1000 ) );
}

// the ball pivoting finds only the outer side of a sphere sampled on its surface and inside,
// while the alpha shape has also the inner side and the triangles around the inner points;
// the leftmost point is a far outlier with no alpha-shape triangles, skipped in the search of the first triangle
TEST( MRMesh, BallPivotingSphere )
{
    PointCloud cloud;
    const int n = 2000;
    for ( int i = 0; i < n; ++i ) // Fibonacci sphere
    {
        const float z = 1 - ( 2 * i + 1 ) / float( n );
        const float rho = std::sqrt( 1 - z * z );
        const float phi = 2.39996323f * i;
        cloud.points.push_back( { rho * std::cos( phi ), rho * std::sin( phi ), z } );
    }
    std::mt19937 gen( 0 );
    std::uniform_real_distribution<float> coord( -0.28f, 0.28f );
    for ( int i = 0; i < 200; ++i )
        cloud.points.push_back( { coord( gen ), coord( gen ), coord( gen ) } );
    cloud.points.push_back( { -5, 0, 0 } );
    cloud.validPoints.resize( cloud.points.size(), true );

    std::vector<MeshBuilder::VertDuplication> dups;
    const auto mesh = findBallPivotingMesh( cloud, 0.2f, &dups );
    EXPECT_TRUE( dups.empty() );
    EXPECT_TRUE( mesh.topology.isClosed() );
    EXPECT_EQ( mesh.topology.numValidVerts(), n );
    EXPECT_EQ( mesh.topology.lastValidVert(), VertId( n - 1 ) );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 * n - 4 );
    EXPECT_GT( mesh.volume(), 4.0 ); // outward orientation: the volume of the unit ball is 4.19

    EXPECT_GT( findAlphaShapeAllTriangles( cloud, 0.2f ).size(), size_t( 2 * ( 2 * n - 4 ) ) );
}

// a tenth of the points of a sphere have twins with smaller ids and another tenth with larger ids:
// only the smallest id of each position must appear, and the triangles must be the same as without the twins
TEST( MRMesh, BallPivotingTwins )
{
    const int n = 2000;
    std::vector<Vector3f> sphere;
    for ( int i = 0; i < n; ++i ) // Fibonacci sphere
    {
        const float z = 1 - ( 2 * i + 1 ) / float( n );
        const float rho = std::sqrt( 1 - z * z );
        const float phi = 2.39996323f * i;
        sphere.push_back( { rho * std::cos( phi ), rho * std::sin( phi ), z } );
    }
    PointCloud cloud;
    std::vector<int> posOf; // the index in sphere of each cloud point
    std::vector<VertId> smallestId( n );
    for ( int i = 5; i < n; i += 10 )
    {
        smallestId[i] = VertId( posOf.size() );
        posOf.push_back( i );
    }
    for ( int i = 0; i < n; ++i )
    {
        if ( !smallestId[i] )
            smallestId[i] = VertId( posOf.size() );
        posOf.push_back( i );
    }
    for ( int i = 0; i < n; i += 10 )
        posOf.push_back( i );
    for ( int i : posOf )
        cloud.points.push_back( sphere[i] );
    cloud.validPoints.resize( cloud.points.size(), true );

    PointCloud plain;
    plain.points.vec_ = sphere;
    plain.validPoints.resize( n, true );

    // the triangles in sphere indices, rotated to start from the smallest one
    auto posTris = []( const Triangulation & tris, auto && pos )
    {
        std::set<std::array<int, 3>> res;
        for ( const auto & t : tris )
        {
            std::array<int, 3> p{ pos( t[0] ), pos( t[1] ), pos( t[2] ) };
            std::rotate( p.begin(), std::min_element( p.begin(), p.end() ), p.end() );
            res.insert( p );
        }
        return res;
    };
    const auto data = getAlphaShapeData( cloud, 0.2f, true );
    EXPECT_EQ( data.twins.count(), 2 * ( n / 10 ) );
    for ( VertId v : data.twins )
    {
        EXPECT_NE( v, smallestId[posOf[v]] );
    }
    const auto tris = *findBallPivotingTriangles( cloud, data );
    const auto plainTris = *findBallPivotingTriangles( plain, getAlphaShapeData( plain, 0.2f, true ) );
    EXPECT_EQ( tris.size(), 2 * n - 4 );
    for ( const auto & t : tris )
        for ( VertId v : t )
        {
            EXPECT_EQ( v, smallestId[posOf[v]] );
        }
    EXPECT_EQ( posTris( tris, [&]( VertId v ) { return posOf[v]; } ), posTris( plainTris, []( VertId v ) { return int( v ); } ) );

    const auto mesh = findBallPivotingMesh( cloud, 0.2f );
    EXPECT_TRUE( mesh.topology.isClosed() );
    EXPECT_EQ( mesh.topology.numValidVerts(), n );
    EXPECT_EQ( MeshComponents::getNumComponents( mesh ), 1 );
}

// three tetrahedra sharing edge (0_v, 1_v), 120 degrees apart around it: the empty balls rotating around the edge
// form three arcs, so each direction of the edge belongs to three triangles; the pivoting over the edge must not stop
// after the first arc is crossed, since the third tetrahedron is reachable only via the edge
TEST( MRMesh, BallPivotingTetrahedraSharingEdge )
{
    PointCloud cloud;
    cloud.points.push_back( { 0, 0, 0 } );
    cloud.points.push_back( { 0, 0, 1 } );
    for ( int i = 0; i < 3; ++i )
        for ( float da : { -0.26f, 0.26f } ) // +-15 degrees
        {
            const float a = i * 2.0944f + da;
            cloud.points.push_back( { 1.5f * std::cos( a ), 1.5f * std::sin( a ), 0.5f } );
        }
    cloud.validPoints.resize( cloud.points.size(), true );

    const auto tris = *findBallPivotingTriangles( cloud, getAlphaShapeData( cloud, 1, true ) );
    EXPECT_EQ( tris.size(), 12 );
    std::map<std::pair<VertId, VertId>, int> edges;
    for ( const auto & t : tris )
        for ( int i = 0; i < 3; ++i )
            ++edges[ { t[i], t[( i + 1 ) % 3] } ];
    for ( const auto & [e, num] : edges ) // every directed edge is balanced by its opposite
    {
        const auto it = edges.find( { e.second, e.first } );
        EXPECT_EQ( num, it == edges.end() ? 0 : it->second );
    }
    EXPECT_EQ( ( edges[ { 0_v, 1_v } ] ), 3 );

    const auto mesh = findBallPivotingMesh( cloud, 1 );
    EXPECT_EQ( MeshComponents::getNumComponents( mesh ), 3 );
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
    std::vector<MeshBuilder::VertDuplication> dups;
    const auto mesh = findAlphaShape( cloud, 0.1f, &dups, &stats );
    // the counters are about 99000, 47600 and 1324000 here, but not exactly the same on every
    // platform, because the neighbourhood of a point is searched in floating point
    EXPECT_GT( stats.consideredTris, stats.touchableTris );
    EXPECT_GT( stats.inBallTests, stats.touchableTris );
    EXPECT_EQ( mesh.topology.numValidFaces(), 584 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 322 );
    EXPECT_EQ( MeshComponents::getNumComponents( mesh ), 1 );
    EXPECT_EQ( mesh.points.size(), cloud.points.size() + dups.size() );
    EXPECT_EQ( dups.size(), 136 );
    for ( const auto & d : dups )
    {
        EXPECT_LT( (int)d.srcVert, (int)cloud.points.size() );
        EXPECT_GE( (int)d.dupVert, (int)cloud.points.size() );
    }
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
