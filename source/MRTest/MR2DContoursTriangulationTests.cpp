#include <MRMesh/MR2DContoursTriangulation.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRContour.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRExtractIsolines.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MR2to3.h>
#include <MRSymbolMesh/MRSymbolMesh.h>
#include <gtest/gtest.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include <functional>
#include <cstdio>
#include <cmath>
#include <limits>

namespace MR
{

TEST( MRMesh, PlanarTriangulation )
{
    // Create a quadrangle with three points on a straight line
    Contour2f cont;
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );
    cont.push_back( Vector2f( 0.f, 2.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );

    Mesh mesh = PlanarTriangulation::triangulateContours( { cont } );
    mesh.pack();
    EXPECT_TRUE( mesh.topology.lastValidFace() == 1_f );
    // Must not contain degenerate faces
    EXPECT_TRUE( mesh.triangleAspectRatio( 0_f ) < 10.0f );
    EXPECT_TRUE( mesh.triangleAspectRatio( 1_f ) < 10.0f );
}

TEST( MRMesh, PlanarTriangulationWindingAndIntersections )
{
    // signed crossing number, independent of the sweep line internals
    auto windingOracle = [] ( const Contours2f& conts, const Vector2f& p )
    {
        int w = 0;
        for ( const auto& cont : conts )
        {
            for ( size_t i = 0; i + 1 < cont.size(); ++i )
            {
                const auto& a = cont[i];
                const auto& b = cont[i + 1];
                if ( a.y <= p.y && b.y > p.y && cross( b - a, p - a ) > 0 )
                    ++w;
                else if ( b.y <= p.y && a.y > p.y && cross( b - a, p - a ) < 0 )
                    --w;
            }
        }
        return w;
    };

    // checks each face's winding against the oracle at the face centroid (a mismatch means the face
    // straddles two winding regions, e.g. if a Delone flip crossed a contour edge), then total areas per winding
    auto checkWinding = [&] ( const Contours2f& conts, const Mesh& mesh, const Vector<int, FaceId>& faceWinding,
        double expectedArea1, double expectedArea2 )
    {
        ASSERT_EQ( faceWinding.size(), mesh.topology.faceSize() );
        double areaByWinding[3] = {};
        for ( auto f : mesh.topology.getValidFaces() )
        {
            const int w = faceWinding[f];
            EXPECT_EQ( w, windingOracle( conts, to2dim( mesh.triCenter( f ) ) ) );
            ASSERT_TRUE( w == 1 || w == 2 );
            areaByWinding[w] += mesh.area( f );
        }
        EXPECT_NEAR( areaByWinding[1], expectedArea1, 1e-4 );
        EXPECT_NEAR( areaByWinding[2], expectedArea2, 1e-4 );
    };

    {
        // two overlapping ccw squares [0,2]^2 and [1,3]^2: the [1,2]^2 overlap has winding number 2, the rest of the union 1
        const Contours2f conts =
        {
            { { 0.f, 0.f }, { 2.f, 0.f }, { 2.f, 2.f }, { 0.f, 2.f }, { 0.f, 0.f } },
            { { 1.f, 1.f }, { 3.f, 1.f }, { 3.f, 3.f }, { 1.f, 3.f }, { 1.f, 1.f } }
        };

        PlanarTriangulation::IntersectionsMap interMap;
        Vector<int, FaceId> faceWinding;
        const Mesh mesh = PlanarTriangulation::triangulateContours( conts,
            { .outFaceWinding = &faceWinding, .outInterMap = &interMap } );

        // squares' edges cross at (2,1) and (1,2); each crossing vertex interpolates both of its source edges
        EXPECT_EQ( interMap.shift, size_t( 8 ) );
        ASSERT_EQ( interMap.map.size(), size_t( 2 ) );
        for ( size_t i = 0; i < interMap.map.size(); ++i )
        {
            const auto& info = interMap.map[i];
            ASSERT_TRUE( info.isIntersection() );
            const auto p = to2dim( mesh.points[VertId( interMap.shift + i )] );
            const auto l = ( 1 - info.lRatio ) * to2dim( mesh.points[info.lOrg] ) + info.lRatio * to2dim( mesh.points[info.lDest] );
            const auto u = ( 1 - info.uRatio ) * to2dim( mesh.points[info.uOrg] ) + info.uRatio * to2dim( mesh.points[info.uDest] );
            EXPECT_LE( ( l - p ).length(), 1e-6f );
            EXPECT_LE( ( u - p ).length(), 1e-6f );
        }

        checkWinding( conts, mesh, faceWinding, 6.0, 1.0 ); // union 7 = 6 + the [1,2]^2 overlap

        // the holeVertsIds overload stays available and resolves without ambiguity
        EXPECT_EQ( PlanarTriangulation::triangulateContours( conts, nullptr ).topology.numValidFaces(), mesh.topology.numValidFaces() );
    }

    {
        // long thin overlap strip [0,10]x[0,0.3] (winding 2) with a far midpoint vertex below: if Delone flips
        // ran here, they would cross the strip's long boundary edges and smear face winding
        const Contours2f conts =
        {
            { { 0.f, -2.f }, { 5.f, -2.f }, { 10.f, -2.f }, { 10.f, 0.3f }, { 0.f, 0.3f }, { 0.f, -2.f } },
            { { -1.f, 0.f }, { 11.f, 0.f }, { 11.f, 2.f }, { -1.f, 2.f }, { -1.f, 0.f } }
        };

        Vector<int, FaceId> faceWinding;
        const Mesh mesh = PlanarTriangulation::triangulateContours( conts, { .outFaceWinding = &faceWinding } );

        checkWinding( conts, mesh, faceWinding, 41.0, 3.0 ); // areas 23 + 24 with the strip counted once per winding
    }
}

TEST( MRMesh, PlanarTriangulationMeshSpace )
{
    // a square boundary lying on a plane tilted off all axes, triangulated in its own 3d space
    const Vector3f normal = Vector3f( 1.f, 2.f, 3.f ).normalized();
    Vector3f u = cross( normal, Vector3f::plusX() );
    if ( u.lengthSq() < 1e-6f )
        u = cross( normal, Vector3f::plusY() );
    u = u.normalized();
    const Vector3f w = cross( normal, u );

    const Vector3f center( 10.f, -5.f, 2.f );
    const std::vector<Vector3f> corners = { center - u - w, center + u - w, center + u + w, center - u + w };

    Mesh mesh;
    const EdgeId e0 = mesh.addSeparateEdgeLoop( corners );
    const EdgeLoop loop = trackRightBoundaryLoop( mesh.topology, e0 );
    ASSERT_GE( loop.size(), size_t( 3 ) );

    const auto res = PlanarTriangulation::triangulateDisjointContours( mesh, EdgeLoops{ loop }, dominantSignedAxis( normal ) );
    ASSERT_TRUE( res.has_value() );
    const Mesh& patch = *res;

    EXPECT_EQ( patch.topology.numValidFaces(), 2 ); // convex quad -> 2 triangles

    // output vertices keep the exact mesh coordinates (no projection round-trip)
    for ( const EdgeId e : loop )
    {
        const Vector3f src = mesh.orgPnt( e );
        float best = std::numeric_limits<float>::max();
        for ( auto vId : patch.topology.getValidVerts() )
            best = std::min( best, ( patch.points[vId] - src ).length() );
        EXPECT_LE( best, 1e-4f );
    }

    // output faces are oriented consistently with the input loop's winding around +normal (validates the dominant-axis parity)
    Vector3f loopNormal;
    for ( const EdgeId e : loop )
        loopNormal += cross( mesh.orgPnt( e ), mesh.destPnt( e ) );
    const float inSign = dot( loopNormal, normal );
    for ( auto f : patch.topology.getValidFaces() )
        EXPECT_GT( inSign * dot( patch.normal( f ), normal ), 0.f );
}

namespace
{

// checks that the patch's boundary loops correspond 1:1 to the input mesh loops:
// same sizes, patch side filled (left face valid), exact original coordinates at every origin
void checkPatchBorders( const Mesh& mesh, const EdgeLoops& loops, const Mesh& patch, const std::vector<EdgePath>& bds )
{
    ASSERT_EQ( bds.size(), loops.size() );
    for ( size_t i = 0; i < loops.size(); ++i )
    {
        ASSERT_EQ( bds[i].size(), loops[i].size() );
        for ( size_t j = 0; j < loops[i].size(); ++j )
        {
            EXPECT_TRUE( patch.topology.left( bds[i][j] ).valid() );
            EXPECT_EQ( patch.points[patch.topology.org( bds[i][j] )], mesh.points[mesh.topology.org( loops[i][j] )] );
            EXPECT_EQ( patch.points[patch.topology.dest( bds[i][j] )], mesh.points[mesh.topology.dest( loops[i][j] )] );
        }
    }
}

}

TEST( MRMesh, PlanarTriangulationMeshSpacePinch )
{
    // two ccw square loops sharing exactly one mesh vertex v0 (a bowtie): the configuration a cut
    // contour leaves when it touches the removed face's boundary at a vertex
    Mesh mesh;
    auto& tp = mesh.topology;
    const VertId v0 = tp.addVertId(); // shared
    const VertId a1 = tp.addVertId(), a2 = tp.addVertId(), a3 = tp.addVertId(); // lower-left square
    const VertId b1 = tp.addVertId(), b2 = tp.addVertId(), b3 = tp.addVertId(); // upper-right square
    mesh.points.autoResizeSet( v0, Vector3f( 0, 0, 0 ) );
    mesh.points.autoResizeSet( a1, Vector3f( -1, 0, 0 ) );
    mesh.points.autoResizeSet( a2, Vector3f( -1, -1, 0 ) );
    mesh.points.autoResizeSet( a3, Vector3f( 0, -1, 0 ) );
    mesh.points.autoResizeSet( b1, Vector3f( 1, 0, 0 ) );
    mesh.points.autoResizeSet( b2, Vector3f( 1, 1, 0 ) );
    mesh.points.autoResizeSet( b3, Vector3f( 0, 1, 0 ) );

    const EdgeId eA0 = tp.makeEdge(), eA1 = tp.makeEdge(), eA2 = tp.makeEdge(), eA3 = tp.makeEdge();
    const EdgeId eB0 = tp.makeEdge(), eB1 = tp.makeEdge(), eB2 = tp.makeEdge(), eB3 = tp.makeEdge();
    // rings at the simple vertices: in-edge and out-edge only
    tp.splice( eA0.sym(), eA1 );
    tp.splice( eA1.sym(), eA2 );
    tp.splice( eA2.sym(), eA3 );
    tp.splice( eB0.sym(), eB1 );
    tp.splice( eB1.sym(), eB2 );
    tp.splice( eB2.sym(), eB3 );
    // ring at v0 in ccw order around +Z: toward b1 (0), b3 (90), a1 (180), a3 (270);
    // each loop's interior wedge stays contiguous, as in a real mesh
    tp.splice( eB0, eB3.sym() );
    tp.splice( eB3.sym(), eA0 );
    tp.splice( eA0, eA3.sym() );
    tp.setOrg( eA0, v0 );
    tp.setOrg( eA1, a1 );
    tp.setOrg( eA2, a2 );
    tp.setOrg( eA3, a3 );
    tp.setOrg( eB1, b1 );
    tp.setOrg( eB2, b2 );
    tp.setOrg( eB3, b3 );

    const EdgeLoops loops = { { eA0, eA1, eA2, eA3 }, { eB0, eB1, eB2, eB3 } };
    std::vector<EdgePath> bds;
    const auto res = PlanarTriangulation::triangulateDisjointContours( mesh, loops, SignedAxis::PlusZ, &bds );
    ASSERT_TRUE( res.has_value() );
    const Mesh& patch = *res;

    EXPECT_EQ( patch.topology.numValidFaces(), 4 ); // 2 triangles per square
    EXPECT_NEAR( patch.area(), 2.0f, 1e-5f );
    // one figure-eight boundary loop: the hole walk passes the shared vertex twice
    EXPECT_EQ( patch.topology.findHoleRepresentiveEdges().size(), 1 );
    for ( auto f : patch.topology.getValidFaces() )
        EXPECT_GT( dot( patch.normal( f ), Vector3f::plusZ() ), 0.f );
    checkPatchBorders( mesh, loops, patch, bds );
}

TEST( MRMesh, PlanarTriangulationMeshSpaceSharedEdge )
{
    // square [0,3]^2 split by a chord: two ccw loops traverse the chord edge in opposite directions —
    // exactly what a cut contour crossing a removed face leaves for the joint per-face fill
    Mesh mesh;
    auto& tp = mesh.topology;
    const VertId c0 = tp.addVertId(), m1 = tp.addVertId(), c1 = tp.addVertId();
    const VertId c2 = tp.addVertId(), m2 = tp.addVertId(), c3 = tp.addVertId();
    mesh.points.autoResizeSet( c0, Vector3f( 0, 0, 0 ) );
    mesh.points.autoResizeSet( m1, Vector3f( 1.5f, 0, 0 ) );
    mesh.points.autoResizeSet( c1, Vector3f( 3, 0, 0 ) );
    mesh.points.autoResizeSet( c2, Vector3f( 3, 3, 0 ) );
    mesh.points.autoResizeSet( m2, Vector3f( 1.5f, 3, 0 ) );
    mesh.points.autoResizeSet( c3, Vector3f( 0, 3, 0 ) );

    const EdgeId eBL = tp.makeEdge(); // c0 -> m1
    const EdgeId eBR = tp.makeEdge(); // m1 -> c1
    const EdgeId eR = tp.makeEdge();  // c1 -> c2
    const EdgeId eTR = tp.makeEdge(); // c2 -> m2
    const EdgeId eTL = tp.makeEdge(); // m2 -> c3
    const EdgeId eL = tp.makeEdge();  // c3 -> c0
    const EdgeId eC = tp.makeEdge();  // m1 -> m2, the chord
    tp.splice( eL.sym(), eBL );      // c0
    tp.splice( eBR.sym(), eR );      // c1
    tp.splice( eR.sym(), eTR );      // c2
    tp.splice( eTL.sym(), eL );      // c3
    // ring at m1, ccw: toward c1 (0), m2 (90), c0 (180)
    tp.splice( eBR, eC );
    tp.splice( eC, eBL.sym() );
    // ring at m2, ccw: toward c2 (0), c3 (180), m1 (270)
    tp.splice( eTR.sym(), eTL );
    tp.splice( eTL, eC.sym() );
    tp.setOrg( eBL, c0 );
    tp.setOrg( eBR, m1 );
    tp.setOrg( eR, c1 );
    tp.setOrg( eTR, c2 );
    tp.setOrg( eTL, m2 );
    tp.setOrg( eL, c3 );

    const EdgeLoops loops = { { eBL, eC, eTL, eL }, { eBR, eR, eTR, eC.sym() } };
    std::vector<EdgePath> bds;
    const auto res = PlanarTriangulation::triangulateDisjointContours( mesh, loops, SignedAxis::PlusZ, &bds );
    ASSERT_TRUE( res.has_value() );
    const Mesh& patch = *res;

    EXPECT_NEAR( patch.area(), 9.0f, 1e-5f ); // both sides of the chord filled
    EXPECT_EQ( patch.topology.findHoleRepresentiveEdges().size(), 1 ); // one outer boundary, chord interior
    for ( auto f : patch.topology.getValidFaces() )
        EXPECT_GT( dot( patch.normal( f ), Vector3f::plusZ() ), 0.f );
    checkPatchBorders( mesh, loops, patch, bds );
    // the two loops' chord entries are one shared patch edge seen from both sides, as in the mesh
    ASSERT_EQ( bds.size(), size_t( 2 ) );
    ASSERT_EQ( bds[0].size(), size_t( 4 ) );
    ASSERT_EQ( bds[1].size(), size_t( 4 ) );
    EXPECT_EQ( bds[0][1], bds[1][3].sym() );
    EXPECT_TRUE( patch.topology.left( bds[0][1] ).valid() );
    EXPECT_TRUE( patch.topology.right( bds[0][1] ).valid() );
}

TEST( MRMesh, PlanarTriangulationMeshSpaceAnnulus )
{
    // nested square loops with no shared vertices: the copy path must reproduce the classic case exactly
    const std::vector<Vector3f> outer = { { 0, 0, 0 }, { 3, 0, 0 }, { 3, 3, 0 }, { 0, 3, 0 } };
    const std::vector<Vector3f> inner = { { 1, 1, 0 }, { 1, 2, 0 }, { 2, 2, 0 }, { 2, 1, 0 } }; // cw: a hole in the fill

    Mesh mesh;
    const EdgeId o0 = mesh.addSeparateEdgeLoop( outer );
    const EdgeId i0 = mesh.addSeparateEdgeLoop( inner );
    EdgeLoops loops( 2 );
    loops[0] = trackRightBoundaryLoop( mesh.topology, o0 );
    loops[1] = trackRightBoundaryLoop( mesh.topology, i0 );
    ASSERT_EQ( loops[0].size(), size_t( 4 ) );
    ASSERT_EQ( loops[1].size(), size_t( 4 ) );

    std::vector<EdgePath> bds;
    const auto res = PlanarTriangulation::triangulateDisjointContours( mesh, loops, SignedAxis::PlusZ, &bds );
    ASSERT_TRUE( res.has_value() );
    const Mesh& patch = *res;

    EXPECT_NEAR( patch.area(), 8.0f, 1e-5f ); // 9 - 1
    EXPECT_EQ( patch.topology.numValidFaces(), 8 ); // annulus on 8 boundary verts
    EXPECT_EQ( patch.topology.findHoleRepresentiveEdges().size(), 2 );
    checkPatchBorders( mesh, loops, patch, bds );
}

namespace
{
// circle of n points (closed: first == last)
Contour2d circle( int n, double r, const Vector2d& center )
{
    Contour2d cont;
    cont.reserve( n + 1 );
    for ( int i = 0; i < n; ++i )
    {
        const double a = 2.0 * PI * i / n;
        cont.push_back( center + Vector2d( r * std::cos( a ), r * std::sin( a ) ) );
    }
    cont.push_back( cont.front() );
    return cont;
}
}

TEST( MRMesh, PlanarTriangulationMergeSame1 )
{
    Contours2d conts( 2 );
    conts[0] = circle( 10, 20, Vector2d( 100, 100 ) );
    conts[1] =
    {
        {0.0,0.0},
        {0.0,0.0},

        {5.0,10.0},
        {15.0,15.0},
        {20.0,5.0},

        {0.0,0.0},
        {0.0,0.0}
    };
    auto mesh = PlanarTriangulation::triangulateContours( conts );
    EXPECT_NEAR( mesh.area(), -calcOrientedArea( conts[0] ) + calcOrientedArea( conts[1] ), 1e-3 );
}

TEST( MRMesh, PlanarTriangulationMergeSame2 )
{
    Contours2d conts( 2 );
    conts[0] = circle( 10, 20, Vector2d( 100, 100 ) );
    conts[1] =
    {
        {0.0,0.0},
        {0.0,0.0},
        {0.0,0.0},
        {0.0,0.0}
    };
    auto mesh = PlanarTriangulation::triangulateContours( conts );
    EXPECT_NEAR( mesh.area(), -calcOrientedArea( conts[0] ), 1e-3 );
}

TEST( MRMesh, PlanarTriangulationMergeSame3 )
{
    Contours2d conts( 2 );
    conts[0] = circle( 10, 20, Vector2d( 0.0, 0.0 ) );
    conts[1] =
    {
        {0.0,0.0},
        {45.0,45.0},
        {0.0,0.0},
        {0.0,0.0}
    };
    auto mesh = PlanarTriangulation::triangulateContours( conts );
    EXPECT_NEAR( mesh.area(), -calcOrientedArea( conts[0] ), 1e-3 );
}

namespace
{

// star polygon {n/step} as a single self-intersecting closed contour (needs gcd(n,step)==1)
Contour2d benchStar( int n, int step, double r, const Vector2d& center )
{
    Contour2d cont;
    cont.reserve( n + 1 );
    for ( int i = 0; i < n; ++i )
    {
        const int idx = ( i * step ) % n;
        const double a = 2.0 * PI * idx / n;
        cont.push_back( center + Vector2d( r * std::cos( a ), r * std::sin( a ) ) );
    }
    cont.push_back( cont.front() );
    return cont;
}

template <typename Contours>
size_t countVerts( const Contours& cs )
{
    size_t n = 0;
    for ( const auto& c : cs )
        n += c.size();
    return n;
}

template <typename Contours>
double triangulateOnceMs( const Contours& conts )
{
    const auto t0 = std::chrono::steady_clock::now();
    Mesh m = PlanarTriangulation::triangulateContours( conts );
    const auto t1 = std::chrono::steady_clock::now();
    volatile size_t sink = m.topology.faceSize();
    (void)sink;
    return std::chrono::duration<double, std::milli>( t1 - t0 ).count();
}

// runs `once` (returns elapsed ms) warmup+iters times, prints min/median/mean
void runBench( const char* name, size_t nverts, int warmup, int iters, const std::function<double()>& once )
{
    for ( int i = 0; i < warmup; ++i )
        once();
    std::vector<double> ts;
    ts.reserve( iters );
    for ( int i = 0; i < iters; ++i )
        ts.push_back( once() );
    std::sort( ts.begin(), ts.end() );
    double sum = 0.0;
    for ( double t : ts )
        sum += t;
    std::printf( "[BENCH] %-22s verts=%-8zu min=%9.3f median=%9.3f mean=%9.3f ms\n",
        name, nverts, ts.front(), ts[ts.size() / 2], sum / ts.size() );
    std::fflush( stdout );
}

} // anonymous namespace

// local A/B benchmark for the SweepLineQueue predicate refactor; opt-in:
//   MRTest.exe --gtest_also_run_disabled_tests --gtest_filter=*PlanarTriangulationBench*
// Order matters for interleaved (DLL-swap) A/B: the priority sort-bound workload runs
// FIRST (measured from a cool CPU), the heavy sort-insensitive control runs LAST.
TEST( MRMesh, DISABLED_PlanarTriangulationBench )
{
    constexpr int warmup = 3, iters = 30;

    // 1) one big circle: single large monotone polygon -> dominated by the `less` sort.
    //    This is the path the predicate refactor regressed and parallel_sort targets.
    {
        Contours2d conts{ circle( 100000, 1.0, Vector2d() ) };
        runBench( "one-big-circle", countVerts( conts ), warmup, iters,
            [&] { return triangulateOnceMs( conts ); } );
    }

    // 2) many disjoint circles: sort/sweep/monotone bound (stresses `less` + `ccw`, ~no intersections)
    {
        Contours2d conts;
        constexpr int grid = 24, ptsPer = 48;
        for ( int gx = 0; gx < grid; ++gx )
            for ( int gy = 0; gy < grid; ++gy )
                conts.push_back( circle( ptsPer, 0.4, Vector2d( double( gx ), double( gy ) ) ) );
        runBench( "disjoint-circles", countVerts( conts ), warmup, iters,
            [&] { return triangulateOnceMs( conts ); } );
    }

    // 3) grid of overlapping circles: many cross-contour intersections
    {
        Contours2d conts;
        constexpr int grid = 10, ptsPer = 40;
        for ( int gx = 0; gx < grid; ++gx )
            for ( int gy = 0; gy < grid; ++gy )
                conts.push_back( circle( ptsPer, 0.5, Vector2d( 0.8 * gx, 0.8 * gy ) ) );
        runBench( "overlapping-circles", countVerts( conts ), warmup, iters,
            [&] { return triangulateOnceMs( conts ); } );
    }

    // 4) single heavily self-intersecting star polygon
    {
        Contours2d conts{ benchStar( 101, 10, 1.0, Vector2d() ) }; // gcd(101,10)==1 -> one loop
        runBench( "self-intersecting-star", countVerts( conts ), warmup, iters,
            [&] { return triangulateOnceMs( conts ); } );
    }

    // 5) text outlines: many contours, letters with holes (multi-contour + winding)
    {
        SymbolMeshParams sp;
        sp.text = "MeshLib planar triangulation 0123456789 quick brown fox";
        auto exp = createSymbolContours( sp );
        if ( exp.has_value() && !exp->empty() )
        {
            const Contours2f& tc = *exp;
            runBench( "text-symbols", countVerts( tc ), warmup, iters,
                [&] { return triangulateOnceMs( tc ); } );
        }
        else
            std::printf( "[BENCH] text-symbols           SKIPPED (createSymbolContours failed)\n" );
    }

    // 6) CONTROL: real cross-sections of a torus. Each slice contour is small, so the per-slice
    //    sort is below parallel_sort's serial cutoff -> this workload is ~insensitive to the sort
    //    change. If interleaved A/B shows B ~= M here, the measurement method is validated.
    //    Runs LAST because it is the heaviest (CPU-heating) workload.
    {
        const Mesh torus = makeTorus( 2.0f, 0.7f, 256, 64 );
        const Box3f bb = torus.computeBoundingBox();
        std::vector<Contours2f> slices;
        constexpr int nSlices = 16;
        for ( int i = 1; i < nSlices; ++i )
        {
            const float z = bb.min.z + ( bb.max.z - bb.min.z ) * float( i ) / float( nSlices );
            const PlaneSections sec = extractXYPlaneSections( torus, z );
            Contours2f cs = planeSectionsToContours2f( torus, sec, AffineXf3f() );
            Contours2f closed;
            for ( auto& c : cs )
                if ( c.size() >= 3 )
                {
                    if ( c.front() != c.back() )
                        c.push_back( c.front() );
                    closed.push_back( std::move( c ) );
                }
            if ( !closed.empty() )
                slices.push_back( std::move( closed ) );
        }
        size_t nv = 0;
        for ( const auto& s : slices )
            nv += countVerts( s );
        auto once = [&] ()
        {
            double ms = 0.0;
            size_t faces = 0;
            for ( const auto& s : slices )
            {
                const auto t0 = std::chrono::steady_clock::now();
                Mesh m = PlanarTriangulation::triangulateContours( s );
                const auto t1 = std::chrono::steady_clock::now();
                ms += std::chrono::duration<double, std::milli>( t1 - t0 ).count();
                faces += m.topology.faceSize();
            }
            volatile size_t sink = faces;
            (void)sink;
            return ms;
        };
        if ( !slices.empty() )
            runBench( "mesh-slices(torus)", nv, warmup, iters, once );
        else
            std::printf( "[BENCH] mesh-slices(torus)     SKIPPED (no sections)\n" );
    }
}

} //namespace MR
