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

    const auto res = PlanarTriangulation::triangulateDisjointContours( mesh, EdgeLoops{ loop }, normal );
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

// circle of n points (closed: first == last)
Contour2d benchCircle( int n, double r, const Vector2d& center )
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
        Contours2d conts{ benchCircle( 100000, 1.0, Vector2d() ) };
        runBench( "one-big-circle", countVerts( conts ), warmup, iters,
            [&] { return triangulateOnceMs( conts ); } );
    }

    // 2) many disjoint circles: sort/sweep/monotone bound (stresses `less` + `ccw`, ~no intersections)
    {
        Contours2d conts;
        constexpr int grid = 24, ptsPer = 48;
        for ( int gx = 0; gx < grid; ++gx )
            for ( int gy = 0; gy < grid; ++gy )
                conts.push_back( benchCircle( ptsPer, 0.4, Vector2d( double( gx ), double( gy ) ) ) );
        runBench( "disjoint-circles", countVerts( conts ), warmup, iters,
            [&] { return triangulateOnceMs( conts ); } );
    }

    // 3) grid of overlapping circles: many cross-contour intersections
    {
        Contours2d conts;
        constexpr int grid = 10, ptsPer = 40;
        for ( int gx = 0; gx < grid; ++gx )
            for ( int gy = 0; gy < grid; ++gy )
                conts.push_back( benchCircle( ptsPer, 0.5, Vector2d( 0.8 * gx, 0.8 * gy ) ) );
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
