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
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MR2to3.h>
#include <MRSymbolMesh/MRSymbolMesh.h>
#include <gtest/gtest.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include <functional>
#include <utility>
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

// the sweep orders vertices by this coordinate alone: the axis most aligned with the normal is
// dropped, and of the two kept axes the first one is the sweep direction (they swap with the normal)
float sweepCoord( const Vector3f& p, const Vector3f& normal )
{
    int dropAx = 0;
    for ( int i = 1; i < 3; ++i )
        if ( normal[i] * normal[i] > normal[dropAx] * normal[dropAx] )
            dropAx = i;
    return p[( dropAx + ( normal[dropAx] < 0 ? 2 : 1 ) ) % 3];
}

// a loop is monotone when it is two chains that both advance with the sweep, so walking it the sweep
// coordinate turns around exactly twice; edges perpendicular to the sweep do not turn it around, the
// sweep resolves those by an infinitesimal perturbation
bool isMonotoneHole( const Mesh& mesh, EdgeId e0, const Vector3f& normal )
{
    const EdgeLoop loop = trackRightBoundaryLoop( mesh.topology, e0 );
    const int n = int( loop.size() );
    if ( n < 3 )
        return false;
    std::vector<int> dirs;
    for ( int i = 0; i < n; ++i )
    {
        const float a = sweepCoord( mesh.orgPnt( loop[i] ), normal );
        const float b = sweepCoord( mesh.orgPnt( loop[( i + 1 ) % n] ), normal );
        if ( a != b )
            dirs.push_back( a < b ? 1 : -1 );
    }
    int turns = 0;
    for ( int i = 0; i < int( dirs.size() ); ++i )
        if ( dirs[i] != dirs[( i + 1 ) % dirs.size()] )
            ++turns;
    return turns == 2;
}

// one representative edge per hole an executed monotone plan left behind: every part is incident to
// an original loop edge or to an added edge, so the left rings of those cover all the parts
std::vector<EdgeId> collectParts( const MeshTopology& tp, const EdgeLoops& loops, const HoleFillPlan& executedPlan )
{
    std::vector<EdgeId> starts;
    for ( const auto& loop : loops )
        starts.insert( starts.end(), loop.begin(), loop.end() );
    for ( const auto& item : executedPlan.items )
    {
        const EdgeId e( item.edgeCode1 ); // execution replaced the code with the created edge
        starts.push_back( e );
        starts.push_back( e.sym() );
    }
    std::vector<EdgeId> res;
    EdgeBitSet visited( tp.edgeSize() );
    for ( EdgeId s : starts )
    {
        if ( visited.test( s ) || tp.left( s ) )
            continue;
        res.push_back( s );
        for ( EdgeId e : leftRing( tp, s ) )
            visited.set( e );
    }
    return res;
}

std::vector<Vector3f> to3d( const std::vector<Vector2f>& pts )
{
    std::vector<Vector3f> res;
    res.reserve( pts.size() );
    for ( const auto& p : pts )
        res.push_back( to3dim( p ) );
    return res;
}

// the boundary loop through e0 that has the polygon interior on its left
EdgeLoop interiorLoop( const Mesh& mesh, EdgeId e0, const Vector3f& normal )
{
    EdgeLoop loop = trackRightBoundaryLoop( mesh.topology, e0 );
    Vector3f doubleArea;
    for ( EdgeId e : loop )
        doubleArea += cross( mesh.orgPnt( e ), mesh.destPnt( e ) );
    if ( dot( doubleArea, normal ) < 0 ) // the interior is on the left of the loop running ccw around +normal
        loop = trackRightBoundaryLoop( mesh.topology, e0.sym() );
    return loop;
}

// a mesh that is nothing but one closed edge loop, with that loop taken as above
std::pair<Mesh, EdgeLoop> makeFreeLoop( const std::vector<Vector3f>& poly, const Vector3f& normal )
{
    Mesh mesh;
    const EdgeId e0 = mesh.addSeparateEdgeLoop( poly );
    EdgeLoop loop = interiorLoop( mesh, e0, normal );
    return { std::move( mesh ), std::move( loop ) };
}

// the grid below is sheared a little, so that no two of its vertices share a sweep coordinate along
// either axis: then the cut the sweep makes does not depend on how it breaks ties between equally
// placed vertices, which is by vertex number and so by where the caller's loop happens to start
Vector3f gridPnt( int i, int j )
{
    return { float( i ) + 0.05f * float( j ), float( j ) + 0.05f * float( i ), 0.f };
}

// a grid of n x n quads with the cells `isHole` marks left out: the region to fill is then a real
// mesh hole with faces around it, and every coordinate stays exact
Mesh makeGridWithHoles( int n, const std::function<bool( int, int )>& isHole )
{
    std::vector<Vector3f> pts;
    pts.reserve( ( n + 1 ) * ( n + 1 ) );
    for ( int j = 0; j <= n; ++j )
        for ( int i = 0; i <= n; ++i )
            pts.push_back( gridPnt( i, j ) );
    auto vid = [n] ( int i, int j ) { return VertId( j * ( n + 1 ) + i ); };
    Triangulation t;
    for ( int j = 0; j < n; ++j )
        for ( int i = 0; i < n; ++i )
        {
            if ( isHole( i, j ) )
                continue; // the cell's two triangles and their shared diagonal are all left out
            t.push_back( { vid( i, j ), vid( i + 1, j ), vid( i + 1, j + 1 ) } );
            t.push_back( { vid( i, j ), vid( i + 1, j + 1 ), vid( i, j + 1 ) } );
        }
    return Mesh::fromTriangles( VertCoords( std::move( pts ) ), t );
}

// the hole loop of `mesh` running from `from` to `to`, so with that hole on its left; an edge tells
// two holes apart even where they meet in one vertex, which a point alone would not
EdgeLoop findHoleByEdge( const Mesh& mesh, const Vector3f& from, const Vector3f& to )
{
    for ( EdgeId e : mesh.topology.findHoleRepresentiveEdges() )
        for ( EdgeId le : trackRightBoundaryLoop( mesh.topology, e ) )
            if ( mesh.orgPnt( le ) == from && mesh.destPnt( le ) == to )
                return trackRightBoundaryLoop( mesh.topology, le );
    return {};
}

// the cells of a C, its notch opening toward +x, so that it splits a sweep along x
bool cCell( int i, int j )
{
    return ( j == 1 && i >= 1 && i <= 3 ) || ( i == 1 && j == 2 ) || ( j == 3 && i >= 1 && i <= 3 );
}

// the same shape turned a quarter, so that it splits a sweep along y instead
bool uCell( int i, int j )
{
    return ( j == 1 && i >= 1 && i <= 3 ) || ( i == 1 && j >= 2 && j <= 3 ) || ( i == 3 && j >= 2 && j <= 3 );
}

// a ring of cells around one kept in the middle: the kept cell touches nothing, so it is an island
// floating inside the region, and the ring is not monotone along either axis
bool ringCell( int i, int j )
{
    return i >= 1 && i <= 3 && j >= 1 && j <= 3 && !( i == 2 && j == 2 );
}

// executes the monotone plan and checks that the parts it leaves are monotone holes, and that
// filling them reproduces what the full sweep line triangulation of the same region produces
// \param expectedChords the number of edges the plan must add, or -1 not to check it
void checkMonotonePlan( Mesh mesh, const EdgeLoops& loops, const Vector3f& normal, int expectedChords )
{
    const auto refPatch = PlanarTriangulation::triangulateDisjointContours( mesh, loops, normal );
    ASSERT_TRUE( refPatch.has_value() );

    auto plan = PlanarTriangulation::getMonotonePlan( mesh, loops, normal );
    ASSERT_TRUE( plan.has_value() );
    EXPECT_EQ( plan->numTris, 0 ); // the plan only adds edges
    if ( expectedChords >= 0 )
        EXPECT_EQ( int( plan->items.size() ), expectedChords );
    if ( !plan->items.empty() )
        executeHoleFillPlan( mesh, loops[0][0], *plan );

    const auto parts = collectParts( mesh.topology, loops, *plan );
    EXPECT_FALSE( parts.empty() );
    for ( EdgeId e : parts )
        EXPECT_TRUE( isMonotoneHole( mesh, e, normal ) );

    const auto numFaces0 = mesh.topology.numValidFaces();
    const double area0 = mesh.area();
    auto fillPlans = getPlanarHoleFillPlans( mesh, parts );
    ASSERT_EQ( fillPlans.size(), parts.size() );
    for ( int i = 0; i < int( parts.size() ); ++i )
        executeHoleFillPlan( mesh, parts[i], fillPlans[i] );

    // a triangulation of the region that uses only its boundary vertices always has the same number
    // of triangles, whatever the diagonals are
    EXPECT_EQ( mesh.topology.numValidFaces() - numFaces0, refPatch->topology.numValidFaces() );
    EXPECT_NEAR( mesh.area() - area0, refPatch->area(), 1e-4 * refPatch->area() );
}

// a C opening toward +x: its notch splits the region in two arms as the sweep passes x = 1
const std::vector<Vector2f> cShapeCcw =
    { { 0.f, 0.f }, { 4.f, 0.f }, { 4.f, 1.f }, { 1.f, 1.f }, { 1.f, 3.f }, { 4.f, 3.f }, { 4.f, 4.f }, { 0.f, 4.f } };

// three arms on a spine, of increasing length so that they also end in sweep order: each of the two
// notches between them splits the region once
const std::vector<Vector2f> combCcw =
    { { 0.f, 0.f }, { 4.f, 0.f }, { 4.f, 1.f }, { 1.f, 1.f }, { 1.f, 2.f }, { 6.f, 2.f }, { 6.f, 3.f },
      { 1.f, 3.f }, { 1.f, 4.f }, { 8.f, 4.f }, { 8.f, 5.f }, { 0.f, 5.f } };

} // anonymous namespace

TEST( MRMesh, PlanarTriangulationMonotonePlanFreeLoops )
{
    const Vector3f normal = Vector3f::plusZ();

    { // one split vertex, so one chord
        const auto [mesh, loop] = makeFreeLoop( to3d( cShapeCcw ), normal );
        checkMonotonePlan( mesh, { loop }, normal, 1 );
    }

    { // one chord per notch, where the region splits off another arm
        const auto [mesh, loop] = makeFreeLoop( to3d( combCcw ), normal );
        checkMonotonePlan( mesh, { loop }, normal, 2 );
    }

    { // a convex loop is already monotone: an empty plan, not a failure
        const auto [mesh, loop] = makeFreeLoop( to3d( { { 0.f, 0.f }, { 4.f, 0.f }, { 4.f, 4.f }, { 0.f, 4.f } } ), normal );
        const auto plan = PlanarTriangulation::getMonotonePlan( mesh, { loop }, normal );
        ASSERT_TRUE( plan.has_value() );
        EXPECT_TRUE( plan->items.empty() );
        checkMonotonePlan( mesh, { loop }, normal, 0 );
    }

    { // the same C shape carried onto a plane tilted off all axes, in the mesh's own 3d space:
      // its z is whatever the plane dictates, so only the projection the sweep runs on is the C
        const Vector3f tilted = Vector3f( 1.f, 2.f, 3.f ).normalized();
        const Vector3f center( 10.f, -5.f, 2.f );
        std::vector<Vector3f> poly;
        for ( const auto& p : cShapeCcw )
        {
            const float z = center.z - ( tilted.x * ( p.x - center.x ) + tilted.y * ( p.y - center.y ) ) / tilted.z;
            poly.push_back( { p.x, p.y, z } );
        }
        const auto [mesh, loop] = makeFreeLoop( poly, tilted );
        checkMonotonePlan( mesh, { loop }, tilted, 1 );
    }
}

TEST( MRMesh, PlanarTriangulationMonotonePlanInMesh )
{
    const Vector3f normal = Vector3f::plusZ();

    { // the anchors are now real mesh hole edges, so the wedge guard has something to check
        const Mesh mesh = makeGridWithHoles( 5, cCell );
        const EdgeLoop loop = findHoleByEdge( mesh, gridPnt( 1, 1 ), gridPnt( 2, 1 ) );
        ASSERT_EQ( loop.size(), size_t( 16 ) );
        checkMonotonePlan( mesh, { loop }, normal, 1 );

        // which way round the loops run is not part of the input: the sweep takes the region from the
        // winding number, and the anchors translate back to the same mesh edges either way. It may
        // still cut the region differently, because reversing the loop renumbers the vertices and the
        // sweep breaks ties between equally placed ones by their number
        EdgeLoop backwards;
        for ( auto it = loop.rbegin(); it != loop.rend(); ++it )
            backwards.push_back( it->sym() );
        checkMonotonePlan( mesh, { backwards }, normal, -1 );
    }

    { // two holes pinched together at one vertex, which then has two hole sectors in the mesh ring;
      // each of them is monotone on its own, so a correct plan adds nothing at all here
        const Mesh mesh = makeGridWithHoles( 4, [] ( int i, int j ) { return ( i == 1 && j == 1 ) || ( i == 2 && j == 2 ); } );
        const EdgeLoop l1 = findHoleByEdge( mesh, gridPnt( 1, 1 ), gridPnt( 2, 1 ) );
        const EdgeLoop l2 = findHoleByEdge( mesh, gridPnt( 2, 2 ), gridPnt( 3, 2 ) );
        ASSERT_EQ( l1.size(), size_t( 4 ) );
        ASSERT_EQ( l2.size(), size_t( 4 ) );
        checkMonotonePlan( mesh, { l1, l2 }, normal, 0 );
    }

    { // an annulus: the island touches nothing, so only a chord can bridge it to the outer loop -
      // one where the sweep splits the region around it, and one where it closes back up
        const Mesh mesh = makeGridWithHoles( 5, ringCell );
        const EdgeLoop lRing = findHoleByEdge( mesh, gridPnt( 1, 1 ), gridPnt( 2, 1 ) );
        const EdgeLoop lIsland = findHoleByEdge( mesh, gridPnt( 3, 2 ), gridPnt( 2, 2 ) );
        ASSERT_EQ( lRing.size(), size_t( 12 ) );
        ASSERT_EQ( lIsland.size(), size_t( 4 ) );
        checkMonotonePlan( mesh, { lRing, lIsland }, normal, 2 );
    }

    { // a mesh oriented the other way round, swept around -normal: that swaps the two kept axes, so
      // the sweep now runs along y and the region has to be notched the other way to need a chord
        Mesh mesh = makeGridWithHoles( 5, uCell );
        mesh.topology.flipOrientation();
        const EdgeLoop loop = findHoleByEdge( mesh, gridPnt( 2, 1 ), gridPnt( 1, 1 ) );
        ASSERT_EQ( loop.size(), size_t( 16 ) );
        checkMonotonePlan( mesh, { loop }, -normal, 1 );
    }
}

TEST( MRMesh, PlanarTriangulationMonotonePlanRejects )
{
    const Vector3f normal = Vector3f::plusZ();
    // an annulus is not monotone along either axis, so it needs chords whichever way it is swept
    const Mesh mesh = makeGridWithHoles( 5, ringCell );
    const EdgeLoop lRing = findHoleByEdge( mesh, gridPnt( 1, 1 ), gridPnt( 2, 1 ) );
    const EdgeLoop lIsland = findHoleByEdge( mesh, gridPnt( 3, 2 ), gridPnt( 2, 2 ) );
    ASSERT_EQ( lRing.size(), size_t( 12 ) );
    ASSERT_EQ( lIsland.size(), size_t( 4 ) );

    // asking for the region around the opposite normal swaps the two kept axes, which mirrors the
    // plane the sweep works in: every chord would then be spliced in a wedge that mesh faces occupy,
    // and the wedge guard rejects the plan instead of corrupting the mesh
    EXPECT_FALSE( PlanarTriangulation::getMonotonePlan( mesh, { lRing, lIsland }, -normal ).has_value() );

    // a loop of less than 3 edges is not a contour the sweep can take
    EXPECT_FALSE( PlanarTriangulation::getMonotonePlan( mesh, { EdgeLoop{ lIsland[0], lIsland[1] } }, normal ).has_value() );

    { // intersecting loops: there is no region to decompose, the caller keeps its per loop path
        Mesh two;
        const EdgeId a0 = two.addSeparateEdgeLoop( to3d( { { 0.f, 0.f }, { 4.f, 0.f }, { 4.f, 4.f }, { 0.f, 4.f } } ) );
        const EdgeId b0 = two.addSeparateEdgeLoop( to3d( { { 2.f, 2.f }, { 6.f, 2.f }, { 6.f, 6.f }, { 2.f, 6.f } } ) );
        const EdgeLoops crossing = { interiorLoop( two, a0, normal ), interiorLoop( two, b0, normal ) };
        EXPECT_FALSE( PlanarTriangulation::getMonotonePlan( two, crossing, normal ).has_value() );
    }
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
