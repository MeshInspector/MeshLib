#include <MRMesh/MRFillContours2D.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMeshTrimWithPlane.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MR2DContoursTriangulation.h>
#include <gtest/gtest.h>
#include <limits>

namespace MR
{

TEST( MRMesh, fillContours2D )
{
    Mesh sphereBig = makeUVSphere( 1.0f, 32, 32 );
    Mesh sphereSmall = makeUVSphere( 0.7f, 16, 16 );

    sphereSmall.topology.flipOrientation();
    sphereBig.addMesh( sphereSmall );

    trimWithPlane( sphereBig, TrimWithPlaneParams{ .plane = Plane3f::fromDirAndPt( Vector3f::plusZ(), Vector3f() ) } );
    sphereBig.pack();

    auto firstNewFace = sphereBig.topology.lastValidFace() + 1;
    auto v = fillContours2D( sphereBig, sphereBig.topology.findHoleRepresentiveEdges() );
    EXPECT_TRUE( v.has_value() );
    for ( FaceId f = firstNewFace; f <= sphereBig.topology.lastValidFace(); ++f )
    {
        EXPECT_TRUE( std::abs( dot( sphereBig.dirDblArea( f ).normalized(), Vector3f::minusZ() ) - 1.0f ) < std::numeric_limits<float>::epsilon() );
    }
}

// Characterizes fillContours2D on a hole lying in a plane tilted off all axes (so the projection /
// dominant-axis path is exercised). Pins the observable result so the upcoming mesh-space rewrite
// cannot change it blindly: the hole must close watertight, reuse the boundary vertices in place
// (no drift or duplication), and the patch must lie in the cut plane.
TEST( MRMesh, fillContours2DTiltedPlane )
{
    Mesh mesh = makeUVSphere( 1.0f, 32, 32 );
    const Vector3f normal = Vector3f( 1.f, 2.f, 3.f ).normalized();
    trimWithPlane( mesh, TrimWithPlaneParams{ .plane = Plane3f::fromDirAndPt( normal, Vector3f() ) } );
    mesh.pack();

    ASSERT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), size_t( 1 ) );
    const int vertsBefore = mesh.topology.numValidVerts();
    const FaceId firstNewFace = mesh.topology.lastValidFace() + 1;

    const auto res = fillContours2D( mesh, mesh.topology.findHoleRepresentiveEdges() );
    EXPECT_TRUE( res.has_value() );

    // hole closed watertight, with the boundary vertices reused in place (no new / duplicated verts)
    EXPECT_TRUE( mesh.topology.findHoleRepresentiveEdges().empty() );
    EXPECT_EQ( mesh.topology.numValidVerts(), vertsBefore );

    // every patch face lies in the cut plane, oriented toward the removed -normal side (this also rules
    // out degenerate faces, whose normal would not align). Triangle quality is intentionally not pinned:
    // the current fill emits slivers, and the exact triangulation is not a stable invariant to guard.
    for ( FaceId f = firstNewFace; f <= mesh.topology.lastValidFace(); ++f )
        EXPECT_GT( dot( mesh.normal( f ), -normal ), 0.99f );
}

// A pinched hole: an island triangle touches the cavity's outer arc at one vertex, so the hole boundary
// is a single 8-edge loop passing through that vertex twice
TEST( MRMesh, fillContours2DPlanPinchedHole )
{
    VertCoords points;
    points.vec_ = {
        { 0.f, 0.f, 0.f },                                                     // the pinch vertex
        { 1.2f, -0.4f, 0.f }, { 1.5f, 0.9f, 0.f }, { 0.2f, 1.6f, 0.f }, { -1.1f, 0.9f, 0.f }, // outer arc
        { 0.7f, 0.3f, 0.f }, { 0.1f, 0.75f, 0.f },                             // island touching the pinch
        { 1.9f, -1.4f, 0.f }, { 2.8f, 1.5f, 0.f }, { 0.3f, 2.8f, 0.f },        // outer pentagon
        { -2.6f, 0.6f, 0.f }, { -1.2f, -1.7f, 0.f }
    }; // generic position: no two boundary directions at the pinch are collinear (ties are SoS-fragile)
    const Triangulation t{
        { VertId( 0 ), VertId( 5 ), VertId( 6 ) },   // the island
        { VertId( 0 ), VertId( 4 ), VertId( 10 ) }, { VertId( 0 ), VertId( 10 ), VertId( 11 ) },
        { VertId( 0 ), VertId( 11 ), VertId( 7 ) }, { VertId( 0 ), VertId( 7 ), VertId( 1 ) },
        { VertId( 1 ), VertId( 7 ), VertId( 2 ) }, { VertId( 2 ), VertId( 7 ), VertId( 8 ) },
        { VertId( 2 ), VertId( 8 ), VertId( 3 ) }, { VertId( 3 ), VertId( 8 ), VertId( 9 ) },
        { VertId( 3 ), VertId( 9 ), VertId( 4 ) }, { VertId( 4 ), VertId( 9 ), VertId( 10 ) } };
    Mesh mesh = Mesh::fromTriangles( points, t );
    ASSERT_EQ( mesh.topology.numValidFaces(), 11 );

    // two holes: the outer pentagon outline and the pinched cavity; the cavity is the 8-edge one
    EdgeId holeEdge;
    for ( EdgeId e : mesh.topology.findHoleRepresentiveEdges() )
        if ( trackRightBoundaryLoop( mesh.topology, e ).size() == 8 )
            holeEdge = e;
    ASSERT_TRUE( holeEdge.valid() );
    const EdgeLoop loop = trackRightBoundaryLoop( mesh.topology, holeEdge );
    int pinchVisits = 0;
    for ( EdgeId e : loop )
        if ( mesh.topology.org( e ) == VertId( 0 ) )
            ++pinchVisits;
    ASSERT_EQ( pinchVisits, 2 );

    auto plan = fillContours2DPlan( mesh, holeEdge );
    ASSERT_TRUE( plan.has_value() ) << plan.error();
    EXPECT_EQ( plan->numTris, 6 ); // the pinched 8-gon fills like a disk: n - 2 triangles
    EXPECT_EQ( plan->items.size(), size_t( 5 ) ); // and n - 3 chords
    executeHoleFillPlan( mesh, holeEdge, *plan );
    EXPECT_EQ( mesh.topology.numValidFaces(), 17 );
    EXPECT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), size_t( 1 ) ); // the pentagon outline remains
    auto multiples = findMultipleEdges( mesh.topology );
    EXPECT_TRUE( multiples.has_value() && multiples->empty() );
}

// The classic sandclock: two triangles joined at one shared vertex, whose single pinched 6-edge hole
// loop is filled from the other side. The 2-triangle mirror patch is pinched at that vertex too, so
// stitching it would close each lobe into its own pillow and split the vertex into two disjoint edge
// rings; addPartByMask detects this and fillContours2D must fail cleanly, leaving the mesh unchanged.
// The plan-based fillContours2DPlan bridges such loops disk-like instead (see fillContours2DPlanPinchedHole).
TEST( MRMesh, fillContours2DPinchedHoleValidity )
{
    VertCoords points;
    points.vec_ = {
        { -1.f,  1.f, 0.f }, {  1.f,  1.f, 0.f },
        {  0.f,  0.f, 0.f }, // the shared middle vertex
        {  1.f, -1.f, 0.f }, { -1.f, -1.f, 0.f } };
    const Triangulation t{
        { VertId( 0 ), VertId( 2 ), VertId( 1 ) },
        { VertId( 2 ), VertId( 4 ), VertId( 3 ) } };
    Mesh mesh = Mesh::fromTriangles( points, t );
    ASSERT_EQ( mesh.topology.numValidVerts(), 5 );
    ASSERT_TRUE( mesh.topology.checkValidity() );
    const auto holes = mesh.topology.findHoleRepresentiveEdges();
    ASSERT_EQ( holes.size(), size_t( 1 ) );
    ASSERT_EQ( trackRightBoundaryLoop( mesh.topology, holes[0] ).size(), size_t( 6 ) );

    const auto res = fillContours2D( mesh, holes );
    EXPECT_FALSE( res.has_value() );

    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 5 );
    EXPECT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), size_t( 1 ) );
    EXPECT_TRUE( mesh.topology.checkValidity() );
}

// The mesh both tests below fill: two triangular lobes (areas 0.141 and 0.029, generic position)
// joined at one shared vertex, so the hole boundary is a single 6-edge loop passing that vertex
// twice and winding counterclockwise around +Z - the configuration whose sweep frame is the
// identity (x,y) with no axis swap
static Mesh makePinchedTwoLobeMesh()
{
    VertCoords points;
    points.vec_ = {
        { -0.381682f, -0.002250f, 0.f }, { -0.923765f,  0.363788f, 0.f },
        {  0.057481f,  0.276528f, 0.f }, {  0.626216f,  0.839981f, 0.f },
        { -0.037916f,  0.285011f, 0.f } }; // the pinch vertex, on neither lobe's outer edge
    const Triangulation t{
        { VertId( 3 ), VertId( 2 ), VertId( 4 ) },
        { VertId( 4 ), VertId( 0 ), VertId( 1 ) } };
    return Mesh::fromTriangles( points, t );
}

// total area of the faces starting at `first`, and whether they all face the same way
static void patchAreaAndFolding( const Mesh& mesh, FaceId first, double& area, bool& folded )
{
    Vector3d sumDir;
    double sumAbs = 0;
    for ( FaceId f = first; f <= mesh.topology.lastValidFace(); ++f )
    {
        if ( !mesh.topology.hasFace( f ) )
            continue;
        const Vector3d d( mesh.dirDblArea( f ) );
        sumDir += d;
        sumAbs += d.length();
    }
    area = 0.5 * sumAbs;
    // a patch that covers each region once has |sum of directed areas| == sum of their lengths
    folded = sumAbs > 0 && sumDir.length() < sumAbs * ( 1 - 1e-6 );
}

// A pinched boundary cannot be closed by a per-lobe mirror patch in valid topology (the pinch vertex
// would be split into two disjoint edge rings), so fillContours2D must reject it cleanly, leaving the
// mesh unchanged; the triangulation itself is checked by the test below.
TEST( MRMesh, fillContours2DPinchedBoundary )
{
    Mesh mesh = makePinchedTwoLobeMesh();
    ASSERT_EQ( mesh.topology.numValidFaces(), 2 );
    ASSERT_EQ( mesh.topology.numValidVerts(), 5 ); // the pinch vertex is shared
    const auto holes = mesh.topology.findHoleRepresentiveEdges();
    ASSERT_EQ( holes.size(), size_t( 1 ) );
    ASSERT_EQ( trackRightBoundaryLoop( mesh.topology, holes[0] ).size(), size_t( 6 ) );

    const auto res = fillContours2D( mesh, holes );
    EXPECT_FALSE( res.has_value() );

    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );
    EXPECT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), size_t( 1 ) );
    EXPECT_TRUE( mesh.topology.checkValidity() );
}

// The triangulation itself must mirror the two lobes: 2 triangles, total area equal to the mesh's
// own, all facing the same way - not a folded sheet tiling the whole quadrilateral.
TEST( MRMesh, triangulateDisjointContoursPinchedLoop )
{
    const Mesh mesh = makePinchedTwoLobeMesh();
    const auto holes = mesh.topology.findHoleRepresentiveEdges();
    ASSERT_EQ( holes.size(), size_t( 1 ) );
    EdgeLoops loops{ trackRightBoundaryLoop( mesh.topology, holes[0] ) };

    Vector3d sumCross;
    for ( EdgeId e : loops[0] )
        sumCross += cross( Vector3d( mesh.orgPnt( e ) ), Vector3d( mesh.destPnt( e ) ) );
    EXPECT_GT( sumCross.z, 0. ); // the +Z configuration: identity sweep frame
    const auto patch = PlanarTriangulation::triangulateDisjointContours( mesh, loops, Vector3f( sumCross.normalized() ) );
    ASSERT_TRUE( patch.has_value() );

    double patchArea = 0;
    bool folded = true;
    patchAreaAndFolding( *patch, FaceId( 0 ), patchArea, folded );
    EXPECT_FALSE( folded );
    EXPECT_NEAR( patchArea, mesh.area(), 1e-5 );
    EXPECT_EQ( patch->topology.numValidFaces(), 2 );
}

}
