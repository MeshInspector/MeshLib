#include <MRMesh/MRFillContours2D.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMeshTrimWithPlane.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MR2DContoursTriangulation.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRphmap.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cfloat>
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

TEST( MRMesh, fillContours2DPlanCacheReuse )
{
    // the plan path is where the cache really retains state between runs (the patch topology, its
    // points, and the input->patch edge maps all survive), so plan several holes through one cache
    // and compare every plan with the same call on a fresh cache
    VertCoords points;
    points.vec_ = {
        { 0.f, 0.f, 0.f },                                                     // the pinch vertex
        { 1.2f, -0.4f, 0.f }, { 1.5f, 0.9f, 0.f }, { 0.2f, 1.6f, 0.f }, { -1.1f, 0.9f, 0.f }, // outer arc
        { 0.7f, 0.3f, 0.f }, { 0.1f, 0.75f, 0.f },                             // island touching the pinch
        { 1.9f, -1.4f, 0.f }, { 2.8f, 1.5f, 0.f }, { 0.3f, 2.8f, 0.f },        // outer pentagon
        { -2.6f, 0.6f, 0.f }, { -1.2f, -1.7f, 0.f }
    };
    const Triangulation t{
        { VertId( 0 ), VertId( 5 ), VertId( 6 ) },   // the island
        { VertId( 0 ), VertId( 4 ), VertId( 10 ) }, { VertId( 0 ), VertId( 10 ), VertId( 11 ) },
        { VertId( 0 ), VertId( 11 ), VertId( 7 ) }, { VertId( 0 ), VertId( 7 ), VertId( 1 ) },
        { VertId( 1 ), VertId( 7 ), VertId( 2 ) }, { VertId( 2 ), VertId( 7 ), VertId( 8 ) },
        { VertId( 2 ), VertId( 8 ), VertId( 3 ) }, { VertId( 3 ), VertId( 8 ), VertId( 9 ) },
        { VertId( 3 ), VertId( 9 ), VertId( 4 ) }, { VertId( 4 ), VertId( 9 ), VertId( 10 ) } };
    const Mesh mesh = Mesh::fromTriangles( points, t );

    // the pinched 8-edge cavity and the outer pentagon outline
    EdgeId pinched, pentagon;
    for ( EdgeId e : mesh.topology.findHoleRepresentiveEdges() )
    {
        if ( trackRightBoundaryLoop( mesh.topology, e ).size() == 8 )
            pinched = e;
        else
            pentagon = e;
    }
    ASSERT_TRUE( pinched.valid() && pentagon.valid() );

    // the same hole twice in a row, then the other one, then back: every run must see a clean cache
    const std::vector<EdgeId> order{ pinched, pinched, pentagon, pinched, pentagon };
    auto cache = makeFillContours2DPlanCache();
    for ( EdgeId e : order )
    {
        const auto fresh = fillContours2DPlan( mesh, e );
        const auto cached = fillContours2DPlan( mesh, e, cache.get() );
        ASSERT_TRUE( fresh.has_value() ) << fresh.error();
        ASSERT_TRUE( cached.has_value() ) << cached.error();
        EXPECT_EQ( fresh->numTris, cached->numTris );
        ASSERT_EQ( fresh->items.size(), cached->items.size() );
        for ( size_t i = 0; i < fresh->items.size(); ++i )
        {
            EXPECT_EQ( fresh->items[i].edgeCode1, cached->items[i].edgeCode1 );
            EXPECT_EQ( fresh->items[i].edgeCode2, cached->items[i].edgeCode2 );
        }
    }
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

// A polygon in the plane z = 0, with chords: polylines of edges with no faces between two of its vertices, so the hole inside
// the polygon runs along every chord in both directions, as in a face cut by crossing contours
struct PolygonWithChords
{
    Mesh mesh;
    EdgeId e0; // the first polygon edge, with the inside of the polygon on its left
    std::vector<EdgePath> chords; // the edges of every chord, from its first polygon vertex to its last one
};

struct ChordSpec
{
    int from = 0, to = 0; // the polygon vertices at the ends of the chord, each with at most one chord
    std::vector<Vector3f> via; // the vertices of the chord inside the polygon
};

// the polygon is counterclockwise around +Z
static PolygonWithChords makePolygonWithChords( const std::vector<Vector3f>& polygon, const std::vector<ChordSpec>& chords )
{
    PolygonWithChords res;
    res.e0 = res.mesh.addSeparateEdgeLoop( polygon );
    auto& tp = res.mesh.topology;
    const EdgeLoop sides = trackRightBoundaryLoop( tp, res.e0 );
    for ( int i = 0; i < int( polygon.size() ); ++i )
        EXPECT_EQ( res.mesh.orgPnt( sides[i] ), polygon[i] );
    for ( const auto& spec : chords )
    {
        // a chord goes into the polygon right after the polygon edge from its vertex
        EdgePath& path = res.chords.emplace_back();
        EdgeId prev = sides[spec.from];
        for ( size_t i = 0; i <= spec.via.size(); ++i )
        {
            const EdgeId e = tp.makeEdge();
            tp.splice( prev, e );
            if ( i < spec.via.size() )
            {
                tp.setOrg( e.sym(), res.mesh.addPoint( spec.via[i] ) );
                prev = e.sym();
            }
            else
                tp.splice( sides[spec.to], e.sym() );
            path.push_back( e );
        }
    }
    return res;
}

static std::vector<Vector3f> regularPolygon( int n, float r, float angle0 = 0.1f )
{
    std::vector<Vector3f> res;
    for ( int i = 0; i < n; ++i )
    {
        const float a = angle0 + 2 * PI_F * i / n;
        res.emplace_back( r * std::cos( a ), r * std::sin( a ), 0.f );
    }
    return res;
}

// the points where the chords properly cross each other in the projection on z = 0
static std::vector<Vector3f> chordCrossings( const Mesh& mesh, const std::vector<EdgePath>& chords )
{
    std::vector<Vector3f> res;
    for ( size_t i = 0; i < chords.size(); ++i )
        for ( size_t j = i + 1; j < chords.size(); ++j )
            for ( EdgeId a : chords[i] )
                for ( EdgeId b : chords[j] )
                {
                    const Vector3d a0( mesh.orgPnt( a ) ), a1( mesh.destPnt( a ) ), b0( mesh.orgPnt( b ) ), b1( mesh.destPnt( b ) );
                    const Vector2d r( a1.x - a0.x, a1.y - a0.y ), s( b1.x - b0.x, b1.y - b0.y ), ab( b0.x - a0.x, b0.y - a0.y );
                    const double d = cross( r, s );
                    if ( d == 0 )
                        continue;
                    const double t = cross( ab, s ) / d, u = cross( ab, r ) / d;
                    if ( t > 0 && t < 1 && u > 0 && u < 1 )
                        res.push_back( Vector3f( a0 + t * ( a1 - a0 ) ) );
                }
    return res;
}

// the hole loops inside the polygon: several ones if its chords split it topologically, whatever their crossings
static EdgeLoops innerLoops( const PolygonWithChords& pc )
{
    EdgeLoops res;
    for ( EdgeId e : pc.mesh.topology.findHoleRepresentiveEdges() )
    {
        auto loop = trackRightBoundaryLoop( pc.mesh.topology, e );
        if ( std::find( loop.begin(), loop.end(), pc.e0.sym() ) == loop.end() ) // not the outside of the polygon
            res.push_back( std::move( loop ) );
    }
    return res;
}

static const std::string cOuterCrossing = "an edge of the region boundary crosses another edge";

// the error of splitInnerCrossings, which must leave the mesh as it is then
static std::string splitError( Mesh& mesh, const EdgeLoops& loops, const Vector3f& normal )
{
    const auto numEdges = mesh.topology.edgeSize();
    const auto numVerts = mesh.topology.vertSize();
    const auto res = PlanarTriangulation::splitInnerCrossings( mesh, loops, normal );
    EXPECT_EQ( mesh.topology.edgeSize(), numEdges );
    EXPECT_EQ( mesh.topology.vertSize(), numVerts );
    return res ? "no error" : res.error();
}

static float pathLength( const Mesh& mesh, const EdgePath& path )
{
    float res = 0;
    for ( EdgeId e : path )
        res += mesh.edgeLength( e.undirected() );
    return res;
}

// splits the chords at their crossings, and puts the pieces in the chords, as cutMesh is to do in its paths;
// unless the chords touch exactly, checks the new vertices against the crossings in the plane; returns the number of new vertices
static int splitAndCheck( PolygonWithChords& pc, bool exactTouches = false )
{
    Mesh& mesh = pc.mesh;
    const auto expected = chordCrossings( mesh, pc.chords );
    const auto numVerts = mesh.topology.numValidVerts();
    const VertId firstNewVert( mesh.topology.vertSize() );
    std::vector<float> chordLengths;
    for ( const auto& path : pc.chords )
        chordLengths.push_back( pathLength( mesh, path ) );
    const auto pieces = PlanarTriangulation::splitInnerCrossings( mesh, innerLoops( pc ), Vector3f::plusZ() );
    if ( !pieces )
    {
        ADD_FAILURE() << pieces.error();
        return 0;
    }
    EXPECT_TRUE( mesh.topology.checkValidity() );
    const VertId endNewVert( mesh.topology.vertSize() );
    const int numNewVerts = int( endNewVert ) - int( firstNewVert );
    EXPECT_EQ( mesh.topology.numValidVerts(), numVerts + numNewVerts );
    if ( !exactTouches )
    {
        EXPECT_EQ( numNewVerts, int( expected.size() ) );
        for ( const auto& p : expected )
        {
            float minDist = FLT_MAX;
            for ( VertId v = firstNewVert; v < endNewVert; ++v )
                minDist = std::min( minDist, ( p - mesh.points[v] ).length() );
            EXPECT_LT( minDist, 1e-5f );
        }
    }
    for ( VertId v = firstNewVert; v < endNewVert; ++v )
    {
        // four pieces around the vertex of each crossing, counterclockwise
        int degree = 0;
        for ( EdgeId e : orgRing( mesh.topology, v ) )
        {
            ++degree;
            if ( !exactTouches )
            {
                EXPECT_GT( cross( mesh.edgeVector( e ), mesh.edgeVector( mesh.topology.next( e ) ) ).z, 0.f );
            }
        }
        EXPECT_EQ( degree, 4 );
    }

    size_t numPieces = 0;
    for ( const auto& [ue, path] : *pieces )
    {
        numPieces += path.size();
        EXPECT_EQ( path.back(), EdgeId( ue ) );
    }
    EXPECT_EQ( numPieces, pieces->size() + 2 * numNewVerts );
    for ( size_t i = 0; i < pc.chords.size(); ++i )
    {
        EdgePath path;
        for ( EdgeId e : pc.chords[i] )
        {
            const auto it = pieces->find( e.undirected() );
            if ( it == pieces->end() )
                path.push_back( e );
            else if ( e.even() )
                path.insert( path.end(), it->second.begin(), it->second.end() );
            else
                for ( auto rit = it->second.rbegin(); rit != it->second.rend(); ++rit )
                    path.push_back( rit->sym() );
        }
        for ( size_t j = 0; j + 1 < path.size(); ++j )
            EXPECT_EQ( mesh.topology.dest( path[j] ), mesh.topology.org( path[j + 1] ) );
        EXPECT_NEAR( pathLength( mesh, path ), chordLengths[i], 1e-5f );
        pc.chords[i] = std::move( path );
    }
    return numNewVerts;
}

// fills the holes inside the polygon with the usual planners, as cutMesh does, and checks that the filling covers the polygon once
// with triangles of double area above minDblArea
static void fillAndCheck( PolygonWithChords& pc, float minDblArea = 1e-4f )
{
    Mesh& mesh = pc.mesh;
    std::vector<EdgeId> holes;
    Vector3d dirArea; // of any surface spanning the loops inside the polygon
    for ( const auto& loop : innerLoops( pc ) )
    {
        holes.push_back( loop.front() );
        for ( EdgeId e : loop )
            dirArea += cross( Vector3d( mesh.orgPnt( e ) ), Vector3d( mesh.destPnt( e ) ) );
    }
    dirArea *= 0.5;
    auto plans = getPlanarHoleFillPlans( mesh, holes );
    FaceBitSet newFaces;
    for ( size_t i = 0; i < holes.size(); ++i )
    {
        if ( !isFillingMultipleEdgeFree( mesh.topology, plans[i] ) )
            plans[i] = getPlanarHoleFillPlan( mesh, holes[i], false );
        executeHoleFillPlan( mesh, holes[i], plans[i], &newFaces );
    }
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), size_t( 1 ) ); // only the outside of the polygon is left
    EXPECT_LT( ( mesh.dirArea( newFaces ) - dirArea ).length(), 1e-5 );
    for ( FaceId f : newFaces )
        EXPECT_GT( mesh.dirDblArea( f ).z, minDblArea ); // no triangle is upside down or of zero area
}

static int splitFillAndCheck( PolygonWithChords& pc )
{
    const int res = splitAndCheck( pc );
    fillAndCheck( pc );
    return res;
}

// Two chords crossing in the middle of a polygon, in one hole that the planners cannot fill as it is
TEST( MRMesh, splitInnerCrossingsTwoChords )
{
    auto pc = makePolygonWithChords( regularPolygon( 12, 2 ), { { 0, 6 }, { 3, 9 } } );
    EXPECT_FALSE( fillContours2DPlan( pc.mesh, pc.e0 ).has_value() );
    EXPECT_EQ( splitFillAndCheck( pc ), 1 );
}

// Three chords crossing pairwise: going along each of them, the other two come in cyclic order.
// They split the hole in two loops, each running every chord once, so only both loops together have the chords inside
TEST( MRMesh, splitInnerCrossingsThreeChords )
{
    auto pc = makePolygonWithChords( regularPolygon( 12, 2 ), { { 0, 5 }, { 4, 9 }, { 8, 1 } } );
    const auto loops = innerLoops( pc );
    ASSERT_EQ( loops.size(), size_t( 2 ) );
    for ( const auto& loop : loops )
        EXPECT_EQ( splitError( pc.mesh, EdgeLoops{ loop }, Vector3f::plusZ() ), cOuterCrossing );
    EXPECT_EQ( splitFillAndCheck( pc ), 3 );
}

// A chord cutting off a corner beside two crossing ones: the corner is a hole of three edges
TEST( MRMesh, splitInnerCrossingsCornerCutOff )
{
    auto pc = makePolygonWithChords( regularPolygon( 12, 2 ), { { 0, 6 }, { 3, 8 }, { 9, 11 } } );
    ASSERT_EQ( innerLoops( pc ).size(), size_t( 2 ) );
    EXPECT_EQ( splitFillAndCheck( pc ), 1 );
}

// Chords crossed several times
TEST( MRMesh, splitInnerCrossingsManyOnOneChord )
{
    // two chords, one going back and forth over the other one
    {
        auto pc = makePolygonWithChords( regularPolygon( 16, 2 ),
            { { 0, 8 }, { 4, 12, { { -0.6f, -0.5f, 0.f }, { 0.1f, 0.6f, 0.f }, { 0.5f, -0.6f, 0.f } } } } );
        EXPECT_EQ( splitFillAndCheck( pc ), 3 );
    }
    // one chord crossed by three others, once by two of them and three times by one more, in three loops
    {
        auto pc = makePolygonWithChords( regularPolygon( 16, 2 ),
            { { 0, 8 }, { 2, 14 }, { 4, 12, { { -0.5f, -0.3f, 0.f }, { 0.2f, 0.4f, 0.f } } }, { 6, 10 } } );
        EXPECT_EQ( innerLoops( pc ).size(), size_t( 3 ) );
        EXPECT_EQ( splitFillAndCheck( pc ), 5 );
    }
}

// Holes large enough for the swept line triangulation after the split
TEST( MRMesh, splitInnerCrossingsLargeHoles )
{
    auto pc = makePolygonWithChords( regularPolygon( 64, 2 ), { { 0, 32 }, { 16, 48 } } );
    EXPECT_EQ( splitAndCheck( pc ), 1 );
    const auto loops = innerLoops( pc );
    EXPECT_EQ( loops.size(), size_t( 4 ) );
    for ( const auto& loop : loops )
    {
        EXPECT_EQ( loop.size(), size_t( 18 ) );
        const auto plan = fillContours2DPlan( pc.mesh, loop.front() );
        EXPECT_TRUE( plan.has_value() ) << plan.error();
    }
    fillAndCheck( pc );
}

// A crossing involving the boundary of the region is an error
TEST( MRMesh, splitInnerCrossingsOuter )
{
    const std::vector<Vector3f> notched{ { 0, 0, 0 }, { 4, 0, 0 }, { 4, 4, 0 }, { 2.5f, 4, 0 }, { 2, 1, 0 }, { 1.5f, 4, 0 }, { 0, 4, 0 } };
    // a chord leaving the region through the notch, crossing two of its edges: it splits the hole in two
    {
        auto pc = makePolygonWithChords( notched, { { 6, 1 } } );
        EXPECT_EQ( innerLoops( pc ).size(), size_t( 2 ) );
        EXPECT_EQ( splitError( pc.mesh, innerLoops( pc ), Vector3f::plusZ() ), cOuterCrossing );
    }
    // the same chord crossing another one inside the region too, which makes one hole
    {
        auto pc = makePolygonWithChords( notched, { { 6, 1 }, { 0, 5 } } );
        EXPECT_EQ( innerLoops( pc ).size(), size_t( 1 ) );
        EXPECT_EQ( splitError( pc.mesh, innerLoops( pc ), Vector3f::plusZ() ), cOuterCrossing );
    }
    // a pentagon whose vertex D, lifted, pokes through the opposite edge AB in the projection: the boundary crosses itself
    {
        Mesh mesh;
        const EdgeId e0 = mesh.addSeparateEdgeLoop( { { 0, 0, 0 }, { 4, 0, 0 }, { 4, 2, 0 }, { 2, -0.1f, 0.2f }, { 0, 2, 0 } } );
        EXPECT_EQ( splitError( mesh, { trackRightBoundaryLoop( mesh.topology, e0 ) }, Vector3f::plusZ() ), cOuterCrossing );
        EXPECT_EQ( splitError( mesh, { trackRightBoundaryLoop( mesh.topology, e0.sym() ) }, Vector3f::minusZ() ), cOuterCrossing );
    }
    // a loop spiraling twice around the middle
    {
        constexpr int n = 15;
        std::vector<Vector3f> pts;
        for ( int i = 0; i < n; ++i )
        {
            const float a = 4 * PI_F * i / n, r = 1 - 0.3f * i / n;
            pts.emplace_back( r * std::cos( a ), r * std::sin( a ), 0.01f * i );
        }
        Mesh mesh;
        const EdgeId e = mesh.addSeparateEdgeLoop( pts );
        EXPECT_EQ( splitError( mesh, { trackRightBoundaryLoop( mesh.topology, e ) }, Vector3f::plusZ() ), cOuterCrossing );
    }
}

// A flat ring around a gap, and an island disk raised above it, sticking out of the gap over the ring:
// the island's loop crosses the ring's inner loop twice, which are both boundaries of the region
TEST( MRMesh, splitInnerCrossingsIslandOverRing )
{
    constexpr int n = 16, m = 12;
    VertCoords pts;
    for ( int ring = 0; ring < 2; ++ring ) // circles of radius 2 and 3
        for ( int i = 0; i < n; ++i )
        {
            const float a = 2 * PI_F * i / n + 0.1f; // off the axes, so no two vertices share an x
            pts.push_back( Vector3f( ( ring + 2 ) * std::cos( a ), ( ring + 2 ) * std::sin( a ), 0.f ) );
        }
    pts.push_back( Vector3f( 1.5f, 0.f, 0.1f ) ); // the island center
    for ( int i = 0; i < m; ++i )
    {
        const float a = 2 * PI_F * i / m + 0.2f;
        pts.push_back( Vector3f( 1.5f + std::cos( a ), std::sin( a ), 0.1f ) );
    }
    auto rv = [] ( int ring, int i ) { return VertId( ring * n + i % n ); };
    auto iv = [] ( int i ) { return VertId( 2 * n + 1 + i % m ); };
    Triangulation t;
    for ( int i = 0; i < n; ++i )
    {
        t.push_back( { rv( 0, i ), rv( 1, i ), rv( 1, i + 1 ) } );
        t.push_back( { rv( 0, i ), rv( 1, i + 1 ), rv( 0, i + 1 ) } );
    }
    for ( int i = 0; i < m; ++i )
        t.push_back( { VertId( 2 * n ), iv( i ), iv( i + 1 ) } );
    Mesh mesh = Mesh::fromTriangles( std::move( pts ), t );

    // the loops of the gap: the ring's inner boundary and the island's boundary, not the ring's outer one
    EdgeLoops loops = findRightBoundary( mesh.topology );
    std::erase_if( loops, [&] ( const EdgeLoop& l ) { return mesh.orgPnt( l.front() ).lengthSq() > 8.f; } );
    ASSERT_EQ( loops.size(), size_t( 2 ) );
    EXPECT_EQ( splitError( mesh, loops, Vector3f::plusZ() ), cOuterCrossing );
}

// Chords touching exactly or nearly: simulation of simplicity turns exact touches in proper crossings,
// whose vertices can then coincide with other ones, giving zero-area triangles
TEST( MRMesh, splitInnerCrossingsTouch )
{
    const std::vector<Vector3f> octagon{ { 2, 0, 0 }, { 1.5f, 1.5f, 0 }, { 0, 2, 0 }, { -1.5f, 1.5f, 0 }, { -2, 0, 0 }, { -1.5f, -1.5f, 0 }, { 0, -2, 0 }, { 1.5f, -1.5f, 0 } };
    struct Case
    {
        std::vector<ChordSpec> chords;
        int numCrossings = 0;
        bool exact = false;
    };
    const Case cases[] =
    {
        { { { 0, 4 }, { 2, 6, { { 0.5f, 0, 0 } } } }, 1, true }, // a chord bending exactly on the other one
        { { { 0, 4 }, { 2, 6 }, { 1, 5 } }, 3, true }, // three chords through one point
        { { { 0, 4 }, { 2, 6, { { 0.5f, -0.25f, 0 } } } }, 1, false }, // a bend near the other chord
        { { { 0, 4 }, { 2, 6 }, { 1, 5, { { 0.1f, -0.05f, 0 } } } }, 3, false } // three crossings near one point
    };
    for ( const auto& c : cases )
    {
        auto pc = makePolygonWithChords( octagon, c.chords );
        EXPECT_EQ( splitAndCheck( pc, c.exact ), c.numCrossings );
        fillAndCheck( pc, c.exact ? -1e-6f : 1e-4f );
    }
}

}
