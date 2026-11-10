#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRObjectMeshData.h>
#include <MRMesh/MRMeshSubdivide.h>
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRLineSegm.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRphmap.h>
#include <MRMesh/MRRegularGridMesh.h>
#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MRMeshMetrics.h>
#include <MRMesh/MRMeshMeshDistance.h>
#include <MRMesh/MRClosestPointInTriangle.h>
#include <MRMesh/MRPlane3.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <gtest/gtest.h>

namespace MR
{

namespace
{

const Color cStock( 150, 60, 60 );
const Color cTool( 0, 255, 0 );

// a stock box with a cylindrical pocket, the faces from the cylinder have the tool color
ObjectMeshData makeColoredCut()
{
    auto stock = makeCube( Vector3f( 100, 100, 52 ), Vector3f( -50, -50, -50 ) );
    subdivideMesh( stock, { .maxEdgeLen = 4.0f, .maxEdgeSplits = 10'000'000 } );
    const auto tool = makeCylinder( 5.0f, 10.0f, 32 );
    const auto xf = AffineXf3f::translation( Vector3f( 20, 0, -4 ) );
    BooleanResultMapper mapper;
    auto res = boolean( stock, tool, BooleanOperation::DifferenceAB, &xf, &mapper );
    EXPECT_TRUE( res.valid() );
    ObjectMeshData c;
    c.mesh = std::make_shared<Mesh>( std::move( res.mesh ) );
    c.faceColors.resize( c.mesh->topology.faceSize(), cStock );
    for ( auto f : mapper.map( tool.topology.getValidFaces(), BooleanResultMapper::MapObject::B ) )
        c.faceColors[f] = cTool;
    return c;
}

void copyNewColors( FaceColors & colors, const FaceHashMap & new2Old, size_t faceSize )
{
    colors.resize( faceSize );
    for ( const auto & [nf, of] : new2Old )
        colors[nf] = colors[of];
}

// directed edges with a stock face on the left and a tool face on the right
std::vector<EdgeId> borderEdges( const MeshTopology & t, const FaceColors & colors )
{
    std::vector<EdgeId> res;
    for ( EdgeId e( 0 ); e < t.edgeSize(); ++e )
    {
        const auto l = t.left( e ), r = t.right( e );
        if ( l && r && colors[l] == cStock && colors[r] == cTool )
            res.push_back( e );
    }
    return res;
}

// duplicates edge (eb): splits its left face and flips the new edge going to the vertex opposite to (eb)
void addMultipleEdge( Mesh & mesh, EdgeId eb )
{
    auto & t = mesh.topology;
    const auto c = t.dest( t.next( eb ) );
    const auto v = mesh.splitFace( t.left( eb ) );
    auto e = t.edgeWithOrg( v );
    while ( t.dest( e ) != c )
        e = t.next( e );
    t.flipEdge( e );
    mesh.invalidateCaches();
}

Vector3f lerpD( const Vector3f & p, const Vector3f & q, double t )
{
    return Vector3f( Vector3d( p ) + ( Vector3d( q ) - Vector3d( p ) ) * t );
}

// splits the left face of each given edge at the point (height) of the way from the edge's middle to the face's center
void addCaps( Mesh & mesh, FaceColors & colors, const std::vector<EdgeId> & edges, double height )
{
    auto & t = mesh.topology;
    FaceHashMap n2o;
    FaceBitSet split;
    for ( auto e : edges )
    {
        const auto f = t.left( e );
        if ( split.test( f ) )
            continue;
        split.autoResizeSet( f );
        const auto mid = lerpD( mesh.orgPnt( e ), mesh.destPnt( e ), 0.5 );
        mesh.splitFace( f, lerpD( mid, mesh.triCenter( f ), height ), nullptr, &n2o );
    }
    copyNewColors( colors, n2o, t.faceSize() );
}

// needle slivers and cap slivers on the color border
void addDefects( Mesh & mesh, FaceColors & colors )
{
    const auto be = borderEdges( mesh.topology, colors );
    ASSERT_GE( be.size(), 36 );
    FaceHashMap n2o;
    for ( size_t i = 5; i < 15; ++i )
        mesh.splitEdge( be[i], lerpD( mesh.orgPnt( be[i] ), mesh.destPnt( be[i] ), 1e-6 ), nullptr, &n2o );
    copyNewColors( colors, n2o, mesh.topology.faceSize() );

    addCaps( mesh, colors, { be.begin() + 20, be.begin() + 30 }, 1e-7 );
    // these are too high to be fixed within small deviation
    addCaps( mesh, colors, { be.begin() + 32, be.begin() + 36 }, 1e-4 );
    mesh.invalidateCaches();
}

// the colored cut with the defects and all attributes: texture ids and face selection following the colors,
// the color borders as creases, uv and vertex colors from the positions
ObjectMeshData makeDefectCase()
{
    auto data = makeColoredCut();
    addDefects( *data.mesh, data.faceColors );
    const auto & t = data.mesh->topology;
    data.texturePerFace.resize( t.faceSize() );
    for ( auto f : t.getValidFaces() )
    {
        const bool tool = data.faceColors[f] == cTool;
        data.texturePerFace[f] = TextureId( tool ? 1 : 0 );
        if ( tool )
            data.selectedFaces.autoResizeSet( f );
    }
    data.creases = edgesBetweenDifferentColors( t, data.faceColors );
    data.uvCoordinates.resize( t.vertSize() );
    data.vertColors.resize( t.vertSize() );
    for ( auto v : t.getValidVerts() )
    {
        const auto pos = data.mesh->points[v];
        data.uvCoordinates[v] = { pos.x, pos.y };
        data.vertColors[v] = Color( Vector3f( 0.5f, 0.5f, 0.5f ) + pos / 200.0f );
    }
    return data;
}

// checks that the attributes of the defect case are consistent with the mesh and with each other after the fixing
void checkAttributes( const ObjectMeshData & data, FixMeshDegeneraciesParams::Mode mode )
{
    const auto & t = data.mesh->topology;
    EXPECT_EQ( data.faceColors.size(), t.faceSize() );
    EXPECT_EQ( data.texturePerFace.size(), t.faceSize() );
    EXPECT_EQ( data.uvCoordinates.size(), t.vertSize() );
    EXPECT_EQ( data.vertColors.size(), t.vertSize() );
    EXPECT_TRUE( data.selectedFaces.is_subset_of( t.getValidFaces() ) );
    EXPECT_TRUE( data.creases.any() );
    for ( auto ue : data.creases )
        EXPECT_FALSE( t.isLoneEdge( ue ) );

    // the face attributes are copied together, and not blended
    for ( auto f : t.getValidFaces() )
    {
        const bool tool = data.faceColors[f] == cTool;
        EXPECT_TRUE( tool || data.faceColors[f] == cStock );
        EXPECT_EQ( data.texturePerFace[f], TextureId( tool ? 1 : 0 ) );
        EXPECT_EQ( data.selectedFaces.test( f ), tool );
    }

    // the vertices from edge splits get interpolated uv, and the vertices of a patch the uv of the removed surface nearby
    const float uvTolerance = mode == FixMeshDegeneraciesParams::Mode::RemeshPatch ? 2.0f : 1e-3f;
    for ( auto v : t.getValidVerts() )
    {
        const auto pos = data.mesh->points[v];
        EXPECT_LT( ( data.uvCoordinates[v] - Vector2f( pos.x, pos.y ) ).length(), uvTolerance );
    }
}

// total area of the faces with another color than the original surface at their centers
float wrongColorArea( const ObjectMeshData & data, const ObjectMeshData & original )
{
    float res = 0;
    for ( auto f : data.mesh->topology.getValidFaces() )
        if ( data.faceColors[f] != original.faceColors[findProjection( data.mesh->triCenter( f ), *original.mesh ).proj.face] )
            res += data.mesh->area( f );
    return res;
}

const FixMeshDegeneraciesParams cFixParams[] =
{
    { .maxDeviation = 1e-3f, .tinyEdgeLength = 1e-4f },
    // the highest caps remain after the decimation and the subdivision, and Mode::RemeshPatch patches them
    { .maxDeviation = 1e-6f, .tinyEdgeLength = 1e-6f },
    // the subdivision splits the thin triangles remaining after the decimation
    { .maxDeviation = 1e-3f, .tinyEdgeLength = 1e-4f, .criticalTriAspectRatio = 100 }
};

std::string traceName( const FixMeshDegeneraciesParams & p )
{
    return "maxDeviation=" + std::to_string( p.maxDeviation ) + " criticalTriAspectRatio=" + std::to_string( p.criticalTriAspectRatio ) + " mode=" + std::to_string( int( p.mode ) );
}

constexpr FixMeshDegeneraciesParams::Mode cModes[] =
{
    FixMeshDegeneraciesParams::Mode::Decimate,
    FixMeshDegeneraciesParams::Mode::Remesh,
    FixMeshDegeneraciesParams::Mode::RemeshPatch
};

} // namespace

TEST( MRMesh, FixMultipleEdgesNew2Old )
{
    Mesh mesh = makeCube();
    subdivideMesh( mesh, { .maxEdgeLen = 0.3f, .maxEdgeSplits = 1000 } );
    for ( auto p : { Vector3f( 0.6f, 0.1f, 0.1f ), Vector3f( -0.6f, -0.1f, 0.1f ) } )
        addMultipleEdge( mesh, mesh.topology.edgeWithLeft( findProjection( p, mesh ).proj.face ) );
    const auto multipleEdges = findMultipleEdges( mesh.topology ).value();
    ASSERT_EQ( multipleEdges.size(), 2 );

    const auto faceSize0 = mesh.topology.faceSize();
    const auto vertSize0 = mesh.topology.vertSize();
    Mesh ref = mesh;
    fixMultipleEdges( ref, multipleEdges );
    FaceHashMap new2Old;
    fixMultipleEdges( mesh, multipleEdges, &new2Old );
    EXPECT_TRUE( mesh == ref );
    EXPECT_FALSE( hasMultipleEdges( mesh.topology ) );

    const auto & t = mesh.topology;
    for ( FaceId f( faceSize0 ); f < t.faceSize(); ++f )
        EXPECT_EQ( t.hasFace( f ), new2Old.contains( f ) );
    EXPECT_EQ( new2Old.size(), 4 );
    for ( const auto & [nf, of] : new2Old )
    {
        EXPECT_LT( of, faceSize0 );
        EXPECT_TRUE( t.hasFace( of ) );
        // a new face is a part of the old one, and they share an edge incident to the new vertex
        bool sharesNewEdge = false;
        for ( auto e : leftRing( t, nf ) )
            if ( t.right( e ) == of && ( t.org( e ) >= vertSize0 || t.dest( e ) >= vertSize0 ) )
                sharesNewEdge = true;
        EXPECT_TRUE( sharesNewEdge );
    }
}

TEST( MRMesh, FixMeshDataDegeneraciesWithoutProtection )
{
    const auto c = makeDefectCase();
    ASSERT_GT( findDegenerateFaces( *c.mesh, 1e4f ).value().count(), 0 );
    for ( auto p : cFixParams )
    for ( auto mode : cModes )
    {
        p.mode = mode;
        p.protectAttributeBorders = false;
        SCOPED_TRACE( traceName( p ) );
        Mesh ref = *c.mesh;
        EXPECT_TRUE( fixMeshDegeneracies( ref, p ).has_value() );

        // the same mesh as from fixMeshDegeneracies
        auto data = c.clone();
        EXPECT_TRUE( fixMeshDataDegeneracies( data, p ).has_value() );
        EXPECT_TRUE( *data.mesh == ref );
        checkAttributes( data, mode );
    }
}

TEST( MRMesh, FixMeshDataDegeneracies )
{
    const auto original = makeColoredCut();
    std::vector<LineSegm3f> border;
    for ( auto ue : edgesBetweenDifferentColors( original.mesh->topology, original.faceColors ) )
        border.emplace_back( original.mesh->orgPnt( ue ), original.mesh->destPnt( ue ) );
    auto distToBorder = [&] ( const Vector3f & p )
    {
        float res = FLT_MAX;
        for ( const auto & s : border )
            res = std::min( res, ( closestPointOnLineSegm( p, s ) - p ).length() );
        return res;
    };

    const auto c = makeDefectCase();
    for ( auto p : cFixParams )
    for ( auto mode : cModes )
    {
        p.mode = mode;
        SCOPED_TRACE( traceName( p ) );
        auto data = c.clone();
        EXPECT_TRUE( fixMeshDataDegeneracies( data, p ).has_value() );
        checkAttributes( data, mode );
        const auto & t = data.mesh->topology;
        if ( p.criticalTriAspectRatio >= 1e4f )
        {
            EXPECT_LT( t.numValidFaces(), 1.1 * c.mesh->topology.numValidFaces() );
        }

        // only the degenerate triangles with the longest edge on the color border can remain, if they are too high to collapse
        // (and the thin triangles without short edges in Mode::Decimate, since it only collapses and flips edges)
        const auto degenerateFaces = findDegenerateFaces( *data.mesh, p.criticalTriAspectRatio ).value();
        for ( auto f : degenerateFaces )
        {
            EXPECT_NE( mode, FixMeshDegeneraciesParams::Mode::RemeshPatch );
            if ( mode == FixMeshDegeneraciesParams::Mode::Decimate && p.criticalTriAspectRatio < 1e4f )
                continue;
            EdgeId longest = t.edgeWithLeft( f );
            for ( auto e : leftRing( t, f ) )
                if ( data.mesh->edgeLengthSq( e ) > data.mesh->edgeLengthSq( longest ) )
                    longest = e;
            EXPECT_NE( data.faceColors[t.left( longest )], data.faceColors[t.right( longest )] );
        }

        // no face changes its color, and the creases stay on the color border
        EXPECT_LT( wrongColorArea( data, original ), 1e-4f );
        for ( auto ue : data.creases )
        {
            EXPECT_LT( distToBorder( data.mesh->orgPnt( ue ) ), 1e-2f );
            EXPECT_LT( distToBorder( data.mesh->destPnt( ue ) ), 1e-2f );
        }
    }
}

// two triangles with the same vertices glued along all three edges (a pillow) are eliminated with all their edges and vertices,
// only if asked
TEST( MRMesh, EliminateDoubleTrisPillow )
{
    auto t = MeshBuilder::fromTriangles( { { 0_v, 1_v, 2_v }, { 0_v, 2_v, 1_v } } );
    FaceBitSet region = t.getValidFaces();
    for ( auto e : leftRing( t, 0_f ) )
    {
        EXPECT_TRUE( isEdgeBetweenDoubleTris( t, e ) );
        EXPECT_FALSE( eliminateDoubleTris( t, e, &region ) );
    }
    EXPECT_EQ( t.numValidFaces(), 2 );

    const auto e = t.edgeWithLeft( 0_f );
    EXPECT_EQ( eliminateDoubleTris( t, e, &region, true ), e );
    EXPECT_EQ( t.numValidFaces(), 0 );
    EXPECT_EQ( t.numValidVerts(), 0 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 0 );
    EXPECT_TRUE( region.none() );
    EXPECT_TRUE( t.checkValidity() );

    // a lone triangle has the same topology with a hole of three edges instead of the other triangle: it is kept
    auto lone = MeshBuilder::fromTriangles( { { 0_v, 1_v, 2_v } } );
    for ( auto le : leftRing( lone, 0_f ) )
        for ( auto x : { le, le.sym() } )
            EXPECT_FALSE( eliminateDoubleTris( lone, x, nullptr, true ) );
    EXPECT_EQ( lone.numValidFaces(), 1 );
    EXPECT_EQ( lone.numValidVerts(), 3 );
}

// a flat square of 3x3 cells in the plane z = 0
static Mesh makeGrid3x3()
{
    return *makeRegularGridMesh( 4, 4, [] ( size_t, size_t ) { return true; },
        [] ( size_t x, size_t y ) { return Vector3f( float( x ), float( y ), 0 ); } );
}

static void moveVert( Mesh & mesh, const Vector3f & from, const Vector3f & to )
{
    for ( auto & p : mesh.points )
        if ( p == from )
            p = to;
}

// a vertex moved over its neighbour folds its triangles over the next ones, away from its own star
TEST( MRMesh, FixMeshCreasesFlatFold )
{
    auto folded = makeGrid3x3();
    moveVert( folded, { 2, 2, 0 }, { 2, 0.4f, 0 } );
    auto mesh = folded;
    FaceHashMap new2Old;
    fixMeshCreases( mesh, { .new2Old = &new2Old } );

    // all triangles face up and cover the square once
    for ( auto f : mesh.topology.getValidFaces() )
        EXPECT_GT( mesh.dirDblArea( f ).z, 0 );
    EXPECT_NEAR( mesh.area(), 9.0, 1e-5 );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.topology.findNumHoles(), 1 );

    // a new face takes an original face facing up under its center, the others keep their ids
    EXPECT_FALSE( new2Old.empty() );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        const auto it = new2Old.find( f );
        if ( it == new2Old.end() )
        {
            EXPECT_TRUE( folded.topology.hasFace( f ) && mesh.topology.getTriVerts( f ) == folded.topology.getTriVerts( f ) );
            continue;
        }
        EXPECT_GT( folded.dirDblArea( it->second ).z, 0 );
        const auto c = mesh.triCenter( f );
        const auto [a, b, d] = folded.getTriPoints( it->second );
        EXPECT_LT( ( closestPointInTriangle( c, a, b, d ).first - c ).length(), 1e-5f );
    }
}

// the top of a box folded by a vertex moved over its neighbour: fixing the fold keeps the box
TEST( MRMesh, FixMeshCreasesKeepsShape )
{
    auto box = makeGrid3x3();
    const auto e = extendHole( box, box.topology.findHoleRepresentiveEdges().front(), Plane3f( Vector3f::plusZ(), -1 ) );
    fillHole( box, e, { .metric = getPlaneFillMetric( box, e ) } );

    auto mesh = box;
    moveVert( mesh, { 2, 1, 0 }, { 0.5f, 1.4f, 0 } );
    fixMeshCreases( mesh );
    EXPECT_NEAR( mesh.area(), box.area(), 1e-4 );
    EXPECT_NEAR( mesh.volume(), box.volume(), 1e-4 );
    EXPECT_LT( findMaxDistanceSqOneWay( mesh, box ), 1e-10f );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.topology.findNumHoles(), 0 );
}

// a vertex next to the pole of a coarse sphere slid along its meridian past its neighbour folds the curved surface,
// which has no flat area around the fold: the fan of triangles around the fold is re-triangulated
TEST( MRMesh, FixMeshCreasesCurvedFold )
{
    auto mesh = makeUVSphere( 1, 12, 12 );
    const VertId v( 11 * 12 + 11 ), u( 10 * 12 + 11 ); // the last two rows of vertices are next to the pole
    mesh.points[v] = ( mesh.points[v] + 1.3f * ( mesh.points[u] - mesh.points[v] ) ).normalized();
    const auto creaseAngle = FixCreasesParams{}.creaseAngle;
    EXPECT_TRUE( mesh.findCreaseEdges( creaseAngle ).any() );
    fixMeshCreases( mesh );
    EXPECT_TRUE( mesh.findCreaseEdges( creaseAngle ).none() );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.topology.findNumHoles(), 0 );
}

TEST( MRMesh, FixMeshCreasesKeepsCurvedShape )
{
    const auto sphere = makeUVSphere( 1, 160, 160 );
    auto mesh = sphere;
    const VertId v( 16781 ), u( 16622 ); // next to each other
    mesh.points[v] = ( mesh.points[v] + 1.333f * ( mesh.points[u] - mesh.points[v] ) ).normalized();
    const auto creaseAngle = FixCreasesParams{}.creaseAngle;
    EXPECT_TRUE( mesh.findCreaseEdges( creaseAngle ).any() );
    fixMeshCreases( mesh );
    EXPECT_TRUE( mesh.findCreaseEdges( creaseAngle ).none() );
    // the faces around the fold are parallel within 5 degrees, but not in one plane: no area is re-triangulated in a plane,
    // and the fill of the fan around the moved vertex removes it
    EXPECT_LT( findMaxDistanceSq( mesh, sphere ), sqr( 0.0005f ) );
}

// a border strip folded under a sheet: the planar triangulation of the two layers would keep only the part of the sheet not covered by the strip
TEST( MRMesh, FixMeshCreasesFoldAtBorder )
{
    // the sheet is 8 x 4, and its strip of width 3 is folded under it along x = 0
    const auto folded = *makeRegularGridMesh( 12, 5, [] ( size_t, size_t ) { return true; },
        [] ( size_t x, size_t y ) { return Vector3f( std::abs( float( x ) - 3 ), float( y ), 0 ); } );
    auto mesh = folded;
    fixMeshCreases( mesh );
    // the fans re-filled around the fold line take at most one cell from its side, as they always did
    EXPECT_LT( findMaxDistanceSqOneWay( mesh, folded ), sqr( 1.5f ) );
    EXPECT_GT( mesh.area(), 32 );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.topology.findNumHoles(), 1 );
}

// a fold of a flat area and a fold of a curved area in one mesh: the planar triangulation of the first does not postpone
// the creases next to the fans re-filled before it, so one iteration leaves the curved fold as without the flat fold
TEST( MRMesh, FixMeshCreasesFlatAndCurvedFolds )
{
    auto curved = makeUVSphere( 1, 16, 16 );
    for ( auto & p : curved.points )
        p.x = std::min( p.x, 0.7f ); // a flat side
    for ( auto v : { 0_v, 31_v } ) // two neighbours of a vertex next to the pole slid past it
        curved.points[v] = ( curved.points[v] + 1.5f * ( curved.points[15_v] - curved.points[v] ) ).normalized();
    auto both = curved;
    both.points[100_v] += 1.3f * ( both.points[101_v] - both.points[100_v] ); // a vertex of the flat side moved over its neighbour
    const FixCreasesParams oneIter{ .maxIters = 1 };
    const auto numCreases = [&] ( const Mesh & m ) { return m.findCreaseEdges( oneIter.creaseAngle ).count(); };
    EXPECT_GT( numCreases( both ), numCreases( curved ) );
    fixMeshCreases( curved, oneIter );
    fixMeshCreases( both, oneIter );
    EXPECT_EQ( numCreases( both ), numCreases( curved ) );
}

} //namespace MR
