#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectMeshData.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRMesh/MRMeshSubdivide.h>
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRphmap.h>
#include <gtest/gtest.h>

namespace MR
{

namespace
{

const Color cStock( 150, 60, 60 );
const Color cTool( 0, 255, 0 );

// a stock box with a cylindrical pocket, the faces from the cylinder have the tool color
struct ColoredCut
{
    Mesh mesh;
    FaceColors colors;
};

ColoredCut makeColoredCut()
{
    auto stock = makeCube( Vector3f( 100, 100, 52 ), Vector3f( -50, -50, -50 ) );
    subdivideMesh( stock, { .maxEdgeLen = 4.0f, .maxEdgeSplits = 10'000'000 } );
    const auto tool = makeCylinder( 5.0f, 10.0f, 32 );
    const auto xf = AffineXf3f::translation( Vector3f( 20, 0, -4 ) );
    BooleanResultMapper mapper;
    auto res = boolean( stock, tool, BooleanOperation::DifferenceAB, &xf, &mapper );
    EXPECT_TRUE( res.valid() );
    ColoredCut c;
    c.mesh = std::move( res.mesh );
    c.colors.resize( c.mesh.topology.faceSize(), cStock );
    for ( auto f : mapper.map( tool.topology.getValidFaces(), BooleanResultMapper::MapObject::B ) )
        c.colors[f] = cTool;
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

// a multiple edge, needle slivers and cap slivers on the color border, and cap slivers on sharp edges between stock faces
void addDefects( Mesh & mesh, FaceColors & colors )
{
    auto & t = mesh.topology;
    const auto be = borderEdges( t, colors );
    ASSERT_GE( be.size(), 40 );
    const auto splitColor = colors[t.left( be[0] )];
    addMultipleEdge( mesh, be[0] );
    colors.resize( t.faceSize(), splitColor );

    FaceHashMap n2o;
    for ( size_t i = 5; i < 15; ++i )
        mesh.splitEdge( be[i], lerpD( mesh.orgPnt( be[i] ), mesh.destPnt( be[i] ), 1e-6 ), nullptr, &n2o );
    copyNewColors( colors, n2o, t.faceSize() );

    addCaps( mesh, colors, { be.begin() + 20, be.begin() + 30 }, 1e-7 );
    // these are too high to be fixed within small deviation
    addCaps( mesh, colors, { be.begin() + 32, be.begin() + 36 }, 1e-4 );

    std::vector<EdgeId> sharpEdges;
    for ( EdgeId e( 0 ); e < t.edgeSize(); e += 2 )
        if ( t.left( e ) && t.right( e ) && colors[t.left( e )] == cStock && colors[t.right( e )] == cStock
            && std::abs( dot( mesh.normal( t.left( e ) ), mesh.normal( t.right( e ) ) ) ) < 0.1f )
            sharpEdges.push_back( e );
    ASSERT_GE( sharpEdges.size(), 60 );
    std::vector<EdgeId> someSharpEdges;
    for ( size_t i = 0; i < sharpEdges.size(); i += sharpEdges.size() / 6 )
        someSharpEdges.push_back( sharpEdges[i] );
    addCaps( mesh, colors, someSharpEdges, 1e-7 );
    mesh.invalidateCaches();
}

// the colored cut with the defects (multiple edges already fixed), and the original colored cut
struct DefectCase
{
    ObjectMeshData data;
    ColoredCut original;
};

DefectCase makeDefectCase()
{
    DefectCase res;
    res.original = makeColoredCut();
    auto mesh = std::make_shared<Mesh>( res.original.mesh );
    auto colors = res.original.colors;
    addDefects( *mesh, colors );

    FaceHashMap new2Old;
    fixMultipleEdges( *mesh, findMultipleEdges( mesh->topology ).value(), &new2Old );
    copyNewColors( colors, new2Old, mesh->topology.faceSize() );

    res.data.mesh = std::move( mesh );
    res.data.faceColors = std::move( colors );
    return res;
}

// total area of the faces with another color than the original surface at their centers
float wrongColorArea( const ObjectMeshData & data, const ColoredCut & original )
{
    float res = 0;
    for ( auto f : data.mesh->topology.getValidFaces() )
        if ( data.faceColors[f] != original.colors[findProjection( data.mesh->triCenter( f ), original.mesh ).proj.face] )
            res += data.mesh->area( f );
    return res;
}

constexpr FixMeshDegeneraciesParams::Mode cModes[] =
{
    FixMeshDegeneraciesParams::Mode::Decimate,
    FixMeshDegeneraciesParams::Mode::Remesh,
    FixMeshDegeneraciesParams::Mode::RemeshPatch
};

FixMeshDegeneraciesParams fixParams( FixMeshDegeneraciesParams::Mode mode )
{
    return { .maxDeviation = 1e-3f, .tinyEdgeLength = 1e-4f, .mode = mode };
}

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

TEST( MRMesh, FixMeshDataDegeneraciesWithoutAttributes )
{
    const auto c = makeDefectCase();
    for ( auto mode : cModes )
    {
        Mesh ref = *c.data.mesh;
        EXPECT_TRUE( fixMeshDegeneracies( ref, fixParams( mode ) ).has_value() );

        ObjectMeshData data;
        data.mesh = std::make_shared<Mesh>( *c.data.mesh );
        EXPECT_TRUE( fixMeshDataDegeneracies( data, fixParams( mode ) ).has_value() );
        EXPECT_TRUE( *data.mesh == ref );
        EXPECT_TRUE( data.faceColors.empty() );
        EXPECT_TRUE( data.vertColors.empty() );
    }
}

TEST( MRMesh, FixMeshDataDegeneraciesKeepsColors )
{
    const auto c = makeDefectCase();
    const auto & t0 = c.data.mesh->topology;
    ASSERT_GT( findDegenerateFaces( *c.data.mesh, 1e4f ).value().count(), 0 );

    for ( auto p : {
        // decimation fixes everything
        FixMeshDegeneraciesParams{ .maxDeviation = 1e-3f, .tinyEdgeLength = 1e-4f },
        // the highest caps on the color border remain after decimation, and are patched in Mode::RemeshPatch
        FixMeshDegeneraciesParams{ .maxDeviation = 1e-6f, .tinyEdgeLength = 1e-6f },
        // subdivision fixes the thin triangles remaining after decimation
        FixMeshDegeneraciesParams{ .maxDeviation = 1e-3f, .tinyEdgeLength = 1e-4f, .criticalTriAspectRatio = 100 } } )
    for ( auto mode : cModes )
    {
        SCOPED_TRACE( "maxDeviation=" + std::to_string( p.maxDeviation ) + " criticalTriAspectRatio=" + std::to_string( p.criticalTriAspectRatio ) + " mode=" + std::to_string( int( mode ) ) );
        p.mode = mode;
        auto data = c.data.clone();
        const auto & t = data.mesh->topology;
        data.texturePerFace.resize( t.faceSize() );
        for ( auto f : t.getValidFaces() )
        {
            data.texturePerFace[f] = TextureId( data.faceColors[f] == cTool ? 1 : 0 );
            if ( data.mesh->triCenter( f ).x < 20 )
                data.selectedFaces.autoResizeSet( f );
        }
        data.uvCoordinates.resize( t.vertSize() );
        data.vertColors.resize( t.vertSize() );
        for ( auto v : t.getValidVerts() )
        {
            const auto pos = data.mesh->points[v];
            data.uvCoordinates[v] = { pos.x, pos.y };
            data.vertColors[v] = Color( Vector3f( 0.5f, 0.5f, 0.5f ) + pos / 200.0f );
        }

        EXPECT_TRUE( fixMeshDataDegeneracies( data, p ).has_value() );
        EXPECT_EQ( data.faceColors.size(), t.faceSize() );
        EXPECT_EQ( data.texturePerFace.size(), t.faceSize() );
        EXPECT_EQ( data.uvCoordinates.size(), t.vertSize() );
        EXPECT_EQ( data.vertColors.size(), t.vertSize() );
        EXPECT_TRUE( data.selectedFaces.is_subset_of( t.getValidFaces() ) );
        if ( p.criticalTriAspectRatio >= 1e4f )
        {
            EXPECT_LT( t.numValidFaces(), 1.1 * t0.numValidFaces() );
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

        // no color is changed by flips across the color border or blended, and textures change together with colors
        EXPECT_LT( wrongColorArea( data, c.original ), 1e-4f );
        for ( auto f : t.getValidFaces() )
        {
            EXPECT_TRUE( data.faceColors[f] == cStock || data.faceColors[f] == cTool );
            EXPECT_EQ( data.texturePerFace[f], TextureId( data.faceColors[f] == cTool ? 1 : 0 ) );
            const auto x = data.mesh->triCenter( f ).x;
            EXPECT_TRUE( data.selectedFaces.test( f ) ? x < 24 : x > 16 );
        }

        // the vertices from edge splits get interpolated uv, and the vertices of a patch the uv of the removed surface nearby
        const float uvTolerance = mode == FixMeshDegeneraciesParams::Mode::RemeshPatch ? 2.0f : 1e-3f;
        for ( auto v : t.getValidVerts() )
        {
            const auto pos = data.mesh->points[v];
            EXPECT_LT( ( data.uvCoordinates[v] - Vector2f( pos.x, pos.y ) ).length(), uvTolerance );
        }
    }
}

TEST( MRMesh, FixMeshDegeneraciesNotFlippable )
{
    // the mesh version with not flippable color borders gives the same mesh as the data version
    const auto c = makeDefectCase();
    for ( auto mode : cModes )
    {
        const auto params = fixParams( mode );
        auto data = c.data.clone();
        EXPECT_TRUE( fixMeshDataDegeneracies( data, params ).has_value() );

        Mesh mesh = *c.data.mesh;
        auto notFlippable = edgesBetweenDifferentColors( mesh.topology, c.data.faceColors );
        auto params1 = params;
        params1.notFlippable = &notFlippable;
        EXPECT_TRUE( fixMeshDegeneracies( mesh, params1 ).has_value() );
        EXPECT_TRUE( mesh == *data.mesh );

        EXPECT_TRUE( notFlippable.any() );
        for ( auto ue : notFlippable )
            EXPECT_FALSE( mesh.topology.isLoneEdge( ue ) );
    }
}

TEST( MRMesh, FixMeshDataDegeneraciesThenDecimate )
{
    // the fixes followed by decimation with packing keep the colors of the colored cut
    auto c = makeDefectCase();
    EXPECT_TRUE( fixMeshDataDegeneracies( c.data, fixParams( FixMeshDegeneraciesParams::Mode::Decimate ) ).has_value() );
    auto res = decimateObjectMeshData( c.data, { .maxError = 0.03f, .tinyEdgeLength = 1e-6f, .stabilizer = 1e-5f, .packMesh = true } );
    EXPECT_GT( res.facesDeleted, 0 );
    EXPECT_EQ( c.data.faceColors.size(), c.data.mesh->topology.faceSize() );
    EXPECT_LT( wrongColorArea( c.data, c.original ), 1e-4f );
}

} //namespace MR
