#include "MRWasmBindings.h"
#include "MRWasmMeshTriPoint.h"

#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRMeshTriPoint.h"
#include "MRMesh/MRId.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <memory>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_mesh )
{
    emscripten::class_<Mesh>( "Mesh" )
        .smart_ptr<std::shared_ptr<Mesh>>( "MeshPtr" )
        .property( "points", +[]( const Mesh& m ) { return m.points; } )
        .property( "topology", +[]( const Mesh& m ) { return m.topology; } )
        .class_function( "fromTriangles", +[]( const VertCoords& vertexCoordinates, const Triangulation& t )
        {
            return std::make_shared<Mesh>( Mesh::fromTriangles( vertexCoordinates, t ) );
        } )
        .function( "pack", +[]( Mesh& m ) { m.pack(); } )
        .function( "volume", +[]( const Mesh& m ) { return m.volume(); } )
        .function( "area", +[]( const Mesh& m ) { return m.area(); } )
        .function( "computeBoundingBox", +[]( const Mesh& m ) { return m.computeBoundingBox(); } )
        .function( "invalidateCaches", +[]( Mesh& m ) { m.invalidateCaches(); } )
        .function( "transform", +[]( Mesh& m, const AffineXf3f& xf ) { m.transform( xf ); } )
        .function( "averageEdgeLength", +[]( const Mesh& m ) { return m.averageEdgeLength(); } )
        .function( "findCenterFromPoints", +[]( const Mesh& m ) { return m.findCenterFromPoints(); } )
        .function( "findCenterFromFaces", +[]( const Mesh& m ) { return m.findCenterFromFaces(); } )
        .function( "findCenterFromBBox", +[]( const Mesh& m ) { return m.findCenterFromBBox(); } )
        .function( "addMesh", +[]( Mesh& m, std::shared_ptr<Mesh> from ) { m.addMesh( *from ); } )
        .function( "edgeLength", +[]( const Mesh& m, int e ) { return m.edgeLength( UndirectedEdgeId( e ) ); } )
        .function( "edgeLengthSq", +[]( const Mesh& m, int e ) { return m.edgeLengthSq( UndirectedEdgeId( e ) ); } )
        .function( "toTriPoint", +[]( const Mesh& m, int f, const Vector3f& p )
        {
            const MeshTriPoint mtp = m.toTriPoint( FaceId( f ), p );
            emscripten::val bary = emscripten::val::object();
            bary.set( "a", mtp.bary.a );
            bary.set( "b", mtp.bary.b );
            emscripten::val out = emscripten::val::object();
            out.set( "e", (int)mtp.e );
            out.set( "bary", bary );
            return Wasm::MeshTriPointVal( out );
        } )
        .class_function( "fromTrianglesDuplicatingNonManifoldVertices", +[]( const VertCoords& vertexCoordinates, Triangulation& t )
        {
            return std::make_shared<Mesh>( Mesh::fromTrianglesDuplicatingNonManifoldVertices( vertexCoordinates, t ) );
        } );
}
