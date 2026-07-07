#include "MRWasmBindings.h"

#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_region_boundary )
{
    emscripten::function( "getIncidentVertsFromFaces", +[]( const MeshTopology& t, const FaceBitSet& faces )
    {
        return getIncidentVerts( t, faces );
    } );
    emscripten::function( "getIncidentVertsFromEdges", +[]( const MeshTopology& t, const UndirectedEdgeBitSet& edges )
    {
        return getIncidentVerts( t, edges );
    } );

    emscripten::function( "getIncidentFacesFromVerts", +[]( const MeshTopology& t, const VertBitSet& verts )
    {
        return getIncidentFaces( t, verts );
    } );
    emscripten::function( "getIncidentFacesFromEdges", +[]( const MeshTopology& t, const UndirectedEdgeBitSet& edges )
    {
        return getIncidentFaces( t, edges );
    } );

    emscripten::function( "getIncidentEdgesFromFaces", +[]( const MeshTopology& t, const FaceBitSet& faces )
    {
        return getIncidentEdges( t, faces );
    } );
    emscripten::function( "getIncidentEdgesFromEdges", +[]( const MeshTopology& t, const UndirectedEdgeBitSet& edges )
    {
        return getIncidentEdges( t, edges );
    } );

    emscripten::function( "getInnerVertsFromFaces", +[]( const MeshTopology& t, const FaceBitSet& region )
    {
        return getInnerVerts( t, &region );
    } );
    emscripten::function( "getInnerVertsFromEdges", +[]( const MeshTopology& t, const UndirectedEdgeBitSet& edges )
    {
        return getInnerVerts( t, edges );
    } );

    emscripten::function( "getInnerFaces", +[]( const MeshTopology& t, const VertBitSet& verts )
    {
        return getInnerFaces( t, verts );
    } );

    emscripten::function( "getInnerEdgesFromVerts", +[]( const MeshTopology& t, const VertBitSet& verts )
    {
        return getInnerEdges( t, verts );
    } );
    emscripten::function( "getInnerEdgesFromFaces", +[]( const MeshTopology& t, const FaceBitSet& region )
    {
        return getInnerEdges( t, region );
    } );

    emscripten::function( "getBoundaryVerts", +[]( const MeshTopology& t, const FaceBitSet& region )
    {
        return getBoundaryVerts( t, &region );
    } );

    emscripten::function( "trackRightBoundaryLoop", +[]( const MeshTopology& t, int e0 )
    {
        return Wasm::packedToTypedArray<EdgeLoop, uint32_t>( trackRightBoundaryLoop( t, EdgeId( e0 ) ) );
    } );
    emscripten::function( "findRightBoundary", +[]( const MeshTopology& t )
    {
        auto loops = findRightBoundary( t );
        auto out = emscripten::val::array();
        for ( const auto& loop : loops )
            out.call<void>( "push", Wasm::packedToTypedArray<EdgeLoop, uint32_t>( loop ) );
        return out;
    } );
}
