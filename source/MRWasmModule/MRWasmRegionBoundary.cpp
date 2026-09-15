#include "MRWasmBindings.h"

#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_DECLARE_VAL_TYPE( RightBoundaryVal )

EMSCRIPTEN_BINDINGS( meshlib_region_boundary )
{
    emscripten::register_type<RightBoundaryVal>( "Uint32Array[]" );

    emscripten::function( "getIncidentVertsFromFaces", +[]( const MeshTopology& topology, const FaceBitSet& faces )
    {
        return getIncidentVerts( topology, faces );
    } );
    emscripten::function( "getIncidentVertsFromEdges", +[]( const MeshTopology& topology, const UndirectedEdgeBitSet& edges )
    {
        return getIncidentVerts( topology, edges );
    } );

    emscripten::function( "getIncidentFacesFromVerts", +[]( const MeshTopology& topology, const VertBitSet& verts )
    {
        return getIncidentFaces( topology, verts );
    } );
    emscripten::function( "getIncidentFacesFromEdges", +[]( const MeshTopology& topology, const UndirectedEdgeBitSet& edges )
    {
        return getIncidentFaces( topology, edges );
    } );

    emscripten::function( "getIncidentEdgesFromFaces", +[]( const MeshTopology& topology, const FaceBitSet& faces )
    {
        return getIncidentEdges( topology, faces );
    } );
    emscripten::function( "getIncidentEdgesFromEdges", +[]( const MeshTopology& topology, const UndirectedEdgeBitSet& edges )
    {
        return getIncidentEdges( topology, edges );
    } );

    emscripten::function( "getInnerVertsFromFaces", +[]( const MeshTopology& topology, const FaceBitSet& region )
    {
        return getInnerVerts( topology, &region );
    } );
    emscripten::function( "getInnerVertsFromEdges", +[]( const MeshTopology& topology, const UndirectedEdgeBitSet& edges )
    {
        return getInnerVerts( topology, edges );
    } );

    emscripten::function( "getInnerFaces", +[]( const MeshTopology& topology, const VertBitSet& verts )
    {
        return getInnerFaces( topology, verts );
    } );

    emscripten::function( "getInnerEdgesFromVerts", +[]( const MeshTopology& topology, const VertBitSet& verts )
    {
        return getInnerEdges( topology, verts );
    } );
    emscripten::function( "getInnerEdgesFromFaces", +[]( const MeshTopology& topology, const FaceBitSet& region )
    {
        return getInnerEdges( topology, region );
    } );

    emscripten::function( "getBoundaryVerts", +[]( const MeshTopology& topology, const FaceBitSet& region )
    {
        return getBoundaryVerts( topology, &region );
    } );

    emscripten::function( "trackRightBoundaryLoop", +[]( const MeshTopology& topology, int e0 )
    {
        return Wasm::packedToTypedArray<EdgeLoop, uint32_t>( trackRightBoundaryLoop( topology, EdgeId( e0 ) ) );
    } );
    emscripten::function( "findRightBoundary", +[]( const MeshTopology& topology )
    {
        auto loops = findRightBoundary( topology );
        auto out = emscripten::val::array();
        for ( const auto& loop : loops )
            out.call<void>( "push", Wasm::packedToTypedArray<EdgeLoop, uint32_t>( loop ) );
        return RightBoundaryVal( out );
    } );
}
