#include "MRWasmBindings.h"

#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MRMeshTopology.h"
#include "MRMesh/MRBitSet.h"

#include <emscripten/bind.h>

using namespace MR;

EMSCRIPTEN_BINDINGS( meshlib_expand_shrink )
{
    emscripten::function( "expandFaces", +[]( const MeshTopology& topology, const FaceBitSet& region )
    {
        return expandFaces( topology, region );
    } );

    emscripten::function( "shrinkFaces", +[]( const MeshTopology& topology, const FaceBitSet& region )
    {
        return shrinkFaces( topology, region );
    } );

    emscripten::function( "getBoundaryFaces", +[]( const MeshTopology& topology, const FaceBitSet& region )
    {
        return getBoundaryFaces( topology, region );
    } );

    emscripten::function( "expandVerts", +[]( const MeshTopology& topology, const VertBitSet& region, int hops )
    {
        VertBitSet result = region;
        expand( topology, result, hops );
        return result;
    } );
    emscripten::function( "shrinkVerts", +[]( const MeshTopology& topology, const VertBitSet& region, int hops )
    {
        VertBitSet result = region;
        shrink( topology, result, hops );
        return result;
    } );
}
