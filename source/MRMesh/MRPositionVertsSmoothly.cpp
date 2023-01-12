#include "MRPositionVertsSmoothly.h"
#include "MRTimer.h"

namespace MR
{

void positionVertsSmoothly( Mesh& mesh, const VertBitSet& verts,
    Laplacian::EdgeWeights egdeWeightsType,
    const VertBitSet * fixedSharpVertices )
{
    MR_TIMER;

    Laplacian laplacian( mesh );
    laplacian.init( verts, egdeWeightsType, Laplacian::RememberShape::No );
    if ( fixedSharpVertices )
        for ( auto v : *fixedSharpVertices )
            laplacian.fixVertex( v, false );
    laplacian.apply();
}

} //namespace MR
