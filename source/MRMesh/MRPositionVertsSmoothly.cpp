#include "MRPositionVertsSmoothly.h"
#include "MRTimer.h"

namespace MR
{

void positionVertsSmoothly( Mesh& mesh, const VertBitSet& verts, Laplacian::EdgeWeights egdeWeightsType )
{
    MR_TIMER;

    Laplacian laplacian( mesh );
    laplacian.init( verts, egdeWeightsType, Laplacian::RememberShape::No );
    laplacian.apply();
}

} //namespace MR
