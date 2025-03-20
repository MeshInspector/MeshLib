#include "MRMeshDoubleLayer.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MREdgeMetric.h"
#include "MRFillContourByGraphCut.h"
#include "MRMesh.h"
#include "MRTimer.h"

namespace MR
{

FaceBitSet findOuterLayer( const Mesh & mesh )
{
    MR_TIMER
    const auto szFaces = mesh.topology.faceSize();
    FaceBitSet innerSeeds( szFaces ), outerSeeds( szFaces );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId f )
    {
        const auto n = mesh.normal( f );
        const auto c = mesh.triCenter( f );
        const bool outer = !rayMeshIntersect( mesh, Line3f{ c, n }, 0.0f, FLT_MAX, nullptr, false, [f]( FaceId t ) { return t != f; } );
        const bool inner = !rayMeshIntersect( mesh, Line3f{ c,-n }, 0.0f, FLT_MAX, nullptr, false, [f]( FaceId t ) { return t != f; } );
        if ( outer == inner )
            return;
        if ( outer )
            outerSeeds.set( f );
        else
            innerSeeds.set( f );
    } );
    return segmentByGraphCut( mesh.topology, outerSeeds, innerSeeds, edgeLengthMetric( mesh ) );
}

} //namespace MR
