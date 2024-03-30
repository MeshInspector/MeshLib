#include "MREdgeMetric.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRParallelFor.h"
#include "MRBuffer.h"
#include "MRTimer.h"

namespace MR
{

EdgeMetric identityMetric() 
{ 
    return []( EdgeId ) { return 1.0f; };
}

EdgeMetric edgeLengthMetric( const Mesh & mesh )
{
    return [&mesh]( EdgeId e )
    {
        return mesh.edgeLength( e );
    };
}

EdgeMetric discreteAbsMeanCurvatureMetric( const Mesh & mesh )
{
    return [&mesh]( EdgeId e )
    {
        return std::abs( mesh.discreteMeanCurvature( e.undirected() ) );
    };
}

EdgeMetric discreteMinusAbsMeanCurvatureMetric( const Mesh & mesh )
{
    return [&mesh]( EdgeId e )
    {
        return -std::abs( mesh.discreteMeanCurvature( e.undirected() ) );
    };
}

EdgeMetric edgeCurvMetric( const Mesh & mesh, float angleSinFactor, float angleSinForBoundary )
{
    const float bdFactor = exp( angleSinFactor * angleSinForBoundary );

    return [&mesh, angleSinFactor, bdFactor ]( EdgeId e ) -> float
    {
        auto edgeLen = mesh.edgeLength( e );
        if ( mesh.topology.isBdEdge( e, nullptr ) )
            return edgeLen * bdFactor;

        return edgeLen * exp( angleSinFactor * mesh.dihedralAngleSin( e ) );
    };
}

EdgeMetric edgeTableSymMetric( const MeshTopology & topology, const EdgeMetric & metric )
{
    MR_TIMER

    Buffer<float, UndirectedEdgeId> table( topology.undirectedEdgeSize() );
    ParallelFor( table.beginId(), table.endId(), [&]( UndirectedEdgeId ue )
    {
        if ( topology.isLoneEdge( ue ) )
            return;
        table[ue] = metric( EdgeId( ue ) );
#ifndef NDEBUG
        assert( table[ue] == metric( EdgeId( ue ).sym() ) );
#endif
    } );

    // shared_ptr is necessary to satisfy CopyConstructible Callable target requirement of std::function
    return [tablePtr = std::make_shared<Buffer<float, UndirectedEdgeId>>( std::move( table ) )]( UndirectedEdgeId ue )
    {
        return (*tablePtr)[ue];
    };
}

} //namespace MR
