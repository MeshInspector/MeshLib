#include "MREdgeMetric.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRParallelFor.h"
#include "MRBuffer.h"
#include "MRTimer.h"

namespace MR
{

UndirectedEdgeToFloatFunc identityMetric() 
{ 
    return []( UndirectedEdgeId ) { return 1.0f; };
}

UndirectedEdgeToFloatFunc edgeLengthMetric( const Mesh & mesh )
{
    return [&mesh]( UndirectedEdgeId ue )
    {
        return mesh.edgeLength( ue );
    };
}

UndirectedEdgeToFloatFunc discreteAbsMeanCurvatureMetric( const Mesh & mesh )
{
    return [&mesh]( UndirectedEdgeId ue )
    {
        return std::abs( mesh.discreteMeanCurvature( ue ) );
    };
}

UndirectedEdgeToFloatFunc discreteMinusAbsMeanCurvatureMetric( const Mesh & mesh )
{
    return [&mesh]( UndirectedEdgeId ue )
    {
        return -std::abs( mesh.discreteMeanCurvature( ue ) );
    };
}

UndirectedEdgeToFloatFunc edgeCurvMetric( const Mesh & mesh, float angleSinFactor, float angleSinForBoundary )
{
    const float bdFactor = exp( angleSinFactor * angleSinForBoundary );

    return [&mesh, angleSinFactor, bdFactor ]( UndirectedEdgeId ue ) -> float
    {
        auto edgeLen = mesh.edgeLength( ue );
        if ( mesh.topology.isBdEdge( ue, nullptr ) )
            return edgeLen * bdFactor;

        return edgeLen * exp( angleSinFactor * mesh.dihedralAngleSin( ue ) );
    };
}

UndirectedEdgeToFloatFunc edgeTableMetric( const MeshTopology & topology, const UndirectedEdgeToFloatFunc & metric )
{
    MR_TIMER

    Buffer<float, UndirectedEdgeId> table( topology.undirectedEdgeSize() );
    ParallelFor( table.beginId(), table.endId(), [&]( UndirectedEdgeId ue )
    {
        if ( topology.isLoneEdge( ue ) )
            return;
        table[ue] = metric( ue );
    } );

    // shared_ptr is necessary to satisfy CopyConstructible Callable target requirement of std::function
    return [tablePtr = std::make_shared<Buffer<float, UndirectedEdgeId>>( std::move( table ) )]( UndirectedEdgeId ue )
    {
        return (*tablePtr)[ue];
    };
}

} //namespace MR
