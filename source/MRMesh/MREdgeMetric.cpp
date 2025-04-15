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

EdgeMetric edgeLengthMetric( const MeshTopology& topology, const VertCoords& points )
{
    return [&topology, &points]( EdgeId e )
    {
        return edgeLength( topology, points, e );
    };
}

EdgeMetric edgeLengthMetric( const Mesh & mesh )
{
    return edgeLengthMetric( mesh.topology, mesh.points );
}

EdgeMetric discreteAbsMeanCurvatureMetric( const MeshTopology& topology, const VertCoords& points )
{
    return [&topology, &points]( EdgeId e )
    {
        return std::abs( discreteMeanCurvature( topology, points, e.undirected() ) );
    };
}

EdgeMetric discreteAbsMeanCurvatureMetric( const Mesh & mesh )
{
    return discreteAbsMeanCurvatureMetric( mesh.topology, mesh.points );
}

EdgeMetric discreteMinusAbsMeanCurvatureMetric( const MeshTopology& topology, const VertCoords& points )
{
    return [&topology, &points]( EdgeId e )
    {
        return -std::abs( discreteMeanCurvature( topology, points, e.undirected() ) );
    };
}

EdgeMetric discreteMinusAbsMeanCurvatureMetric( const Mesh & mesh )
{
    return discreteMinusAbsMeanCurvatureMetric( mesh.topology, mesh.points );
}

EdgeMetric edgeCurvMetric( const MeshTopology& topology, const VertCoords& points, float angleSinFactor, float angleSinForBoundary )
{
    const float bdFactor = exp( angleSinFactor * angleSinForBoundary );

    return [&topology, &points, angleSinFactor, bdFactor ]( EdgeId e ) -> float
    {
        auto edgeLen = edgeLength( topology, points, e );
        if ( topology.isBdEdge( e, nullptr ) )
            return edgeLen * bdFactor;

        return edgeLen * exp( angleSinFactor * dihedralAngleSin( topology, points, e ) );
    };
}

EdgeMetric edgeCurvMetric( const Mesh & mesh, float angleSinFactor, float angleSinForBoundary )
{
    return edgeCurvMetric( mesh.topology, mesh.points, angleSinFactor, angleSinForBoundary );
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
