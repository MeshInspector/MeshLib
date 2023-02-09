#include "MREdgeMetric.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
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

EdgeMetric edgeTableMetric( const MeshTopology & topology, const EdgeMetric & metric )
{
    MR_TIMER

    Vector<float, UndirectedEdgeId> table( topology.undirectedEdgeSize() );
    for ( auto e : undirectedEdges( topology ) )
        table[e] = metric( e );

    return [table = std::move( table )]( EdgeId e )
    {
        return table[e.undirected()];
    };
}

} //namespace MR
