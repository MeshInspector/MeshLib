#include "MREdgePathsBuilder.h"
#include "MREdgeMetric.h"

namespace MR
{

EdgePathsAStarBuilder::EdgePathsAStarBuilder( const Mesh & mesh, VertId target, VertId start ) :
    EdgePathsBuilderT( mesh.topology, edgeLengthMetric( mesh ) )
{
    metricToPenalty_.points = &mesh.points;
    metricToPenalty_.target = mesh.points[target];
    addStart( start, 0 );
}

EdgePathsAStarBuilder::EdgePathsAStarBuilder( const Mesh & mesh, const MeshTriPoint & target, const MeshTriPoint & start ) :
    EdgePathsBuilderT( mesh.topology, edgeLengthMetric( mesh ) )
{
    metricToPenalty_.points = &mesh.points;
    metricToPenalty_.target = mesh.triPoint( target );
    const auto startPt = mesh.triPoint( start );
    mesh.topology.forEachVertex( start, [&]( VertId v )
    {
        addStart( v, ( mesh.points[v] - startPt ).length() );
    } );
}

} //namespace MR
