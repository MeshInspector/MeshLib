#pragma once
#include "MRMeshFwd.h"
#include "MRUnionFind.h"

namespace MR
{

namespace PolylineComponents
{

/// returns one connected component containing given undirected edge id, 
/// not effective to call more than once, if several components are needed use getAllComponents
/// \ingroup ComponentsGroup
MRMESH_API UndirectedEdgeBitSet getComponent( const PolylineTopology& topology, UndirectedEdgeId id );

/// gets all connected components of polyline topology
/// \ingroup ComponentsGroup
MRMESH_API std::vector<UndirectedEdgeBitSet> getAllComponents( const PolylineTopology& topology );

/// gets union-find structure for given polyline
/// \ingroup ComponentsGroup
MRMESH_API UnionFind<UndirectedEdgeId> getUnionFindStructure( const PolylineTopology& topology );

}

}