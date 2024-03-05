#pragma once
#include "MRMeshFwd.h"
#include "MRUnionFind.h"

namespace MR
{

namespace PolylineComponents
{

/// returns the number of connected components in polyline
[[nodiscard]] MRMESH_API size_t getNumComponents( const PolylineTopology& topology);

/// returns one connected component containing given undirected edge id, 
/// not effective to call more than once, if several components are needed use getAllComponents
/// \ingroup ComponentsGroup
MRMESH_API UndirectedEdgeBitSet getComponent( const PolylineTopology& topology, UndirectedEdgeId id );


/// gets all connected components of polyline topology
/// \ingroup ComponentsGroup
/// \note be careful, if mesh is large enough and has many components, the memory overflow will occur
MRMESH_API std::vector<UndirectedEdgeBitSet> getAllComponents( const PolylineTopology& topology );
/// gets all connected components of polyline topology
/// \ingroup ComponentsGroup
/// \detail if components  number more than the maxComponentCount, they will be combined into groups of the same size 
/// \param maxComponentCount should be more then 1
/// \return pair components bitsets vector and number components in one group if components number more than maxComponentCount
MRMESH_API std::pair<std::vector<UndirectedEdgeBitSet>, int> getAllComponents( const PolylineTopology& topology, int maxComponentCount );

/// gets union-find structure for given polyline
/// \ingroup ComponentsGroup
MRMESH_API UnionFind<UndirectedEdgeId> getUnionFindStructure( const PolylineTopology& topology );

/// returns largest by length component
/// \ingroup ComponentsGroup
template <typename V>
UndirectedEdgeBitSet getLargestComponent( const Polyline<V>& polyline );

}

}