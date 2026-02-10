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

struct LargeByLengthComponentsSettings
{
    /// return at most given number of largest by length connected components
    int maxLargeComponents = 2;

    /// optional output: the number of components in addition to returned ones
    int* numSmallerComponents = nullptr;

    /// do not consider a component large if its length is below this value
    float minLength = 0;
};

/// returns requested number of largest by length connected components in descending by length order
/// \ingroup ComponentsGroup
template<typename V>
std::vector<UndirectedEdgeBitSet> getNLargeByLengthComponents( const Polyline<V>& polyline, const LargeByLengthComponentsSettings& settings );

MR_BIND_TEMPLATE( std::vector<UndirectedEdgeBitSet> getNLargeByLengthComponents( const Polyline2& polyline, const LargeByLengthComponentsSettings& settings ) )
MR_BIND_TEMPLATE( std::vector<UndirectedEdgeBitSet> getNLargeByLengthComponents( const Polyline3& polyline, const LargeByLengthComponentsSettings& settings ) )

/// gets union-find structure for given polyline
/// \ingroup ComponentsGroup
MRMESH_API UnionFind<UndirectedEdgeId> getUnionFindStructure( const PolylineTopology& topology );

/// returns largest by length component
/// \ingroup ComponentsGroup
template <typename V>
UndirectedEdgeBitSet getLargestComponent( const Polyline<V>& polyline, float minLength = 0, int* numSmallerComponents = nullptr );

MR_BIND_TEMPLATE( UndirectedEdgeBitSet getLargestComponent( const Polyline2& polyline, float minLength = 0, int* numSmallerComponents = nullptr ) )
MR_BIND_TEMPLATE( UndirectedEdgeBitSet getLargestComponent( const Polyline3& polyline, float minLength = 0, int* numSmallerComponents = nullptr ) )

}

}