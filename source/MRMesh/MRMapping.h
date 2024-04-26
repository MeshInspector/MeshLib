#include "MRMeshFwd.h"
#include "MRVector.h"

namespace MR
{

// oldData - the data that needs to be compared
// newToOld - how to compare indexes
// newIndex - indexes of the new object to be mapped
template<typename ValueT, typename IndexT, typename IndexF>
Vector<ValueT, IndexT> mapNewToOldVector(
    const Vector<ValueT, IndexT>& oldData,
    const Vector<IndexT, IndexT>& newToOld,
    const TaggedBitSet<IndexF>& newIndex )
{
    Vector<ValueT, IndexT> newMap( validIndex.find_last() + 1 );
    for ( auto curId : newIndex )
    {
        if ( auto id = newToOld[curId] )
        {
            newMap[curId] = oldData[id];
        }
    }
    return newMap;
}

// oldData - the data that needs to be compared
// oldToNew - how to compare indexes
// newIndex - indexes of the new object to be mapped
template<typename ValueT, typename IndexT, typename IndexF>
Vector<ValueT, IndexT> mapOldToNewVector(
    const Vector<ValueT, IndexT>& oldData,
    const Vector<IndexT, IndexT>& oldToNew,
    const TaggedBitSet<IndexF>& newIndex )
{
    Vector<ValueT, IndexT> newMap( validIndex.find_last() + 1 );
    for ( auto curId : newIndex )
    {
        if ( auto id = newToOld[curId] )
        {
            newMap[id] = oldData[curId];
        }
    }
    return newMap;
}

}
