#include "MRMeshFwd.h"
#include "MRVector.h"

namespace MR
{

// theseData - the data that needs to be compared
// theseToThose - how to compare indexes
// validIndex - which indexes should be compared
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
