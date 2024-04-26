#include "MRMeshFwd.h"
#include "MRVector.h"

namespace MR
{

// oldData - the data that needs to be compared
// newToOld - how to compare indexes
// newIndex - indexes of the new object to be mapped
template<typename ValueT, typename IndexT>
Vector<ValueT, Id<IndexT>> mapNewToOldVector(
    const Vector<ValueT, Id<IndexT>>& oldData,
    const Vector<Id<IndexT>, Id<IndexT>>& newToOld,
    const TaggedBitSet<IndexT>& newIndex )
{
    Vector<ValueT, Id<IndexT>> newMap( newIndex.find_last() + 1 );
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
template<typename ValueT, typename IndexT>
Vector<ValueT, Id<IndexT>> mapOldToNewVector(
    const Vector<ValueT, Id<IndexT>>& oldData,
    const Vector<Id<IndexT>, Id<IndexT>>& oldToNew,
    const TaggedBitSet<IndexT>& newIndex )
{
    Vector<ValueT, Id<IndexT>> newMap( newIndex.find_last() + 1 );
    for ( auto curId : newIndex )
    {
        if ( auto id = oldToNew[curId] )
        {
            newMap[id] = oldData[curId];
        }
    }
    return newMap;
}

}
