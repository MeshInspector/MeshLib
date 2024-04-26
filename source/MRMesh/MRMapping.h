#include "MRMeshFwd.h"
#include "MRVector.h"

namespace MR
{

// oldData - the data that needs to be compared
// newToOld - how to compare indexes
// newIndex - indexes of the new object to be mapped
template<typename ValueT, typename IndexT>
void mapNewToOldVector(
    const Vector<ValueT, Id<IndexT>>& oldData,
    const Vector<Id<IndexT>, Id<IndexT>>& newToOld,
    const TaggedBitSet<IndexT>& newIndex,
    Vector<ValueT, Id<IndexT>>& newData )
{
    if ( newData.size() < newIndex.find_last() )
        newData.resizeNoInit( newIndex.find_last() + 1 );

    for ( auto curId : newIndex )
    {
        if ( auto id = newToOld[curId] )
        {
            newData[curId] = oldData[id];
        }
    }
}

// oldData - the data that needs to be compared
// oldToNew - how to compare indexes
// newIndex - indexes of the new object to be mapped
template<typename ValueT, typename IndexT>
void mapOldToNewVector(
    const Vector<ValueT, Id<IndexT>>& oldData,
    const Vector<Id<IndexT>, Id<IndexT>>& oldToNew,
    const TaggedBitSet<IndexT>& newIndex,
    Vector<ValueT, Id<IndexT>>& newData )
{
    for ( auto curId : newIndex )
    {
        if ( auto id = oldToNew[curId] )
        {
            newData[id] = oldData[curId];
        }
    }
}

}
