#include "MRMeshFwd.h"
#include "MRVector.h"

namespace MR
{

template<typename ValueT, typename IndexT, typename IndexF>
Vector<ValueT, IndexT> mapNewToOldVector(
    const Vector<ValueT, IndexT>& oldColorMap,
    const Vector<IndexT, IndexT>& mapNew2Old,
    const TaggedBitSet<IndexF>& newFacesId )
{
    Vector<ValueT, IndexT> newColorMap( newFacesId.find_last() + 1 );
    for ( auto id : newFacesId )
    {
        auto curId = mapNew2Old[id];
        if ( curId.valid() )
        {
            newColorMap[id] = oldColorMap[curId];
        }
    }
    return newColorMap;
}

template<typename ValueT, typename IndexT>
Vector<ValueT, IndexT> mapOldToNewVector(
    const Vector<ValueT, IndexT>& oldColorMap,
    const Vector<IndexT, IndexT>& mapOld2New,
    const TaggedBitSet<IndexT>& oldFacesId )
{
    Vector<ValueT, IndexT> newColorMap( oldFacesId.find_last() + 1 );
    for ( auto id : oldFacesId )
    {
        auto curId = mapOld2New[id];
        if ( curId.valid() )
        {
            newColorMap[id] = oldColorMap[curId];
        }
    }
    return newColorMap;
}

}
