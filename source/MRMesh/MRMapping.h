#include "MRMeshFwd.h"
#include "MRVector.h"

namespace MR
{

// theseData - the data that needs to be compared
// theseToThose - how to compare indexes
// validIndex - which indexes should be compared
template<typename ValueT, typename IndexT, typename IndexF>
Vector<ValueT, IndexT> mapTheseToThoseVector(
    const Vector<ValueT, IndexT>& theseData,
    const Vector<IndexT, IndexT>& theseToThose,
    const TaggedBitSet<IndexF>& validIndex )
{
    Vector<ValueT, IndexT> newColorMap( validIndex.find_last() + 1 );
    for ( auto id : validIndex )
    {
        if ( auto curId = theseToThose[id] )
        {
            newColorMap[id] = theseData[curId];
        }
    }
    return newColorMap;
}

}
