#pragma once

#include "MRMeshFwd.h"
#include "MRVector.h"

namespace MR
{

// oldData - the data that needs to be compared
// newToOld - how to compare indexes
// for greater optimality, it is worth making a resize for newData in advance
template<typename ValueT, typename IndexT>
void mapNewToOldVector(
    const Vector<ValueT, Id<IndexT>>& oldData,
    const Vector<Id<IndexT>, Id<IndexT>>& newToOld,
    Vector<ValueT, Id<IndexT>>& newData )
{
    for ( Id<IndexT> newId( 0 ); newId < newToOld.size(); ++newId )
        if ( auto oldId = newToOld[newId] )
            newData.autoResizeSet( newId, oldData[oldId] );
}

// oldData - the data that needs to be compared
// newToOld - how to compare indexes
// for greater optimality, it is worth making a resize for newData in advance
template<typename ValueT, typename IndexT>
void mapOldToNewVector(
    const Vector<ValueT, Id<IndexT>>& oldData,
    const Vector<Id<IndexT>, Id<IndexT>>& newToOld,
    Vector<ValueT, Id<IndexT>>& newData )
{
    for ( Id<IndexT> oldId( 0 ); oldId < newToOld.size(); ++oldId )
        if ( auto newId = newToOld[oldId] )
            newData.autoResizeSet( newId, oldData[oldId] );
}

}
