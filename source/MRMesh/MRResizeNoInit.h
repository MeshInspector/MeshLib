#pragma once

#include "MRMeshFwd.h"
#include "MRMacros.h"
#include <algorithm>
#include <concepts>

namespace MR
{

/// resizes the vector skipping initialization of its elements (more precisely initializing them using ( noInit ) constructor );
/// this is much faster than ordinary resize(), because the memory of vector's elements is not accessed and only one pointer inside vector is repeatedly increased;
/// https://stackoverflow.com/a/74212402/7325599
/// \param exactReserve if false and the capacity is short, it is at least doubled as push_back does, so repeated small growths copy the elements only O(log) times
template <typename T, typename A>
void resizeNoInit( std::vector<T, A> & vec, size_t targetSize, bool exactReserve = true ) MR_REQUIRES_IF_SUPPORTED( sizeof( T ) > 0 && std::constructible_from<T, NoInit> )
{
    // allocate enough memory
    if ( targetSize > vec.capacity() )
        vec.reserve( exactReserve ? targetSize : std::max( targetSize, 2 * vec.capacity() ) );
    // resize without memory access
    while ( vec.size() < targetSize )
        vec.emplace_back( noInit );
    // in case initial size was larger
    vec.resize( targetSize );
}

} //namespace MR
