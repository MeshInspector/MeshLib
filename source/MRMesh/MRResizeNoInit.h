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
template <typename T, typename A>
void resizeNoInit( std::vector<T, A> & vec, size_t targetSize ) MR_REQUIRES_IF_SUPPORTED( sizeof( T ) > 0 && std::constructible_from<T, NoInit> )
{
    // allocate enough memory, at least doubling the size as push_back does: repeated small growths copy the elements only O(log) times
    if ( targetSize > vec.capacity() )
        vec.reserve( std::max( targetSize, 2 * vec.size() ) );
    // resize without memory access
    while ( vec.size() < targetSize )
        vec.emplace_back( noInit );
    // in case initial size was larger
    vec.resize( targetSize );
}

} //namespace MR
