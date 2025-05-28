#pragma once

#include "MRMeshFwd.h"
#include "MRMacros.h"
#include <concepts>

namespace MR
{

/// resizes the vector skipping initialization of its elements (more precisely initializing them using ( noInit ) constructor );
/// this is much faster than ordinary resize(), because the memory of vector's elements is not accessed and only one pointer inside vector is repeatedly increased;
/// https://stackoverflow.com/a/74212402/7325599
template <typename T>
void resizeNoInit( std::vector<T> & vec, size_t targetSize ) MR_REQUIRES_IF_SUPPORTED( sizeof( T ) > 0 && std::constructible_from<T, NoInit> )
{
    // allocate enough memory
    vec.reserve( targetSize );
    // resize without memory access
    while ( vec.size() < targetSize )
        vec.emplace_back( noInit );
    // in case initial size was larger
    vec.resize( targetSize );
}

} //namespace MR
