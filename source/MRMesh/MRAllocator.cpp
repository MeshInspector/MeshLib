#include "MRAllocator.h"

#ifdef MR_ALLOCATOR_IS_MIMALLOC
#include <mimalloc.h>
#ifdef _MSC_VER
#pragma comment( linker, "/EXPORT:mrMimallocNewN" )
#pragma comment( linker, "/EXPORT:mrMimallocNewAligned" )
#pragma comment( linker, "/EXPORT:mrMimallocFree" )
#ifndef MR_USE_CMAKE_CONFIGURE_FILE
#ifdef _DEBUG
#pragma comment( lib, "mimalloc-debug.dll.lib" )
#else
#pragma comment( lib, "mimalloc.dll.lib" )
#endif
#endif
#endif

namespace MR
{

void * mrMimallocNewN( std::size_t count, std::size_t size )
{
    return mi_new_n( count, size );
}

void * mrMimallocNewAligned( std::size_t size, std::size_t alignment )
{
    return mi_new_aligned( size, alignment );
}

void mrMimallocFree( void * p ) noexcept
{
    mi_free( p );
}

} // namespace MR
#endif
