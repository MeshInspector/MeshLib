#include "MRAllocator.h"

#ifdef MR_ALLOCATOR_IS_MIMALLOC
#include <mimalloc.h>
#if defined( _MSC_VER ) && !defined( MR_USE_CMAKE_CONFIGURE_FILE )
#ifdef _DEBUG
#pragma comment( lib, "mimalloc-debug.dll.lib" )
#else
#pragma comment( lib, "mimalloc.dll.lib" )
#endif
#endif

namespace MR::detail
{

void * mimallocNewN( std::size_t count, std::size_t size )
{
    return mi_new_n( count, size );
}

void * mimallocNewAligned( std::size_t size, std::size_t alignment )
{
    return mi_new_aligned( size, alignment );
}

void mimallocFree( void * p ) noexcept
{
    mi_free( p );
}

} // namespace MR::detail
#endif
