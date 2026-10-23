// The C allocation entry points of the whole process, forwarding to MRMimalloc, see mr_enable_mimalloc.
// The Linux list of mimalloc's src/alloc-override.c; no libc headers, whose declarations would conflict.
#include <mimalloc.h>

#pragma GCC visibility push(default)

void* malloc( size_t size ) { return mi_malloc( size ); }
void* calloc( size_t count, size_t size ) { return mi_calloc( count, size ); }
void* realloc( void* p, size_t newsize ) { return mi_realloc( p, newsize ); }
void free( void* p ) { mi_free( p ); }
char* strdup( const char* s ) { return mi_strdup( s ); }
char* strndup( const char* s, size_t n ) { return mi_strndup( s, n ); }

// mi_malloc_size and mi_malloc_usable_size are defined only in mimalloc's override builds
size_t malloc_usable_size( void* p ) { return mi_is_in_heap_region( p ) ? mi_usable_size( p ) : 0; }
size_t malloc_size( const void* p ) { return mi_is_in_heap_region( p ) ? mi_usable_size( p ) : 0; }
void* reallocf( void* p, size_t newsize ) { return mi_reallocf( p, newsize ); }
void* valloc( size_t size ) { return mi_valloc( size ); }
void vfree( void* p ) { mi_free( p ); }
size_t malloc_good_size( size_t size ) { return mi_malloc_good_size( size ); }
int posix_memalign( void** p, size_t alignment, size_t size ) { return mi_posix_memalign( p, alignment, size ); }
void* aligned_alloc( size_t alignment, size_t size ) { return mi_aligned_alloc( alignment, size ); }

void cfree( void* p ) { mi_free( p ); }
void* pvalloc( size_t size ) { return mi_pvalloc( size ); }
void* memalign( size_t alignment, size_t size ) { return mi_memalign( alignment, size ); }
void* _aligned_malloc( size_t size, size_t alignment ) { return mi_malloc_aligned( size, alignment ); }
void* reallocarray( void* p, size_t count, size_t size ) { return mi_reallocarray( p, count, size ); }
__attribute__(( weak )) int reallocarr( void* p, size_t count, size_t size ) { return mi_reallocarr( p, count, size ); }

void* __libc_malloc( size_t size ) { return mi_malloc( size ); }
void* __libc_calloc( size_t count, size_t size ) { return mi_calloc( count, size ); }
void* __libc_realloc( void* p, size_t size ) { return mi_realloc( p, size ); }
void __libc_free( void* p ) { mi_free( p ); }
void __libc_cfree( void* p ) { mi_free( p ); }
void* __libc_valloc( size_t size ) { return mi_valloc( size ); }
void* __libc_pvalloc( size_t size ) { return mi_pvalloc( size ); }
void* __libc_memalign( size_t alignment, size_t size ) { return mi_memalign( alignment, size ); }
int __posix_memalign( void** p, size_t alignment, size_t size ) { return mi_posix_memalign( p, alignment, size ); }

#pragma GCC visibility pop
