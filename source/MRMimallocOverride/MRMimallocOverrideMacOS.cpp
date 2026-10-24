// macOS: interposes malloc, new and the malloc zone API of the whole process with MRMimalloc, see mr_enable_mimalloc.
// A port of mimalloc 3.5.3's src/alloc-override.c (MI_OSX_INTERPOSE, MI_OSX_ZONE) and src/prim/osx/alloc-override-zone.c
// to its public API; dyld applies these interposes only from images loaded at launch.
// Copyright (c) 2018-2026, Microsoft Research, Daan Leijen; MIT license.
#include <mimalloc.h>

#include <AvailabilityMacros.h>
#include <malloc/malloc.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>

extern "C"
{
void vfree( void* p );
int malloc_jumpstart( uintptr_t cookie );
void _malloc_fork_prepare( void );
void _malloc_fork_parent( void );
void _malloc_fork_child( void );

// the mangled operators, as mimalloc declares them
void _ZdlPv( void* p );                  // delete
void _ZdaPv( void* p );                  // delete[]
void _ZdlPvm( void* p, size_t n );       // delete
void _ZdaPvm( void* p, size_t n );       // delete[]
void* _Znwm( size_t n );                 // new
void* _Znam( size_t n );                 // new[]
void* _ZnwmRKSt9nothrow_t( size_t n, const void* tag ); // new nothrow
void* _ZnamRKSt9nothrow_t( size_t n, const void* tag ); // new[] nothrow
}

namespace
{

// dyld applies only the interposes whose replacements are in this image, so every one forwards to MRMimalloc here
void* mr_malloc( size_t size ) { return mi_malloc( size ); }
void* mr_calloc( size_t count, size_t size ) { return mi_calloc( count, size ); }
void* mr_realloc( void* p, size_t newsize ) { return mi_realloc( p, newsize ); }
char* mr_strdup( const char* s ) { return mi_strdup( s ); }
char* mr_strndup( const char* s, size_t n ) { return mi_strndup( s, n ); }
int mr_posix_memalign( void** p, size_t alignment, size_t size ) { return mi_posix_memalign( p, alignment, size ); }
void* mr_reallocf( void* p, size_t newsize ) { return mi_reallocf( p, newsize ); }
void* mr_valloc( size_t size ) { return mi_valloc( size ); }
void* mr_aligned_alloc( size_t alignment, size_t size ) { return mi_aligned_alloc( alignment, size ); }
// mi_malloc_size is defined only in mimalloc's override builds
size_t mr_malloc_size( const void* p ) { return mi_is_in_heap_region( p ) ? mi_usable_size( p ) : 0; }
size_t mr_malloc_good_size( size_t size ) { return mi_malloc_good_size( size ); }
void mr_free( void* p ) { mi_free( p ); }
void mr_free_size( void* p, size_t n ) { mi_free_size( p, n ); }
void* mr_new( size_t n ) { return mi_new( n ); }
void* mr_new_nothrow( size_t n, const void* ) { return mi_new_nothrow( n ); }

malloc_zone_t* mi_get_default_zone();
bool is_mimalloc_zone( malloc_zone_t* zone );

// malloc zone members

size_t zone_size( malloc_zone_t* zone, const void* p )
{
    if ( mi_any_heap_contains( p ) )
        return mi_usable_size( p );
    else if ( !is_mimalloc_zone( zone ) ) // can happen due to interpose
        return zone->size( zone, p );
    return 0;
}

void* zone_malloc( malloc_zone_t*, size_t size ) { return mi_malloc( size ); }
void* zone_calloc( malloc_zone_t*, size_t count, size_t size ) { return mi_calloc( count, size ); }
void* zone_valloc( malloc_zone_t*, size_t size ) { return mi_valloc( size ); }

void zone_free( malloc_zone_t* zone, void* p )
{
    // during C++ thread shutdown `_pthread_tsd_cleanup` may call `zone_free` after mimalloc's mi_thread_done
    if ( !mi_cfree( p ) && !is_mimalloc_zone( zone ) ) // can happen due to interpose
        zone->free( zone, p );
}

void* zone_realloc( malloc_zone_t* zone, void* p, size_t newsize )
{
    if ( p == nullptr || mi_any_heap_contains( p ) )
        return mi_realloc( p, newsize );
    else if ( !is_mimalloc_zone( zone ) ) // can happen due to interpose
        return zone->realloc( zone, p, newsize );
    return nullptr;
}

void* zone_memalign( malloc_zone_t*, size_t alignment, size_t size ) { return mi_malloc_aligned( size, alignment ); }

void zone_destroy( malloc_zone_t* zone )
{
    if ( !is_mimalloc_zone( zone ) )
        zone->destroy( zone );
}

unsigned zone_batch_malloc( malloc_zone_t* zone, size_t size, void** ps, unsigned count )
{
    unsigned i = 0;
    for ( ; i < count; i++ )
    {
        ps[i] = zone_malloc( zone, size );
        if ( ps[i] == nullptr )
            break;
    }
    return i;
}

void zone_batch_free( malloc_zone_t* zone, void** ps, unsigned count )
{
    for ( unsigned i = 0; i < count; i++ )
    {
        zone_free( zone, ps[i] );
        ps[i] = nullptr;
    }
}

size_t zone_pressure_relief( malloc_zone_t*, size_t )
{
    mi_collect( false );
    return 0;
}

void zone_free_definite_size( malloc_zone_t* zone, void* p, size_t ) { zone_free( zone, p ); }
boolean_t zone_claimed_address( malloc_zone_t*, void* p ) { return mi_is_in_heap_region( p ); }

// introspection members

kern_return_t intro_enumerator( task_t, void*, unsigned, vm_address_t, memory_reader_t, vm_range_recorder_t ) { return KERN_SUCCESS; }
size_t intro_good_size( malloc_zone_t*, size_t size ) { return mi_good_size( size ); }
boolean_t intro_check( malloc_zone_t* ) { return true; }
void intro_print( malloc_zone_t*, boolean_t ) { mi_stats_print( nullptr ); }
void intro_log( malloc_zone_t*, void* ) {}
void intro_force_lock( malloc_zone_t* ) {}
void intro_force_unlock( malloc_zone_t* ) {}
void intro_statistics( malloc_zone_t*, malloc_statistics_t* stats ) { *stats = {}; }
boolean_t intro_zone_locked( malloc_zone_t* ) { return false; }
// required for zone version >= 9: macOS calls it from _malloc_fork_child without a null check
void intro_reinit_lock( malloc_zone_t* ) {}

malloc_introspection_t mi_introspect = {
    .enumerator = &intro_enumerator,
    .good_size = &intro_good_size,
    .check = &intro_check,
    .print = &intro_print,
    .log = &intro_log,
    .force_lock = &intro_force_lock,
    .force_unlock = &intro_force_unlock,
    .statistics = &intro_statistics,
    .zone_locked = &intro_zone_locked,
    .reinit_lock = &intro_reinit_lock,
};

// the order of the designators is the declaration order
malloc_zone_t mi_malloc_zone = {
    .size = &zone_size,
    .malloc = &zone_malloc,
    .calloc = &zone_calloc,
    .valloc = &zone_valloc,
    .free = &zone_free,
    .realloc = &zone_realloc,
    .destroy = &zone_destroy,
    .zone_name = "mimalloc",
    .batch_malloc = &zone_batch_malloc,
    .batch_free = &zone_batch_free,
    .introspect = &mi_introspect,
    .version = 10,
    .memalign = &zone_memalign,
    .free_definite_size = &zone_free_definite_size,
    .pressure_relief = &zone_pressure_relief,
    .claimed_address = &zone_claimed_address,
};

bool is_mimalloc_zone( malloc_zone_t* zone ) { return zone == nullptr || zone == &mi_malloc_zone; }

std::atomic<bool> zoneRegistered{ false };

malloc_zone_t* mi_get_default_zone()
{
    // registering avoids a zone error on free
    if ( !zoneRegistered.exchange( true ) )
        malloc_zone_register( &mi_malloc_zone );
    return &mi_malloc_zone;
}

// the malloc_xxx and malloc_zone_xxx API use only the mimalloc zone, even in the loader

malloc_zone_t* mi_malloc_create_zone( vm_size_t, unsigned ) { return mi_get_default_zone(); }
malloc_zone_t* mi_malloc_default_zone() { return mi_get_default_zone(); }
malloc_zone_t* mi_malloc_default_purgeable_zone() { return mi_get_default_zone(); }
void mi_malloc_destroy_zone( malloc_zone_t* ) {}

kern_return_t mi_malloc_get_all_zones( task_t, memory_reader_t, vm_address_t** addresses, unsigned* count )
{
    if ( addresses )
        *addresses = nullptr;
    if ( count )
        *count = 0;
    return KERN_SUCCESS;
}

const char* mi_malloc_get_zone_name( malloc_zone_t* zone ) { return zone == nullptr ? mi_malloc_zone.zone_name : zone->zone_name; }
void mi_malloc_set_zone_name( malloc_zone_t*, const char* ) {}
int mi_malloc_jumpstart( uintptr_t ) { return 1; }
void mi__malloc_fork_prepare() {}
void mi__malloc_fork_parent() {}
void mi__malloc_fork_child() {}
void mi_malloc_printf( const char*, ... ) {}
boolean_t zone_check( malloc_zone_t* ) { return true; }
malloc_zone_t* zone_from_ptr( const void* p ) { return mi_any_heap_contains( p ) ? mi_get_default_zone() : nullptr; }
void zone_log( malloc_zone_t*, void* ) {}
void zone_print( malloc_zone_t*, bool ) {}
void zone_print_ptr_info( void* ) {}
void zone_register( malloc_zone_t* ) {}
void zone_unregister( malloc_zone_t* ) {}

struct Interpose
{
    const void* replacement;
    const void* target;
};

#define MR_INTERPOSE( oldfun, newfun ) { (const void*)&newfun, (const void*)&oldfun }
#define MR_INTERPOSE_MR( fun ) MR_INTERPOSE( fun, mr_##fun )
#define MR_INTERPOSE_MI( fun ) MR_INTERPOSE( fun, mi_##fun )
#define MR_INTERPOSE_ZONE( fun ) MR_INTERPOSE( malloc_##fun, fun )

__attribute__(( used, section( "__DATA,__interpose" ) )) const Interpose interposes[] = {
    MR_INTERPOSE_MR( malloc ),
    MR_INTERPOSE_MR( calloc ),
    MR_INTERPOSE_MR( realloc ),
    MR_INTERPOSE_MR( strdup ),
    MR_INTERPOSE_MR( strndup ),
    // not realpath: mi_realpath calls it, and MRMimalloc's calls are interposed too
    MR_INTERPOSE_MR( posix_memalign ),
    MR_INTERPOSE_MR( reallocf ),
    MR_INTERPOSE_MR( valloc ),
    MR_INTERPOSE_MR( aligned_alloc ),
    MR_INTERPOSE_MR( malloc_size ),
    MR_INTERPOSE_MR( malloc_good_size ),
    // malloc_default_zone is interposed below, so mi_free is safe
    MR_INTERPOSE_MR( free ),
    MR_INTERPOSE( vfree, mr_free ),

    MR_INTERPOSE( _ZdlPv, mr_free ),
    MR_INTERPOSE( _ZdaPv, mr_free ),
    MR_INTERPOSE( _ZdlPvm, mr_free_size ),
    MR_INTERPOSE( _ZdaPvm, mr_free_size ),
    MR_INTERPOSE( _Znwm, mr_new ),
    MR_INTERPOSE( _Znam, mr_new ),
    MR_INTERPOSE( _ZnwmRKSt9nothrow_t, mr_new_nothrow ),
    MR_INTERPOSE( _ZnamRKSt9nothrow_t, mr_new_nothrow ),

    MR_INTERPOSE_MI( malloc_create_zone ),
    MR_INTERPOSE_MI( malloc_default_purgeable_zone ),
    MR_INTERPOSE_MI( malloc_default_zone ),
    MR_INTERPOSE_MI( malloc_destroy_zone ),
    MR_INTERPOSE_MI( malloc_get_all_zones ),
    MR_INTERPOSE_MI( malloc_get_zone_name ),
    MR_INTERPOSE_MI( malloc_jumpstart ),
    MR_INTERPOSE_MI( malloc_printf ),
    MR_INTERPOSE_MI( malloc_set_zone_name ),
    MR_INTERPOSE_MI( _malloc_fork_child ),
    MR_INTERPOSE_MI( _malloc_fork_parent ),
    MR_INTERPOSE_MI( _malloc_fork_prepare ),

    MR_INTERPOSE_ZONE( zone_batch_free ),
    MR_INTERPOSE_ZONE( zone_batch_malloc ),
    MR_INTERPOSE_ZONE( zone_calloc ),
    MR_INTERPOSE_ZONE( zone_check ),
    MR_INTERPOSE_ZONE( zone_free ),
    MR_INTERPOSE_ZONE( zone_from_ptr ),
    MR_INTERPOSE_ZONE( zone_log ),
    MR_INTERPOSE_ZONE( zone_malloc ),
    MR_INTERPOSE_ZONE( zone_memalign ),
    MR_INTERPOSE_ZONE( zone_print ),
    MR_INTERPOSE_ZONE( zone_print_ptr_info ),
    MR_INTERPOSE_ZONE( zone_realloc ),
    MR_INTERPOSE_ZONE( zone_register ),
    MR_INTERPOSE_ZONE( zone_unregister ),
    MR_INTERPOSE_ZONE( zone_valloc ),
};

} // namespace
