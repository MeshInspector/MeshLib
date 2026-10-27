#pragma once

#include "config.h"
#include "MRPch/MRBindingMacros.h"
#include <memory>
#include <new>

// mr_enable_mimalloc executables forward malloc to the same mimalloc; MRMESH_NO_MIMALLOC, see cmake/Modules/Mimalloc.cmake
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_MIMALLOC )
#define MR_ALLOCATOR_IS_MIMALLOC
#endif

#ifdef MR_ALLOCATOR_IS_MIMALLOC
#include <mimalloc.h>
#if defined( _MSC_VER ) && !defined( MR_USE_CMAKE_CONFIGURE_FILE )
// the inline Allocator calls mimalloc from every binary using MRMesh; MeshLib's CMake targets link it instead
#ifdef _DEBUG
#pragma comment( lib, "mimalloc-debug.dll.lib" )
#else
#pragma comment( lib, "mimalloc.dll.lib" )
#endif
#endif
#endif

namespace MR
{

#ifdef MR_ALLOCATOR_IS_MIMALLOC
// not mi_stl_allocator: its construct/destroy members stop std containers from memmove-copying and skipping trivial destructors
template <typename T>
struct AllocatorBase {};
#else
template <typename T>
using AllocatorBase = std::allocator<T>;
#endif

/// allocator of MeshLib containers that benefit from mimalloc: mimalloc explicitly,
/// std::allocator in Emscripten builds, where malloc is mimalloc already (-sMALLOC=mimalloc), and with MRMESH_NO_MIMALLOC;
/// a class rather than an alias, so that container types are spelled the same on all platforms (e.g. in the C bindings generated once);
/// the bindings only see it as a template argument of containers
template <typename T>
struct MR_BIND_IGNORE Allocator : AllocatorBase<T>
{
    using value_type = T;
    template <typename U>
    struct rebind { using other = Allocator<U>; };

    Allocator() noexcept = default;
    template <typename U>
    Allocator( const Allocator<U> & ) noexcept {}

#ifdef MR_ALLOCATOR_IS_MIMALLOC
    using propagate_on_container_move_assignment = std::true_type;

    [[nodiscard]] T * allocate( std::size_t n )
    {
        if constexpr ( alignof( T ) > __STDCPP_DEFAULT_NEW_ALIGNMENT__ )
        {
            if ( n > std::size_t( -1 ) / sizeof( T ) )
                throw std::bad_array_new_length();
            return static_cast<T *>( mi_new_aligned( n * sizeof( T ), alignof( T ) ) );
        }
        else
            return static_cast<T *>( mi_new_n( n, sizeof( T ) ) );
    }
    void deallocate( T * p, std::size_t ) noexcept { mi_free( p ); }
#endif
    [[nodiscard]] Allocator select_on_container_copy_construction() const noexcept { return *this; }
};

template <typename T, typename U>
[[nodiscard]] inline bool operator ==( const Allocator<T> &, const Allocator<U> & ) noexcept { return true; }

} // namespace MR
