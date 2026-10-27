#include <MRMesh/config.h>
#include <MRMesh/MRAllocator.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRBuffer.h>
#include <MRMesh/MRVector.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string_view>

#ifdef __linux__
#include <link.h>
#elif defined( __APPLE__ )
#include <mach-o/dyld.h>
#include <malloc/malloc.h>
#endif

#ifdef __EMSCRIPTEN__
// part of -sMALLOC=mimalloc without a header; mimalloc.h declares it returning `bool`
extern "C" bool mi_is_in_heap_region( const void* p );
#else
#include <mimalloc.h>
#endif

namespace MR
{

// Verifies mimalloc replaced the system allocator for this executable, see mr_enable_mimalloc
// in MeshLib/cmake/Modules/Mimalloc.cmake. Skipped on Windows when MIMALLOC_DISABLE_REDIRECT=1
// (mimalloc's own runtime kill-switch).
TEST( MRMesh, MimallocRedirectActive )
{
#ifdef _WIN32
    if ( const char* disable = std::getenv( "MIMALLOC_DISABLE_REDIRECT" );
         disable && std::string_view( disable ) == "1" )
    {
        GTEST_SKIP() << "MIMALLOC_DISABLE_REDIRECT=1; redirect intentionally disabled.";
    }

    EXPECT_TRUE( mi_is_redirected() );
#endif

    void* p = std::malloc( 64 );
    ASSERT_NE( p, nullptr );
    EXPECT_TRUE( mi_is_in_heap_region( p ) );
    std::free( p );

    auto q = std::make_unique<int[]>( 16 );
    EXPECT_TRUE( mi_is_in_heap_region( q.get() ) );

#ifdef __APPLE__
    // CoreFoundation and Objective-C allocate from the default malloc zone
    void* z = malloc_zone_malloc( malloc_default_zone(), 64 );
    EXPECT_TRUE( mi_is_in_heap_region( z ) );
    malloc_zone_free( malloc_default_zone(), z );
#endif
}

#ifdef MR_ALLOCATOR_IS_MIMALLOC
// BitSet allocates with mimalloc explicitly, see BitSet::Blocks
TEST( MRMesh, MimallocBitSet )
{
    BitSet local( 1000, true );
    EXPECT_TRUE( mi_is_in_heap_region( local.bits().data() ) );

    // allocated inside MRMesh
    const auto mesh = makeCube();
    EXPECT_TRUE( mi_is_in_heap_region( mesh.topology.getValidFaces().bits().data() ) );
}

TEST( MRMesh, MimallocAllocator )
{
    Buffer<int> buf( 1000 );
    EXPECT_TRUE( mi_is_in_heap_region( buf.data() ) );

    // mi_stl_allocator alone ignores alignas
    struct alignas( 64 ) Aligned { char c = 0; };
    MiVector<Aligned> v( 3 );
    EXPECT_TRUE( mi_is_in_heap_region( v.data() ) );
    EXPECT_EQ( reinterpret_cast<std::uintptr_t>( v.data() ) % 64, 0 );
}
#endif

#if defined( __linux__ ) || defined( __APPLE__ )
// The process must have only one mimalloc, of the version in mimalloc.h
TEST( MRMesh, MimallocSharedLibrary )
{
    struct Counts
    {
        int mimallocs = 0;
        int overrides = 0;
        void add( std::string_view path )
        {
            const auto name = path.substr( path.find_last_of( '/' ) + 1 );
            if ( name.starts_with( "libmimalloc" ) )
                ++mimallocs;
            if ( name.starts_with( "libMRMimallocOverride." ) )
                ++overrides;
        }
    } counts;
#ifdef __linux__
    dl_iterate_phdr( []( dl_phdr_info* info, size_t, void* data )
    {
        static_cast<Counts*>( data )->add( info->dlpi_name );
        return 0;
    }, &counts );
#else
    for ( uint32_t i = 0; i < _dyld_image_count(); ++i )
        counts.add( _dyld_get_image_name( i ) );
#endif
    EXPECT_EQ( counts.mimallocs, 1 );
#ifndef MRMESH_NO_MIMALLOC
    // forwards malloc and new to the libraries' mimalloc, see mr_enable_mimalloc
    EXPECT_EQ( counts.overrides, 1 );
#else
    EXPECT_EQ( counts.overrides, 0 );
#endif
    EXPECT_EQ( mi_version(), MI_MALLOC_VERSION );
}
#endif

} // namespace MR
