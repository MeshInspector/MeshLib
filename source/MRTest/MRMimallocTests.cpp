#include <MRMesh/config.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string_view>

#ifdef __linux__
#include <link.h>
#endif

#ifdef __EMSCRIPTEN__
// part of -sMALLOC=mimalloc without a header; mimalloc.h declares it returning `bool`
extern "C" bool mi_is_in_heap_region( const void* p );
#endif

namespace MR
{

// BitSet allocates with mimalloc explicitly, see BitSet::Blocks
TEST( MRMesh, MimallocBitSet )
{
    BitSet local( 1000, true );
    EXPECT_TRUE( mi_is_in_heap_region( local.bits().data() ) );

    // allocated inside MRMesh
    const auto mesh = makeCube();
    EXPECT_TRUE( mi_is_in_heap_region( mesh.topology.getValidFaces().bits().data() ) );
}

// malloc is mimalloc only in Emscripten builds and in mr_enable_mimalloc executables (MR_MIMALLOC_OVERRIDE);
// MRTest is not one of them, but on macOS and Linux linking the override build is enough
TEST( MRMesh, MimallocMallocOverride )
{
#if defined( __EMSCRIPTEN__ ) || ( defined( MR_MIMALLOC_OVERRIDE ) && !defined( _WIN32 ) )
    constexpr bool expected = true;
#else
    constexpr bool expected = false;
#endif
#ifdef _WIN32
    EXPECT_FALSE( mi_is_redirected() );
#endif

    void* p = std::malloc( 64 );
    ASSERT_NE( p, nullptr );
    EXPECT_EQ( mi_is_in_heap_region( p ), expected );
    std::free( p );
}

#ifdef __linux__
// The executable and MeshLib libraries must share one libmimalloc.so of the version in mimalloc.h
TEST( MRMesh, MimallocSharedLibrary )
{
    int numMimallocs = 0;
    dl_iterate_phdr( []( dl_phdr_info* info, size_t, void* data )
    {
        if ( std::string_view( info->dlpi_name ).find( "libmimalloc" ) != std::string_view::npos )
            ++*static_cast<int*>( data );
        return 0;
    }, &numMimallocs );
    EXPECT_EQ( numMimallocs, 1 );
    EXPECT_EQ( mi_version(), MI_MALLOC_VERSION );
}
#endif

} // namespace MR
