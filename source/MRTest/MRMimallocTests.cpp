#include <MRMesh/config.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string_view>

#ifdef __linux__
#include <link.h>
#endif

#ifdef __EMSCRIPTEN__
// part of -sMALLOC=mimalloc without a header; mimalloc.h declares it returning `bool`
extern "C" bool mi_is_in_heap_region( const void* p );
#else
#include <mimalloc.h>
#endif

namespace MR
{

#ifdef MR_MIMALLOC_OVERRIDE
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
}
#else
// BitSet allocates with mimalloc explicitly, see BitSet::Blocks
TEST( MRMesh, MimallocBitSet )
{
    BitSet local( 1000, true );
    EXPECT_TRUE( mi_is_in_heap_region( local.bits().data() ) );

    // allocated inside MRMesh
    const auto mesh = makeCube();
    EXPECT_TRUE( mi_is_in_heap_region( mesh.topology.getValidFaces().bits().data() ) );
}

// MeshLib's own executables keep the system allocator, except in Emscripten builds
TEST( MRMesh, MimallocMallocOverride )
{
#ifdef __EMSCRIPTEN__
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
#endif

#ifdef __linux__
// The process must have only one mimalloc, of the version in mimalloc.h
TEST( MRMesh, MimallocSharedLibrary )
{
    int numMimallocs = 0;
    dl_iterate_phdr( []( dl_phdr_info* info, size_t, void* data )
    {
        // libmimalloc or libMRMimalloc
        if ( std::string_view( info->dlpi_name ).find( "imalloc" ) != std::string_view::npos )
            ++*static_cast<int*>( data );
        return 0;
    }, &numMimallocs );
    EXPECT_EQ( numMimallocs, 1 );
    EXPECT_EQ( mi_version(), MI_MALLOC_VERSION );
}
#endif

} // namespace MR
