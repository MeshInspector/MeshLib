#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string_view>

#ifdef __linux__
#include "MRMesh/MRCube.h"
#include "MRMesh/MRMesh.h"
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

    // allocated inside libMRMesh
    const auto mesh = makeCube();
    EXPECT_TRUE( mi_is_in_heap_region( mesh.points.data() ) );
}
#endif

} // namespace MR
