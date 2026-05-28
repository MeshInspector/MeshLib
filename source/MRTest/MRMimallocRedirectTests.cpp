#include <gtest/gtest.h>

#ifdef _WIN32

#include <cstdlib>
#include <string_view>

extern "C" int mi_is_redirected( void );
extern "C" int mi_is_in_heap_region( const void* p );

namespace MR
{

// Verifies mimalloc's transparent CRT-allocator redirect engaged for this EXE.
// Wired in MeshLib/source/MimallocRedirect.props (MSBuild) and
// MeshLib/cmake/Modules/MimallocRedirect.cmake (CMake). Skipped when
// MIMALLOC_DISABLE_REDIRECT=1 (mimalloc's own runtime kill-switch).
TEST( MRMesh, MimallocRedirectActive )
{
    if ( const char* disable = std::getenv( "MIMALLOC_DISABLE_REDIRECT" );
         disable && std::string_view( disable ) == "1" )
    {
        GTEST_SKIP() << "MIMALLOC_DISABLE_REDIRECT=1; redirect intentionally disabled.";
    }

    EXPECT_EQ( mi_is_redirected(), 1 );

    void* p = std::malloc( 64 );
    ASSERT_NE( p, nullptr );
    EXPECT_EQ( mi_is_in_heap_region( p ), 1 );
    std::free( p );
}

} // namespace MR

#endif // _WIN32
