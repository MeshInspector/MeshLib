#include <gtest/gtest.h>

#if defined( MR_MIMALLOC_ENABLED )

#include <cstdlib>
#include <string_view>

// Declared here to avoid <mimalloc.h>. Return type MUST be bool, not int: MSVC
// returns bool in AL, so an int read picks up garbage in the upper bits (broke Release CI).
extern "C" bool mi_is_in_heap_region( const void* p );
#ifdef _WIN32
extern "C" bool mi_is_redirected();
#endif

namespace MR
{

// Asserts a plain malloc() lands in a mimalloc heap region (the allocator engaged).
// On Windows also asserts the redirect is active; MIMALLOC_DISABLE_REDIRECT=1 skips it.
// Gated by MR_MIMALLOC_ENABLED (set by Mimalloc.cmake / MimallocRedirect.props).
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
}

} // namespace MR

#endif // MR_MIMALLOC_ENABLED
