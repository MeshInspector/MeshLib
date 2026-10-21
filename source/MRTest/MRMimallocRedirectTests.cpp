#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string_view>

// mimalloc.h declares these as `bool`. Matching the actual return type is
// critical: MSVC returns `bool` in the low byte of EAX, so reading the call
// site as `int` picks up garbage in the upper 24 bits (e.g. -980287487).
// Local Debug happened to zero those bits; Release CI didn't.
extern "C" bool mi_is_redirected();
extern "C" bool mi_is_in_heap_region( const void* p );

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

} // namespace MR
