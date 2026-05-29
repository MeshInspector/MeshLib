#include <gtest/gtest.h>

#if defined( MR_MIMALLOC_ENABLED )

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>

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

// Confirms mimalloc services real allocations, not just that the lib is linked.
// Wired by Mimalloc.cmake (CMake) / MimallocRedirect.props (MSBuild) via the
// MR_MIMALLOC_ENABLED define. Each allocation below must land in a mimalloc region:
// C malloc, C++ operator new, and an allocation made INSIDE MRMesh (cross-library -
// the real goal, what the process-wide override must reach, esp. macOS two-level ns).
// On Windows also asserts the redirect engaged; MIMALLOC_DISABLE_REDIRECT=1 skips that.
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

    // C malloc, in this executable
    void* p = std::malloc( 64 );
    ASSERT_NE( p, nullptr );
    EXPECT_TRUE( mi_is_in_heap_region( p ) );
    std::free( p );

    // C++ operator new, in this executable
    int* q = new int[16];
    EXPECT_TRUE( mi_is_in_heap_region( q ) );
    delete[] q;

    // cross-library: makeCube() fills points via operator new compiled into MRMesh,
    // so its buffer must be a mimalloc region too (proves the override reaches the libs)
    const Mesh cube = makeCube();
    ASSERT_NE( cube.points.data(), nullptr );
    EXPECT_TRUE( mi_is_in_heap_region( cube.points.data() ) );
}

} // namespace MR

#endif // MR_MIMALLOC_ENABLED
