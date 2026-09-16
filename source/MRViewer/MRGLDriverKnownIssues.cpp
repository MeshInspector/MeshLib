#include "MRGLDriverKnownIssues.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"
#include "MRPch/MRFmt.h"
#include <compare>
#include <cstdio>

namespace
{

struct Version
{
    int major = 0;
    int minor = 0;
    int patch = 0;

    static Version fromString( const char* str )
    {
        Version result;
        std::sscanf( str, "%d.%d.%d", &result.major, &result.minor, &result.patch );
        return result;
    }

    std::string toString() const
    {
        return fmt::format( "{}.{}.{}", major, minor, patch );
    }

    auto operator <=>( const Version& ) const = default;
};

} // namespace

namespace MR
{

std::optional<GLDriverIssue> glDriverKnownIssues()
{
    if ( !getViewerInstance().isGLInitialized() )
        return {};

    const std::string_view renderer = ( const char* )glGetString( GL_RENDERER );
    const std::string_view version = ( const char* )glGetString( GL_VERSION );

    // https://gitlab.freedesktop.org/mesa/mesa/-/issues/15660 , fixed in Mesa 26.1.6
    if ( renderer.starts_with( "llvmpipe" ) )
    {
        constexpr std::string_view mesaPrefix = "Mesa ";
        const auto mesaPos = version.find( mesaPrefix );
        if ( mesaPos != std::string_view::npos )
        {
            const auto mesaVersion = Version::fromString( version.data() + mesaPos + mesaPrefix.size() );
            if ( Version{ 25, 3, 0 } <= mesaVersion && mesaVersion <= Version{ 26, 1, 5 } )
                return GLDriverIssue{
                    .id = "mesa-15660",
                    .description = fmt::format(
                        "Mesa {} llvmpipe causes rendering issues and might lead to the app crash. "
                        "Upgrade Mesa to 26.1.6 or newer, or enable the hardware rendering.",
                        mesaVersion.toString()
                    ),
                };
        }
    }

    return {};
}

} // namespace MR
