#include "MRDirectory.h"

namespace MR
{

std::filesystem::path findPathWithExtension( const std::filesystem::path & pathWithoutExtension )
{
    std::error_code ec;
    const auto fn = pathWithoutExtension.filename();
    for ( auto dirEntry : Directory{ pathWithoutExtension.parent_path(), ec } )
    {
        if ( !dirEntry.is_regular_file( ec ) )
            continue;
        auto path = dirEntry.path();
        if ( path.stem() == fn )
            return path;
    }
    return {}; //not found
}

} // namespace MR
