#pragma once

#include "MRMeshFwd.h"
#include <filesystem>

namespace MR
{

/// object of this struct can be passed to range based for to enumerate all entries in a file directory
/// _without_ throwing exceptions, see https://stackoverflow.com/q/67776830/7325599
/// e.g.
/// std::error_code ec;
/// for ( auto dirEntry : Directory{ std::filesystem::current_path(), ec } )
///   { ... }
struct Directory
{
    std::filesystem::path dir;
    std::error_code & ec;
};

/// iterator of directory items that will save any errors in (ec) instead of throwing exceptions
struct DirectoryIterator
{
    std::filesystem::directory_iterator it;
    std::error_code & ec;
    DirectoryIterator & operator ++() { it.increment( ec ); return * this; }
    auto operator *() const { return *it; }
};

[[nodiscard]] inline DirectoryIterator begin( const Directory & sd )
{
    return DirectoryIterator{ std::filesystem::directory_iterator( sd.dir, sd.ec ), sd.ec };
}
 
[[nodiscard]] inline std::filesystem::directory_iterator end( const Directory & )
{
    return {};
}

[[nodiscard]] inline bool operator !=( const DirectoryIterator & a, const std::filesystem::directory_iterator & b )
{
    return !a.ec && a.it != b;
}

} // namespace MR
