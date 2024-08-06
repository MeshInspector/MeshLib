#pragma once

#include <filesystem>

namespace std
{

namespace filesystem
{

/// to produce ambiguous call error during compilation
/// if one uses Exception-Throwing exists(path) instead of recommended not-throwing exists(path, error_code)
void exists( const path &, int = 0 ) = delete;

} //namespace filesystem

} //namespace std
