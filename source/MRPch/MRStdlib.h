#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctype.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <istream>
#include <iterator>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>
#include <version>

namespace std
{

namespace filesystem
{

/// to produce ambiguous call error during compilation
/// if one uses Exception-Throwing exists(path) instead of recommended not-throwing exists(path, error_code)
void exists( const path &, int = 0 ) = delete;
void is_regular_file( const path &, int = 0 ) = delete;
void is_directory( const path &, int = 0 ) = delete;

} //namespace filesystem

} //namespace std
