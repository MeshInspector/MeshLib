#pragma once

#if __has_include(<__msvc_int128.hpp>)
  #include <__msvc_int128.hpp>
  // this type is much faster than boost::multiprecision::checked_int128_t but lacks conversion in double and sqrt-function
namespace MR
{
  using FastInt128 = std::_Signed128;
}
#else
namespace MR
{
  using FastInt128 = __int128_t;
}
#endif
