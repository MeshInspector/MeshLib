#pragma once

#if __has_include(<__msvc_int128.hpp>)
  #include <__msvc_int128.hpp>
namespace MR
{
  // these types are much faster than boost::multiprecision::checked_int128_t
  // but lack sqrt-function and stream input/output; for the conversion in double,
  // which std::_Signed128 lacks as well, see MR::toDouble in MRFastInt.h
  using FastInt128 = std::_Signed128;
  using FastUInt128 = std::_Unsigned128;
}
#else
namespace MR
{
  using FastInt128 = __int128_t;
  using FastUInt128 = __uint128_t;
}
#endif
