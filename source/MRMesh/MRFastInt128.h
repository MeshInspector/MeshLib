#pragma once

#include <cstdint>

#if __has_include(<__msvc_int128.hpp>) && !defined(__SIZEOF_INT128__)
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
/// the compiler has __int128_t (GCC, Clang, and clang-cl, which also sees std::_Signed128 above, but makes
/// its arithmetic much slower than the built-in one), and with it a built-in conversion in double,
/// which std::_Signed128 above has no equivalent of; see MR::toDouble in MRFastInt.h
#define MR_HAS_BUILTIN_INT128 1
#ifdef _MSC_VER
  /// clang-cl links with the MSVC runtime, which lacks the compiler-rt functions called by the division of __int128_t
  /// (__divti3) and by its conversion in double (__floattidf); MR::divFastInt128 and MR::toDouble avoid them then
  #define MR_BUILTIN_INT128_NO_RUNTIME 1
  #include <__msvc_int128.hpp>
#endif
namespace MR
{
  using FastInt128 = __int128_t;
  using FastUInt128 = __uint128_t;
}
#endif

namespace MR
{

/// n / d rounded toward zero, as the built-in division of integers
[[nodiscard]] inline FastInt128 divFastInt128( FastInt128 n, FastInt128 d ) noexcept
{
#ifdef MR_BUILTIN_INT128_NO_RUNTIME
    // by the division of std::_Signed128, which calls nothing outside the header
    const FastUInt128 un( n ), ud( d );
    const std::_Signed128 q = std::_Signed128( std::uint64_t( un ), std::uint64_t( un >> 64 ) )
        / std::_Signed128( std::uint64_t( ud ), std::uint64_t( ud >> 64 ) );
    return FastInt128( ( FastUInt128( q._Word[1] ) << 64 ) | q._Word[0] );
#else
    return n / d;
#endif
}

} // namespace MR
