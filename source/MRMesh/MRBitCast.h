#pragma once

#include <version>

#if __cpp_lib_bit_cast >= 201806L
#include <bit>
#endif

namespace MR
{

/// std::bit_cast wrapper with the compiler's intrinsic fallback
template <typename To, typename From>
[[nodiscard]] constexpr To bit_cast( const From& from ) noexcept
{
#if __cpp_lib_bit_cast >= 201806L
    return std::bit_cast<To>( from );
#else
    return __builtin_bit_cast( To, from );
#endif
}

} //namespace MR
