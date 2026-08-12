#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>
#include <cstdint>

#if defined _MSC_VER && !defined __clang__
#include <intrin.h>
#endif

namespace MR
{

/// \addtogroup HighPrecisionGroup
/// \{

/// a 64-bit integer, which product with another one is an exact 128-bit integer
/// obtained by a single widening machine multiplication;
/// addition and subtraction stay 64-bit and can overflow just like std::int64_t
class MR_BIND_IGNORE Int64Mul128
{
public:
    Int64Mul128() noexcept = default;
    constexpr Int64Mul128( std::int64_t v ) noexcept : v_( v ) { }
    [[nodiscard]] constexpr explicit operator std::int64_t() const noexcept { return v_; }

    [[nodiscard]] friend constexpr std::int64_t operator +( Int64Mul128 a ) noexcept { return a.v_; }
    [[nodiscard]] friend constexpr std::int64_t operator -( Int64Mul128 a ) noexcept { return -a.v_; }

    [[nodiscard]] friend constexpr std::int64_t operator +( Int64Mul128 a, Int64Mul128 b ) noexcept { return a.v_ + b.v_; }
    [[nodiscard]] friend constexpr std::int64_t operator -( Int64Mul128 a, Int64Mul128 b ) noexcept { return a.v_ - b.v_; }
    [[nodiscard]] friend constexpr std::int64_t operator /( Int64Mul128 a, Int64Mul128 b ) noexcept { return a.v_ / b.v_; }

    [[nodiscard]] friend FastInt128 operator *( Int64Mul128 a, Int64Mul128 b ) noexcept
    {
#if defined _MSC_VER && !defined __clang__ && defined _M_X64
        std::int64_t hi = 0;
        const auto lo = _mul128( a.v_, b.v_, &hi );
        return FastInt128( std::uint64_t( lo ), std::uint64_t( hi ) );
#elif defined _MSC_VER && !defined __clang__ && defined _M_ARM64
        const auto hi = __mulh( a.v_, b.v_ );
        return FastInt128( std::uint64_t( a.v_ ) * std::uint64_t( b.v_ ), std::uint64_t( hi ) );
#else
        // both arguments are extensions of 64-bit values, which the compiler recognizes
        return FastInt128( a.v_ ) * b.v_;
#endif
    }

    [[nodiscard]] friend constexpr bool operator ==( Int64Mul128 a, Int64Mul128 b ) noexcept { return a.v_ == b.v_; }
    [[nodiscard]] friend constexpr auto operator <=>( Int64Mul128 a, Int64Mul128 b ) noexcept { return a.v_ <=> b.v_; }

private:
    std::int64_t v_;
};

// no bindings for the same reason as for Vector3i128fast
#if !defined MR_PARSING_FOR_ANY_BINDINGS && !defined MR_COMPILING_ANY_BINDINGS
using Vector2i64mul = Vector2<Int64Mul128>;
using Vector3i64mul = Vector3<Int64Mul128>;
#endif

/// \}

} // namespace MR
