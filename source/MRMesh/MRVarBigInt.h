#pragma once

#include "MRMeshFwd.h"
#include "MRBuffer.h"
#include "MRFastInt.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>
#include <algorithm>
#include <array>
#include <compare>
#include <cstdint>
#include <variant>

namespace MR
{

/// \addtogroup HighPrecisionGroup
/// \{

/// a minimal variable-length signed big integer in sign-magnitude form over base-2^64 limbs,
/// a lightweight replacement for boost::multiprecision::cpp_int in the exact predicates;
/// it implements only what the simulation-of-simplicity inSphere tie resolution needs:
/// construction from machine integers and from fixed-precision FastInt<N> / FastInt128,
/// addition, subtraction, multiplication, sign, and comparison.
/// There is deliberately no division and no square root: signOf() in MRInSphere.cpp only
/// multiplies, compares and reads the sign. Unlike FastInt<N> the width grows with the value,
/// so no per-sub-expression bit-width bounds are needed.
///
/// The magnitude limbs live in a std::variant of two same-size-excluding-heap options:
/// a heap-backed Buffer<uint64_t> for really long integers, and an inline std::array<uint64_t,2>
/// giving small-value optimization (values up to 128 bits, e.g. every value in the tiny
/// near-degenerate regime, are stored without touching the heap).
class MR_BIND_IGNORE VarBigInt
{
public:
    VarBigInt() noexcept = default;

    /// Buffer is move-only, so the copy of the heap-backed alternative is done by hand
    VarBigInt( const VarBigInt & o ) { assignLimbs( o.sign_, o.limbs(), o.len_ ); }
    VarBigInt( VarBigInt && ) noexcept = default;
    VarBigInt & operator =( const VarBigInt & o ) { if ( this != &o ) assignLimbs( o.sign_, o.limbs(), o.len_ ); return *this; }
    VarBigInt & operator =( VarBigInt && ) noexcept = default;

    /// from a signed machine integer; non-explicit so that mixed expressions like 2 * x or x + n
    /// keep reading naturally, exactly as they did with boost cpp_int
    VarBigInt( std::int64_t v )
    {
        if ( v == 0 )
            return;
        // magnitude of v as unsigned two's-complement negation, correct also for INT64_MIN
        const std::uint64_t m = v < 0 ? ~std::uint64_t( v ) + 1 : std::uint64_t( v );
        assignLimbs( v < 0 ? -1 : 1, &m, 1 );
    }

    /// from a 128-bit integer (explicit to keep int -> VarBigInt unambiguous via the int64 ctor)
    explicit VarBigInt( FastInt128 v )
    {
        if ( v == 0 )
            return;
        const FastUInt128 u = v < 0 ? FastUInt128( 0 ) - FastUInt128( v ) : FastUInt128( v );
        const std::uint64_t w[2] = { std::uint64_t( u ), std::uint64_t( u >> 64 ) };
        assignLimbs( v < 0 ? -1 : 1, w, magNorm( w, 2 ) );
    }

    /// from a fixed-precision integer of the FastInt family; mirrors toBoostInt(): takes the
    /// magnitude of the two's-complement words so 2^nBits is never needed
    template <int nBits>
    explicit VarBigInt( const FastInt<nBits> & v )
    {
        const int s = v.sign();
        if ( s == 0 )
            return;
        auto w = v.w; // two's-complement words, little-endian
        if ( s < 0 )
        {
            std::uint64_t borrow = 0;
            for ( auto & x : w )
                x = detail::subBorrow64( 0, x, borrow );
        }
        assignLimbs( s, w.data(), magNorm( w.data(), w.size() ) );
    }

    /// -1, 0 or 1 if the value is negative, zero or positive respectively
    [[nodiscard]] int sign() const noexcept { return sign_; }

    [[nodiscard]] VarBigInt operator -() const
    {
        VarBigInt res = *this;
        res.sign_ = -res.sign_;
        return res;
    }

    VarBigInt & operator +=( const VarBigInt & b ) { return *this = *this + b; }
    VarBigInt & operator -=( const VarBigInt & b ) { return *this = *this - b; }

    [[nodiscard]] friend VarBigInt operator +( const VarBigInt & a, const VarBigInt & b )
    {
        if ( a.sign_ == 0 )
            return b;
        if ( b.sign_ == 0 )
            return a;
        if ( a.sign_ == b.sign_ )
            return make( a.sign_, std::max( a.len_, b.len_ ) + 1, false,
                [&]( std::uint64_t * out ) { return magAdd( a.limbs(), a.len_, b.limbs(), b.len_, out ); } );
        const int c = magCmp( a.limbs(), a.len_, b.limbs(), b.len_ );
        if ( c == 0 )
            return VarBigInt{}; // opposite signs, equal magnitudes => zero
        if ( c > 0 )
            return make( a.sign_, a.len_, false,
                [&]( std::uint64_t * out ) { return magSub( a.limbs(), a.len_, b.limbs(), b.len_, out ); } );
        return make( b.sign_, b.len_, false,
            [&]( std::uint64_t * out ) { return magSub( b.limbs(), b.len_, a.limbs(), a.len_, out ); } );
    }

    [[nodiscard]] friend VarBigInt operator -( const VarBigInt & a, const VarBigInt & b ) { return a + ( -b ); }

    [[nodiscard]] friend VarBigInt operator *( const VarBigInt & a, const VarBigInt & b )
    {
        if ( a.sign_ == 0 || b.sign_ == 0 )
            return VarBigInt{};
        return make( a.sign_ * b.sign_, a.len_ + b.len_, true,
            [&]( std::uint64_t * out ) { return magMul( a.limbs(), a.len_, b.limbs(), b.len_, out ); } );
    }

    [[nodiscard]] friend bool operator ==( const VarBigInt & a, const VarBigInt & b ) noexcept
    {
        if ( a.sign_ != b.sign_ || a.len_ != b.len_ )
            return false;
        const std::uint64_t * pa = a.limbs();
        const std::uint64_t * pb = b.limbs();
        for ( std::size_t i = 0; i < a.len_; ++i )
            if ( pa[i] != pb[i] )
                return false;
        return true;
    }

    [[nodiscard]] friend std::strong_ordering operator <=>( const VarBigInt & a, const VarBigInt & b ) noexcept
    {
        if ( a.sign_ != b.sign_ )
            return a.sign_ <=> b.sign_;
        if ( a.sign_ == 0 )
            return std::strong_ordering::equal;
        const int c = magCmp( a.limbs(), a.len_, b.limbs(), b.len_ ); // by magnitude
        const int r = a.sign_ > 0 ? c : -c;                          // flip for negatives
        return r <=> 0;
    }

private:
    int sign_ = 0;         ///< -1, 0 or 1; zero iff len_ == 0
    std::size_t len_ = 0;  ///< number of significant little-endian magnitude limbs (top limb non-zero)
    /// little-endian magnitude: heap Buffer for > 2 limbs, inline array (small-value optimization) otherwise
    std::variant<Buffer<std::uint64_t>, std::array<std::uint64_t, 2>> mag_{ std::array<std::uint64_t, 2>{} };

    /// pointer to the first magnitude limb, whichever alternative currently holds them
    [[nodiscard]] const std::uint64_t * limbs() const noexcept
    {
        return std::visit( []( const auto & s ) -> const std::uint64_t * { return s.data(); }, mag_ );
    }

    /// stores the normalized magnitude p[0, n) with the given sign, picking the inline array (n <= 2)
    /// or the heap Buffer (n > 2); n == 0 yields zero
    void assignLimbs( int sign, const std::uint64_t * p, std::size_t n )
    {
        if ( n == 0 )
        {
            sign_ = 0;
            len_ = 0;
            mag_.emplace<1>(); // inline zero
            return;
        }
        sign_ = sign;
        len_ = n;
        if ( n <= 2 )
        {
            std::array<std::uint64_t, 2> a{ 0, 0 };
            for ( std::size_t i = 0; i < n; ++i )
                a[i] = p[i];
            mag_.emplace<1>( a );
        }
        else
        {
            Buffer<std::uint64_t> b( n );
            for ( std::size_t i = 0; i < n; ++i )
                b[i] = p[i];
            mag_.emplace<0>( std::move( b ) );
        }
    }

    /// runs a limb kernel that writes the result magnitude into an output of up to ub limbs and
    /// returns its normalized length, keeping small results off the heap (inline array when ub <= 2)
    template <typename Kernel>
    [[nodiscard]] static VarBigInt make( int sign, std::size_t ub, bool zeroInit, Kernel kernel )
    {
        VarBigInt res;
        if ( sign == 0 )
            return res;
        if ( ub <= 2 )
        {
            std::array<std::uint64_t, 2> tmp{ 0, 0 };
            const std::size_t n = kernel( tmp.data() );
            res.assignLimbs( sign, tmp.data(), n );
        }
        else
        {
            Buffer<std::uint64_t> buf( ub );
            if ( zeroInit )
                std::fill( buf.data(), buf.data() + ub, std::uint64_t( 0 ) );
            const std::size_t n = kernel( buf.data() );
            if ( n <= 2 )
                res.assignLimbs( sign, buf.data(), n ); // rare heavy cancellation: fall back to inline
            else
            {
                res.sign_ = sign;
                res.len_ = n;
                res.mag_.emplace<0>( std::move( buf ) );
            }
        }
        return res;
    }

    /// drops trailing zero limbs, returning the significant length of p[0, n)
    [[nodiscard]] static std::size_t magNorm( const std::uint64_t * p, std::size_t n ) noexcept
    {
        while ( n > 0 && p[n - 1] == 0 )
            --n;
        return n;
    }

    /// compares two normalized magnitudes: -1 if a < b, 0 if equal, 1 if a > b
    [[nodiscard]] static int magCmp( const std::uint64_t * a, std::size_t na, const std::uint64_t * b, std::size_t nb ) noexcept
    {
        if ( na != nb )
            return na < nb ? -1 : 1;
        for ( std::size_t i = na; i-- > 0; )
            if ( a[i] != b[i] )
                return a[i] < b[i] ? -1 : 1;
        return 0;
    }

    /// out needs room for max( na, nb ) + 1 limbs; returns the normalized length
    static std::size_t magAdd( const std::uint64_t * a, std::size_t na, const std::uint64_t * b, std::size_t nb, std::uint64_t * out )
    {
        const std::size_t n = std::max( na, nb );
        std::uint64_t carry = 0;
        for ( std::size_t i = 0; i < n; ++i )
        {
            const std::uint64_t x = i < na ? a[i] : 0;
            const std::uint64_t y = i < nb ? b[i] : 0;
            out[i] = detail::addCarry64( x, y, carry );
        }
        std::size_t m = n;
        if ( carry )
            out[m++] = carry;
        return m; // the top limb is non-zero, so the result is already normalized
    }

    /// a - b assuming magCmp( a, b ) >= 0; out needs room for na limbs; returns the normalized length
    static std::size_t magSub( const std::uint64_t * a, std::size_t na, const std::uint64_t * b, std::size_t nb, std::uint64_t * out )
    {
        std::uint64_t borrow = 0;
        for ( std::size_t i = 0; i < na; ++i )
        {
            const std::uint64_t x = a[i];
            const std::uint64_t y = i < nb ? b[i] : 0;
            out[i] = detail::subBorrow64( x, y, borrow );
        }
        return magNorm( out, na ); // borrow is 0 here by the precondition
    }

    /// out needs room for na + nb limbs and must be zero-initialized; returns the normalized length
    static std::size_t magMul( const std::uint64_t * a, std::size_t na, const std::uint64_t * b, std::size_t nb, std::uint64_t * out )
    {
        for ( std::size_t i = 0; i < na; ++i ) // schoolbook multiplication of magnitudes
        {
            std::uint64_t carry = 0;
            for ( std::size_t j = 0; j < nb; ++j )
            {
                // at most ( 2^64 - 1 )^2 + 2 * ( 2^64 - 1 ) < 2^128, so it fits in 128 bits
                const FastUInt128 t = FastUInt128( a[i] ) * FastUInt128( b[j] )
                                    + FastUInt128( out[i + j] ) + FastUInt128( carry );
                out[i + j] = std::uint64_t( t );
                carry = std::uint64_t( t >> 64 );
            }
            out[i + nb] += carry; // out[i + nb] was 0 before, so it cannot overflow
        }
        return magNorm( out, na + nb );
    }
};

/// \}

} // namespace MR
