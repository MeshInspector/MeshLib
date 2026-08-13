#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>
#include <algorithm>
#include <compare>
#include <cstdint>
#include <vector>

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
class MR_BIND_IGNORE VarBigInt
{
public:
    VarBigInt() noexcept = default;

    /// from a signed machine integer; non-explicit so that mixed expressions like 2 * x or x + n
    /// keep reading naturally, exactly as they did with boost cpp_int
    VarBigInt( std::int64_t v )
    {
        if ( v == 0 )
            return;
        sign_ = v < 0 ? -1 : 1;
        // magnitude of v as unsigned two's-complement negation, correct also for INT64_MIN
        mag_.push_back( v < 0 ? ~std::uint64_t( v ) + 1 : std::uint64_t( v ) );
    }

    /// from a 128-bit integer (explicit to keep int -> VarBigInt unambiguous via the int64 ctor)
    explicit VarBigInt( FastInt128 v )
    {
        if ( v == 0 )
            return;
        sign_ = v < 0 ? -1 : 1;
        const FastUInt128 u = v < 0 ? FastUInt128( 0 ) - FastUInt128( v ) : FastUInt128( v );
        mag_ = { std::uint64_t( u ), std::uint64_t( u >> 64 ) };
        normalize();
    }

    /// from a fixed-precision integer of the FastInt family; mirrors toBoostInt(): takes the
    /// magnitude of the two's-complement words so 2^nBits is never needed
    template <int nBits>
    explicit VarBigInt( const FastInt<nBits> & v )
    {
        const int s = v.sign();
        if ( s == 0 )
            return;
        sign_ = s;
        auto w = v.w; // two's-complement words, little-endian
        if ( s < 0 )
        {
            std::uint64_t borrow = 0;
            for ( auto & x : w )
                x = detail::subBorrow64( 0, x, borrow );
        }
        mag_.assign( w.begin(), w.end() );
        normalize();
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
        VarBigInt res;
        if ( a.sign_ == b.sign_ )
        {
            res.mag_ = magAdd( a.mag_, b.mag_ );
            res.sign_ = a.sign_;
        }
        else if ( const int c = magCmp( a.mag_, b.mag_ ); c > 0 )
        {
            res.mag_ = magSub( a.mag_, b.mag_ );
            res.sign_ = a.sign_;
        }
        else if ( c < 0 )
        {
            res.mag_ = magSub( b.mag_, a.mag_ );
            res.sign_ = b.sign_;
        }
        // c == 0: opposite signs, equal magnitudes => zero (res left default)
        return res;
    }

    [[nodiscard]] friend VarBigInt operator -( const VarBigInt & a, const VarBigInt & b ) { return a + ( -b ); }

    [[nodiscard]] friend VarBigInt operator *( const VarBigInt & a, const VarBigInt & b )
    {
        VarBigInt res;
        if ( a.sign_ == 0 || b.sign_ == 0 )
            return res;
        res.mag_ = magMul( a.mag_, b.mag_ );
        res.sign_ = a.sign_ * b.sign_;
        return res;
    }

    [[nodiscard]] friend bool operator ==( const VarBigInt & a, const VarBigInt & b ) noexcept
    {
        return a.sign_ == b.sign_ && a.mag_ == b.mag_;
    }

    [[nodiscard]] friend std::strong_ordering operator <=>( const VarBigInt & a, const VarBigInt & b ) noexcept
    {
        if ( a.sign_ != b.sign_ )
            return a.sign_ <=> b.sign_;
        if ( a.sign_ == 0 )
            return std::strong_ordering::equal;
        const int c = magCmp( a.mag_, b.mag_ ); // by magnitude
        const int r = a.sign_ > 0 ? c : -c;     // flip for negatives
        return r <=> 0;
    }

private:
    int sign_ = 0;                    ///< -1, 0 or 1; zero iff mag_ is empty
    std::vector<std::uint64_t> mag_;  ///< little-endian magnitude, no trailing zero limbs

    void normalize() noexcept
    {
        magNorm( mag_ );
        if ( mag_.empty() )
            sign_ = 0;
    }

    static void magNorm( std::vector<std::uint64_t> & a ) noexcept
    {
        while ( !a.empty() && a.back() == 0 )
            a.pop_back();
    }

    /// compares two normalized magnitudes: -1 if a < b, 0 if equal, 1 if a > b
    static int magCmp( const std::vector<std::uint64_t> & a, const std::vector<std::uint64_t> & b ) noexcept
    {
        if ( a.size() != b.size() )
            return a.size() < b.size() ? -1 : 1;
        for ( std::size_t i = a.size(); i-- > 0; )
            if ( a[i] != b[i] )
                return a[i] < b[i] ? -1 : 1;
        return 0;
    }

    static std::vector<std::uint64_t> magAdd( const std::vector<std::uint64_t> & a, const std::vector<std::uint64_t> & b )
    {
        const std::size_t n = std::max( a.size(), b.size() );
        std::vector<std::uint64_t> res;
        res.reserve( n + 1 );
        std::uint64_t carry = 0;
        for ( std::size_t i = 0; i < n; ++i )
        {
            const std::uint64_t x = i < a.size() ? a[i] : 0;
            const std::uint64_t y = i < b.size() ? b[i] : 0;
            res.push_back( detail::addCarry64( x, y, carry ) );
        }
        if ( carry )
            res.push_back( carry );
        return res;
    }

    /// a - b assuming magCmp( a, b ) >= 0
    static std::vector<std::uint64_t> magSub( const std::vector<std::uint64_t> & a, const std::vector<std::uint64_t> & b )
    {
        std::vector<std::uint64_t> res;
        res.reserve( a.size() );
        std::uint64_t borrow = 0;
        for ( std::size_t i = 0; i < a.size(); ++i )
        {
            const std::uint64_t x = a[i];
            const std::uint64_t y = i < b.size() ? b[i] : 0;
            res.push_back( detail::subBorrow64( x, y, borrow ) );
        }
        magNorm( res ); // borrow is 0 here by precondition
        return res;
    }

    static std::vector<std::uint64_t> magMul( const std::vector<std::uint64_t> & a, const std::vector<std::uint64_t> & b )
    {
        if ( a.empty() || b.empty() )
            return {};
        std::vector<std::uint64_t> res( a.size() + b.size(), std::uint64_t( 0 ) );
        for ( std::size_t i = 0; i < a.size(); ++i ) // schoolbook multiplication of magnitudes
        {
            std::uint64_t carry = 0;
            for ( std::size_t j = 0; j < b.size(); ++j )
            {
                // at most ( 2^64 - 1 )^2 + 2 * ( 2^64 - 1 ) < 2^128, so it fits in 128 bits
                const FastUInt128 t = FastUInt128( a[i] ) * FastUInt128( b[j] )
                                    + FastUInt128( res[i + j] ) + FastUInt128( carry );
                res[i + j] = std::uint64_t( t );
                carry = std::uint64_t( t >> 64 );
            }
            res[i + b.size()] += carry; // res[i + b.size()] was untouched before, so it is 0 here
        }
        magNorm( res );
        return res;
    }
};

/// \}

} // namespace MR
