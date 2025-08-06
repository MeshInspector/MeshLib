#pragma once

#include <map>

namespace MR
{

template <typename C, typename D, D M>
class SparsePolynomial;
template <typename C, typename D, D M>
SparsePolynomial<C,D,M> operator *( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b );

/// The class to store a polynomial with a large number of zero coefficient (only non-zeros are stored in std::map)
/// \tparam C - type of coefficients
/// \tparam D - type of degrees
/// \tparam M - maximum degree to store in the polynomial
template <typename C, typename D, D M>
class SparsePolynomial
{
    static_assert( M > 0 );
public:
    /// constructs zero polynomial
    SparsePolynomial() = default;

    /// takes existing coefficients in ownership
    SparsePolynomial( std::map<D, C> && );

    /// constructs polynomial c0 + c1*x^d1
    SparsePolynomial( C c0, D d1, C c1 );

    /// constructs polynomial c0 + c1*x^d1 + c2*x^d2
    SparsePolynomial( C c0, D d1, C c1, D d2, C c2 );

    /// sets coefficient for given degree to zero
    void setZeroCoeff( D d ) { map_.erase( d ); }

    /// returns true if no single polynomial coefficient is defined
    [[nodiscard]] bool empty() const { return map_.empty(); }

    /// returns true if the coefficient for the smallest not-zero degress is positive
    [[nodiscard]] bool isPositive() const;

    /// gets read-only access to all not-zero coefficients
    [[nodiscard]] const std::map<D, C> & get() const { return map_; }

    SparsePolynomial& operator +=( const SparsePolynomial& b );
    SparsePolynomial& operator -=( const SparsePolynomial& b );
    friend SparsePolynomial operator *<>( const SparsePolynomial& a, const SparsePolynomial& b );

private:
    std::map<D, C> map_; // degree -> not-zero coefficient
};

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>::SparsePolynomial( std::map<D, C> && m ) : map_( std::move( m ) )
{
#ifndef NDEBUG
    for ( const auto & [deg, cf] : map_ )
    {
        assert( deg <= M );
        assert( cf != 0 );
    }
#endif
}

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>::SparsePolynomial( C c0, D d1, C c1 )
{
    assert( c1 != 0 );
    assert( d1 != 0 );
    if ( c0 != 0 )
        map_[D(0)] = c0;
    if ( d1 <= M )
        map_[d1] = c1;
}

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>::SparsePolynomial( C c0, D d1, C c1, D d2, C c2 )
{
    assert( c1 != 0 );
    assert( d1 != 0 );
    assert( c2 != 0 );
    assert( d2 != 0 );
    assert( d1 != d2 );
    if ( c0 != 0 )
        map_[D(0)] = c0;
    if ( d1 <= M )
        map_[d1] = c1;
    if ( d2 <= M )
        map_[d2] = c2;
}

template <typename C, typename D, D M>
bool SparsePolynomial<C,D,M>::isPositive() const
{
    if ( !map_.empty() )
        return map_.begin()->second > 0;

    assert (false);
    return false;
}

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>& SparsePolynomial<C,D,M>::operator +=( const SparsePolynomial& b )
{
    for ( const auto & [degB, cfB] : b.map_ )
    {
        assert( cfB != 0 );
        auto [it,inserted] = map_.insert( { degB, cfB } );
        if ( !inserted )
        {
            const auto sum = it->second + cfB;
            if ( sum != 0 )
                it->second = sum;
            else
                map_.erase( it );
        }
    }
    return * this;
}

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>& SparsePolynomial<C,D,M>::operator -=( const SparsePolynomial& b )
{
    for ( const auto & [degB, cfB] : b.map_ )
    {
        assert( cfB != 0 );
        auto [it,inserted] = map_.insert( { degB, -cfB } );
        if ( !inserted )
        {
            const auto sum = it->second - cfB;
            if ( sum != 0 )
                it->second = sum;
            else
                map_.erase( it );
        }
    }
    return * this;
}

template <typename C, typename D, D M>
[[nodiscard]] SparsePolynomial<C,D,M> operator *( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b )
{
    std::map<D,C> resMap;
    for ( const auto & [degA, cfA] : a.map_ )
    {
        assert( cfA != 0 );
        for ( const auto & [degB, cfB] : b.map_ )
        {
            assert( cfB != 0 );
            const auto deg = degA + degB;
            if ( deg > M )
                break;
            const auto cf = cfA * cfB;
            auto [it,inserted] = resMap.insert( { deg, cf } );
            if ( !inserted )
            {
                const auto sum = it->second + cf;
                if ( sum != 0 )
                    it->second = sum;
                else
                    resMap.erase( it );
            }
        }
    }
    return resMap;
}

} //namespace MR
