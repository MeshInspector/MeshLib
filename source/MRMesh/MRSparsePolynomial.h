#pragma once

#include <map>

namespace MR
{

/// \tparam C - type of coefficients
/// \tparam D - type of degrees
template <typename C, typename D = int>
class SparsePolynomial
{
public:
    /// constructs zero polynomial
    SparsePolynomial() = default;

    /// constructs polynomial c0 + c*x^d
    SparsePolynomial( C c0, D d, C c );

    /// gets access to all not-zero coefficients
          std::map<D, C> & get()       { return map_; }
    const std::map<D, C> & get() const { return map_; }

    SparsePolynomial& operator +=( const SparsePolynomial& b );
    SparsePolynomial& operator -=( const SparsePolynomial& b );
    //friend SparsePolynomial<C,D> operator *( const SparsePolynomial<C,D>& a, const SparsePolynomial<C,D>& b );

private:
    std::map<D, C> map_; // degree -> not-zero coefficient
};

template <typename C, typename D>
SparsePolynomial<C,D>::SparsePolynomial( C c0, D d, C c )
{
    assert( c != 0 );
    assert( d > 0 );
    if ( c0 != 0 )
        map_[D(0)] = c0;
    map_[d] = c;
}

template <typename C, typename D>
SparsePolynomial<C,D>& SparsePolynomial<C,D>::operator +=( const SparsePolynomial& b )
{
    for ( const auto & [degB, cfB] : b.map_ )
    {
        assert( degB >= 0 );
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

template <typename C, typename D>
SparsePolynomial<C,D>& SparsePolynomial<C,D>::operator -=( const SparsePolynomial& b )
{
    for ( const auto & [degB, cfB] : b.map_ )
    {
        assert( degB >= 0 );
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

template <typename C, typename D>
SparsePolynomial<C,D> operator *( const SparsePolynomial<C,D>& a, const SparsePolynomial<C,D>& b )
{
    SparsePolynomial<C,D> res;
    for ( const auto & [degA, cfA] : a.get() )
    {
        assert( degA >= 0 );
        assert( cfA != 0 );
        for ( const auto & [degB, cfB] : b.get() )
        {
            assert( degB >= 0 );
            assert( cfB != 0 );
            const auto deg = degA + degB;
            const auto cf = cfA * cfB;
            auto [it,inserted] = res.get().insert( { deg, cf } );
            if ( !inserted )
            {
                const auto sum = it->second + cf;
                if ( sum != 0 )
                    it->second = sum;
                else
                    res.get().erase( it );
            }
        }
    }
    return res;
}

} //namespace MR
