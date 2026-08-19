#pragma once

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace MR
{

template <typename C, typename D, D M>
class SparsePolynomial;
template <typename C, typename D, D M>
SparsePolynomial<C,D,M> operator *( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b );

/// The class to store a polynomial with a large number of zero coefficients
/// (only non-zeros are stored in a vector of terms sorted by ascending degree)
/// \tparam C - type of coefficients
/// \tparam D - type of degrees
/// \tparam M - maximum degree to store in the polynomial
template <typename C, typename D, D M>
class SparsePolynomial
{
    static_assert( M > 0 );
public:
    /// a not-zero coefficient with its degree
    using Term = std::pair<D, C>;

    /// constructs zero polynomial
    SparsePolynomial() = default;

    /// takes existing terms in ownership, which must be sorted by ascending degree
    /// with not-zero coefficients and no repeating degrees
    SparsePolynomial( std::vector<Term> && );

    /// constructs polynomial c0 + c1*x^d1
    SparsePolynomial( C c0, D d1, C c1 );

    /// constructs polynomial c0 + c1*x^d1 + c2*x^d2
    SparsePolynomial( C c0, D d1, C c1, D d2, C c2 );

    /// sets coefficient for given degree to zero
    void setZeroCoeff( D d )
    {
        auto it = std::lower_bound( terms_.begin(), terms_.end(), d,
            []( const Term & t, D d ) { return t.first < d; } );
        if ( it != terms_.end() && it->first == d )
            terms_.erase( it );
    }

    /// returns true if no single polynomial coefficient is defined
    [[nodiscard]] bool empty() const { return terms_.empty(); }

    /// returns true if the coefficient for the smallest not-zero degress is positive
    [[nodiscard]] bool isPositive() const;

    /// gets read-only access to all not-zero coefficients
    [[nodiscard]] const std::vector<Term> & get() const { return terms_; }

    SparsePolynomial& operator +=( const SparsePolynomial& b );
    SparsePolynomial& operator -=( const SparsePolynomial& b );
    [[nodiscard]] friend SparsePolynomial operator +( SparsePolynomial a, const SparsePolynomial& b ) { a += b; return a; }
    [[nodiscard]] friend SparsePolynomial operator -( SparsePolynomial a, const SparsePolynomial& b ) { a -= b; return a; }
    friend SparsePolynomial operator *<>( const SparsePolynomial& a, const SparsePolynomial& b );

private:
    /// merges the terms of a degree-sorted sequence, dropping the vanished ones
    void mergeTerms_();

    std::vector<Term> terms_; // sorted by ascending degree
};

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>::SparsePolynomial( std::vector<Term> && terms ) : terms_( std::move( terms ) )
{
#ifndef NDEBUG
    for ( size_t i = 0; i < terms_.size(); ++i )
    {
        assert( terms_[i].first <= M );
        assert( terms_[i].second != 0 );
        assert( i == 0 || terms_[i - 1].first < terms_[i].first );
    }
#endif
}

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>::SparsePolynomial( C c0, D d1, C c1 )
{
    assert( c1 != 0 );
    assert( d1 != 0 );
    if ( c0 != 0 )
        terms_.emplace_back( D(0), c0 );
    if ( d1 <= M )
        terms_.emplace_back( d1, c1 );
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
        terms_.emplace_back( D(0), c0 );
    if ( d1 > d2 )
    {
        std::swap( d1, d2 );
        std::swap( c1, c2 );
    }
    if ( d1 <= M )
        terms_.emplace_back( d1, c1 );
    if ( d2 <= M )
        terms_.emplace_back( d2, c2 );
}

template <typename C, typename D, D M>
bool SparsePolynomial<C,D,M>::isPositive() const
{
    if ( !terms_.empty() )
        return terms_.front().second > 0;

    assert (false);
    return false;
}

template <typename C, typename D, D M>
void SparsePolynomial<C,D,M>::mergeTerms_()
{
    size_t out = 0;
    for ( size_t i = 0; i < terms_.size(); )
    {
        auto deg = terms_[i].first;
        auto cf = std::move( terms_[i].second );
        for ( ++i; i < terms_.size() && terms_[i].first == deg; ++i )
            cf += terms_[i].second;
        if ( cf != 0 )
            terms_[out++] = { deg, std::move( cf ) };
    }
    terms_.resize( out );
}

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>& SparsePolynomial<C,D,M>::operator +=( const SparsePolynomial& b )
{
    std::vector<Term> res;
    res.reserve( terms_.size() + b.terms_.size() );
    std::merge( std::make_move_iterator( terms_.begin() ), std::make_move_iterator( terms_.end() ),
        b.terms_.begin(), b.terms_.end(), std::back_inserter( res ),
        []( const Term & x, const Term & y ) { return x.first < y.first; } );
    terms_ = std::move( res );
    mergeTerms_();
    return * this;
}

template <typename C, typename D, D M>
SparsePolynomial<C,D,M>& SparsePolynomial<C,D,M>::operator -=( const SparsePolynomial& b )
{
    std::vector<Term> res;
    res.reserve( terms_.size() + b.terms_.size() );
    auto itA = terms_.begin();
    auto itB = b.terms_.begin();
    while ( itA != terms_.end() && itB != b.terms_.end() )
    {
        if ( itB->first < itA->first )
        {
            res.emplace_back( itB->first, -itB->second );
            ++itB;
        }
        else
        {
            res.push_back( std::move( *itA ) );
            ++itA;
        }
    }
    for ( ; itA != terms_.end(); ++itA )
        res.push_back( std::move( *itA ) );
    for ( ; itB != b.terms_.end(); ++itB )
        res.emplace_back( itB->first, -itB->second );
    terms_ = std::move( res );
    mergeTerms_();
    return * this;
}

template <typename C, typename D, D M>
[[nodiscard]] SparsePolynomial<C,D,M> operator *( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b )
{
    using Term = typename SparsePolynomial<C,D,M>::Term;
    std::vector<Term> res;
    res.reserve( a.terms_.size() * b.terms_.size() );
    for ( const auto & [degA, cfA] : a.terms_ )
    {
        assert( cfA != 0 );
        for ( const auto & [degB, cfB] : b.terms_ )
        {
            assert( cfB != 0 );
            const auto deg = degA + degB;
            if ( deg > M )
                break;
            res.emplace_back( deg, cfA * cfB );
        }
    }
    std::sort( res.begin(), res.end(),
        []( const Term & x, const Term & y ) { return x.first < y.first; } );
    SparsePolynomial<C,D,M> r;
    r.terms_ = std::move( res );
    r.mergeTerms_();
    return r;
}

} //namespace MR
