#pragma once

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace MR
{

template <typename C, typename D, D M>
class SparsePolynomial;

/// the type of the polynomial with coefficients of the type of the product of two values of type T
template <typename T, typename D, D M>
using SparsePolynomialProduct = SparsePolynomial<decltype( std::declval<T>() * std::declval<T>() ), D, M>;

/// computes the product of two polynomials, converting every coefficient into type T before multiplication;
/// e.g. T=Int128Mul256 multiplies FastInt128 coefficients without overflow into a polynomial with FastInt256 coefficients
template <typename T, typename C, typename D, D M>
SparsePolynomialProduct<T,D,M> mulAs( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b );

/// computes the product of two polynomials with the coefficients of the same type
template <typename C, typename D, D M>
SparsePolynomial<C,D,M> operator *( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b ) { return mulAs<C>( a, b ); }

/// returns the sign of the polynomial ( a * b - c * d ) for infinitesimal positive argument, i.e. the sign of its lowest-degree not-zero coefficient,
/// or 0 if all its coefficients of degrees not above M are zeros (the terms of higher degrees are not considered, since the arguments store the terms of degrees not above M only);
/// every coefficient is converted into type T before multiplication, and the coefficients of the products have the type of T*T;
/// the coefficients of the difference are computed in the order of increasing degree and only till the first not-zero one
template <typename T, typename C, typename D, D M>
[[nodiscard]] int signOfProductsDiff( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b,
    const SparsePolynomial<C,D,M>& c, const SparsePolynomial<C,D,M>& d );

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

    /// constructs polynomial from arbitrary terms: sorts them by degree, sums the coefficients of equal degrees, drops zero coefficients and the degrees above M
    [[nodiscard]] static SparsePolynomial fromUnsortedTerms( std::vector<Term> && terms );

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
    template <typename T, typename C2, typename D2, D2 M2>
    friend SparsePolynomialProduct<T,D2,M2> mulAs( const SparsePolynomial<C2,D2,M2>& a, const SparsePolynomial<C2,D2,M2>& b );

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
SparsePolynomial<C,D,M> SparsePolynomial<C,D,M>::fromUnsortedTerms( std::vector<Term> && terms )
{
    terms.erase( std::remove_if( terms.begin(), terms.end(), []( const Term & t ) { return t.first > M; } ), terms.end() );
    std::sort( terms.begin(), terms.end(), []( const Term & x, const Term & y ) { return x.first < y.first; } );
    SparsePolynomial res;
    res.terms_ = std::move( terms );
    res.mergeTerms_();
    return res;
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

template <typename T, typename C, typename D, D M>
[[nodiscard]] SparsePolynomialProduct<T,D,M> mulAs( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b )
{
    using Res = SparsePolynomialProduct<T,D,M>;
    using Term = typename Res::Term;
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
            res.emplace_back( deg, T( cfA ) * T( cfB ) );
        }
    }
    return Res::fromUnsortedTerms( std::move( res ) );
}

template <typename T, typename C, typename D, D M>
int signOfProductsDiff( const SparsePolynomial<C,D,M>& a, const SparsePolynomial<C,D,M>& b,
    const SparsePolynomial<C,D,M>& c, const SparsePolynomial<C,D,M>& d )
{
    // the terms of a*b (and of c*d) form rows: the i-th term of a times all the terms of b in the order of increasing degree;
    // the heap keeps the current term of every row, so all terms of both products are visited in the order of increasing degree
    struct RowTerm
    {
        D deg;
        int i, j; // indices of the terms in the two factors
        bool neg; // true for the terms of c*d
    };
    auto greater = []( const RowTerm & x, const RowTerm & y ) { return x.deg > y.deg; };
    std::vector<RowTerm> heap;
    heap.reserve( a.get().size() + c.get().size() );
    if ( !b.get().empty() )
        for ( int i = 0; i < (int)a.get().size(); ++i )
            heap.push_back( { a.get()[i].first + b.get()[0].first, i, 0, false } );
    if ( !d.get().empty() )
        for ( int i = 0; i < (int)c.get().size(); ++i )
            heap.push_back( { c.get()[i].first + d.get()[0].first, i, 0, true } );
    std::make_heap( heap.begin(), heap.end(), greater );

    while ( !heap.empty() )
    {
        const auto deg = heap.front().deg;
        if ( deg > M )
            break;
        decltype( std::declval<T>() * std::declval<T>() ) coeff{};
        do
        {
            std::pop_heap( heap.begin(), heap.end(), greater );
            auto r = heap.back();
            heap.pop_back();
            const auto & f = ( r.neg ? c : a ).get();
            const auto & g = ( r.neg ? d : b ).get();
            const auto prod = T( f[r.i].second ) * T( g[r.j].second );
            if ( r.neg )
                coeff -= prod;
            else
                coeff += prod;
            if ( ++r.j < (int)g.size() )
            {
                r.deg = f[r.i].first + g[r.j].first;
                heap.push_back( r );
                std::push_heap( heap.begin(), heap.end(), greater );
            }
        }
        while ( !heap.empty() && heap.front().deg == deg );
        if ( coeff > 0 )
            return 1;
        if ( coeff < 0 )
            return -1;
    }
    return 0;
}

} //namespace MR
