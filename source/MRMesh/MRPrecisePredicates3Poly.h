#pragma once

#include "MRPrecisePredicates3.h"
#include "MRInt64Mul128.h"
#include "MRMacros.h"
#include "MRSparsePolynomial.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <type_traits>
#include <vector>

namespace MR
{

/// \addtogroup MathGroup
/// \{

// for internal use: namespace detail is not bound to other languages
namespace detail
{

/// the building blocks of the precise predicates in MRPrecisePredicates3.h, shared with the other predicates that must agree with them:
/// the exact volume of a tetrahedron, and its polynomial in eps when the points are perturbed by simulation-of-simplicity

/// the exact signed volume of tetrahedron abcd times 6: dot( a - d, cross( b - d, c - d ) ), whose sign orient3d( a, b, c, d ) returns
/// if it is not zero; 128 bits are enough for coordinates within the range of ConvertToIntVector
[[nodiscard]] MR_FORCE_INLINE FastInt128 orient3dVolume( const Vector3i & a, const Vector3i & b, const Vector3i & c, const Vector3i & d )
{
    const Vector3i64 x( a - d );
    const Vector3i64 y( b - d );
    const Vector3i64 z( c - d );

    return
        Int64Mul128( x.x ) * Int64Mul128( y.y * z.z - y.z * z.y )
     -  Int64Mul128( x.y ) * Int64Mul128( y.x * z.z - y.z * z.x )
     +  Int64Mul128( x.z ) * Int64Mul128( y.x * z.y - y.y * z.x );
}

/// a point with the degree of its perturbation: its coordinates z, y, x get eps^d, eps^(3*d), eps^(9*d)
struct PointDegree
{
    Vector3i pt;
    std::int64_t d = 0; ///< degree of epsilon for pt.z; 0 leaves the point unperturbed

    /// the degree of epsilon for coordinate i
    [[nodiscard]] std::int64_t degree( int i ) const
    {
        constexpr std::int64_t w[3] = { 9, 3, 1 };
        return w[i] * d;
    }
};

/// the largest number of points of one predicate that getPointDegrees and PerturbationStages take by default
constexpr int cMaxPerturbedPoints = 16;

/// finds the degrees of the perturbation of points vs[0..n-1] and writes them in res[0..n-1]: sorted by ids, the points get
/// the degrees 1, r_0, r_0 * r_1, ..., where r_k is the radix of the k-th id, given in radices[] of every point with that id,
/// and points with equal ids get equal degrees; the smallest id moves most, as in orient3d, and the radix of a point must exceed
/// its weight in every term of the polynomials of the predicate (the sum of 9, 3, 1 over its coordinates in the term), so that
/// the degrees order the terms as the perturbation of orient3d does: by the weights of the points from the largest id down;
/// returns the number of different ids, and writes their degrees in ascending order in places[] if it is not null;
/// n is at most N; the caller makes the degrees fit in int64, unless Bounded: then only the first ids whose terms fit count,
/// since the terms with the k-th id and smaller ones have degrees below r_0 * ... * r_k, and the points of the other ids
/// get the largest int64 as their degree, which no stage reaches (see PerturbationStagesOf)
template <int N = cMaxPerturbedPoints, bool Bounded = false>
MR_FORCE_INLINE int getPointDegrees( const PreciseVertCoords * vs, const std::int64_t * radices, int n, PointDegree * res,
    std::int64_t * places = nullptr )
{
    struct VertN
    {
        VertId v;
        int n = 0;
    };
    assert( n > 0 && n <= N );
    std::array<VertN, N> as;
    for ( int i = 0; i < n; ++i )
        as[i] = { vs[i].id, i };
    std::sort( as.begin(), as.begin() + n, []( const auto & a, const auto & b ) { return a.v < b.v; } );

    std::int64_t d = 1;
    int numIds = 1;
    if ( places )
        places[0] = d;
    for ( int i = 0; i < n; ++i )
    {
        const auto k = as[i].n;
        if constexpr ( Bounded )
        {
            if ( d > std::numeric_limits<std::int64_t>::max() / radices[k] )
            {
                // the degrees of the terms with this id do not fit: the ids before it count
                for ( int j = i; j < n; ++j )
                    res[as[j].n] = { vs[as[j].n].pt, std::numeric_limits<std::int64_t>::max() };
                return numIds - 1;
            }
        }
        res[k] = { vs[k].pt, d };
        if ( i + 1 == n )
            break;
        const auto r = radices[k];
        if ( as[i].v < as[i+1].v ) // skip to support triangles with shared vertices
        {
            assert( d <= std::numeric_limits<std::int64_t>::max() / r );
            d *= r;
            if ( places )
                places[numIds] = d;
            ++numIds;
        }
        else
            assert( r == radices[as[i+1].n] );
    }
    return numIds;
}

/// the largest degree that no term exceeds: a polynomial with it keeps all its terms
constexpr std::int64_t cMaxPolyDAll = std::numeric_limits<std::int64_t>::max();

/// the largest degrees of the polynomial terms that segmentIntersectionOrder and segmentIntersectionTriPlaneOrder consider first, found
/// experimentally (the leading term of nom, or the leading term of at least one of two orient3d-polynomials for the segment's ends)
/// in the tests and in a random sweep of degenerate inputs; the terms of higher degrees are not stored to save computation time,
/// and if they are not enough to decide, the predicates repeat the computation with all terms (cMaxPolyDAll);
/// the tests count the inputs above them, so they live here rather than in MRPrecisePredicates3.cpp
constexpr std::int64_t cMaxPolyDTriTri   = 15'943'959; // segmentIntersectionOrder
constexpr std::int64_t cMaxPolyDTriPlane = 430'486'893; // segmentIntersectionTriPlaneOrder: it reaches the polynomial path with all three largest ids in the plane

/// the points of a predicate with the degrees of their perturbation, which moves them in stages: at stage s = 1, ..., numStages()
/// only the points with the s smallest ids are perturbed (orient3dPoly leaves the others unperturbed); a polynomial of the predicate
/// that is not zero at stage s already has the leading term of the full one, since every term with a point of a later stage has
/// a larger degree than every term without; and if it was zero at stage s - 1, its terms without the points of stage s cancel
/// one another, so only the products of the terms of degrees from place( s ) up remain (see `from` in lowestTermOfSumOfProducts);
/// at most N points; if Bounded, the degrees need not fit in int64, and the stages end before the first one whose terms can have
/// degrees above it (see getPointDegrees), so that firstStage gives the empty value if no stage before gives one
template <int N, bool Bounded>
class PerturbationStagesOf
{
public:
    /// the stages of points vs[0..n-1] with the degrees from getPointDegrees
    MR_FORCE_INLINE PerturbationStagesOf( const PreciseVertCoords * vs, const std::int64_t * radices, int n )
    {
        assert( n <= N );
        numStages_ = getPointDegrees<N, Bounded>( vs, radices, n, points_.data(), places_.data() );
    }

    /// the number of stages, which is the number of different ids (if Bounded, of those before the first one whose degrees do not fit)
    [[nodiscard]] int numStages() const { return numStages_; }

    /// the degree of the points perturbed first at stage s: every term with them has at least this degree, every term of the earlier stages less
    [[nodiscard]] std::int64_t place( int s ) const { return places_[s - 1]; }

    /// point i with its degree, as perturbed at the last stage (if Bounded, with the largest int64 if no stage perturbs it)
    [[nodiscard]] const PointDegree & operator[]( int i ) const { return points_[i]; }

    /// point i as perturbed at stage s
    [[nodiscard]] PointDegree at( int i, int s ) const { return points_[i].d <= place( s ) ? points_[i] : PointDegree{ points_[i].pt, 0 }; }

    /// whether point i is perturbed first at stage s
    [[nodiscard]] bool startsAt( int i, int s ) const { return points_[i].d == place( s ); }

    /// calls f( s ) for the stages s = 1, 2, ... until it returns a value, and returns that value (or the empty one after the last stage);
    /// the stages with places above maxDeg are skipped, since they add only the terms of degrees above it
    template <typename F>
    [[nodiscard]] auto firstStage( F && f, std::int64_t maxDeg = cMaxPolyDAll ) const -> std::invoke_result_t<F &, int>
    {
        for ( int s = 1; s <= numStages_ && place( s ) <= maxDeg; ++s )
            if ( auto res = f( s ) )
                return res;
        return {};
    }

private:
    std::array<PointDegree, N> points_;
    std::array<std::int64_t, N> places_;
    int numStages_ = 0;
};

/// the stages of at most cMaxPerturbedPoints points, whose degrees the caller makes fit in int64
using PerturbationStages = PerturbationStagesOf<cMaxPerturbedPoints, false>;

/// 128 bits are enough to store all coefficients of one orient3d-polynomial (products of three coordinate differences),
/// while the products of two such polynomials are computed in 256 bits, see signOfProductsDiff<Int128Mul256>
template <std::int64_t M>
using VolumePoly = SparsePolynomial<FastInt128, std::int64_t, M>;

/// the volume of tetrahedron abcd (see orient3dVolume) of the perturbed points as a polynomial of eps, without the terms of degrees above M
template <std::int64_t M>
[[nodiscard]] VolumePoly<M> orient3dPoly( const PointDegree & a, const PointDegree & b, const PointDegree & c, const PointDegree & d )
{
    // every element of the matrix with the rows (a-d), (b-d), (c-d) is a polynomial: delta + eps^degP - eps^degD,
    // whose terms are stored below (skipping zero delta and the terms of unperturbed points), the last term is -eps^degD if d is perturbed
    struct Element
    {
        typename VolumePoly<M>::Term terms[3];
        int numTerms = 0;
    };
    Element m[3][3];
    const PointDegree * rows[3] = { &a, &b, &c };
    for ( int r = 0; r < 3; ++r )
        for ( int col = 0; col < 3; ++col )
        {
            auto & e = m[r][col];
            if ( const std::int64_t delta = std::int64_t( rows[r]->pt[col] ) - d.pt[col] )
                e.terms[e.numTerms++] = { 0, delta };
            if ( rows[r]->d )
                e.terms[e.numTerms++] = { rows[r]->degree( col ), 1 };
            if ( d.d )
                e.terms[e.numTerms++] = { d.degree( col ), -1 };
        }

    // the determinant is the sum over 6 permutations of the signed products of three elements, each product has up to 27 monomials;
    // the product of the last terms of three elements (eps^(13*degD) with the sign of permutation) is excluded, since 6 of them cancel one another
    constexpr int perms[6][4] = { { 0, 1, 2, 1 }, { 1, 2, 0, 1 }, { 2, 0, 1, 1 }, { 0, 2, 1, -1 }, { 2, 1, 0, -1 }, { 1, 0, 2, -1 } };
    std::vector<typename VolumePoly<M>::Term> terms;
    terms.reserve( 6 * 26 );
    for ( const auto & p : perms )
    {
        const Element & e0 = m[0][p[0]];
        const Element & e1 = m[1][p[1]];
        const Element & e2 = m[2][p[2]];
        for ( int k0 = 0; k0 < e0.numTerms; ++k0 )
            for ( int k1 = 0; k1 < e1.numTerms; ++k1 )
            {
                // all coefficients are not larger than 2^32 by absolute value, so the product of two of them fits in int64
                const std::int64_t c01 = p[3] * std::int64_t( e0.terms[k0].second ) * std::int64_t( e1.terms[k1].second );
                const std::int64_t d01 = e0.terms[k0].first + e1.terms[k1].first;
                const int last2 = d.d && k0 + 1 == e0.numTerms && k1 + 1 == e1.numTerms ? e2.numTerms - 1 : e2.numTerms;
                for ( int k2 = 0; k2 < last2; ++k2 )
                    if ( const auto deg = d01 + e2.terms[k2].first; deg <= M )
                        terms.emplace_back( deg, Int64Mul128( c01 ) * Int64Mul128( std::int64_t( e2.terms[k2].second ) ) );
            }
    }
    return VolumePoly<M>::fromUnsortedTerms( std::move( terms ) );
}

} // namespace detail

/// \}

} // namespace MR
