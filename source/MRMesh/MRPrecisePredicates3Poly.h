#pragma once

#include "MRPrecisePredicates3.h"
#include "MRInt64Mul128.h"
#include "MRMacros.h"
#include "MRSparsePolynomial.h"
#include <algorithm>
#include <array>
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
    std::int64_t d = 0; ///< degree of epsilon for pt.z

    /// the degree of epsilon for coordinate i
    [[nodiscard]] std::int64_t degree( int i ) const
    {
        constexpr std::int64_t w[3] = { 9, 3, 1 };
        return w[i] * d;
    }
};

/// the degrees of the perturbation of the eight points of segmentIntersectionOrder and segmentIntersectionTriPlaneOrder:
/// sorted by ids, the points get the degrees 1, 27, 27^2, ..., and points with equal ids get equal degrees;
/// the smallest id moves most, as in orient3d, and since a product of two orient3d-polynomials takes every point at most twice,
/// with the weight at most 9 + 3 + 1 each time, the degrees order the terms by the weights of the points from the largest id down
[[nodiscard]] inline std::array<PointDegree, 8> getPointDegrees( const std::array<PreciseVertCoords, 8> & vs )
{
    struct VertN
    {
        VertId v;
        int n = 0;
    };
    std::array<VertN, 8> as;
    for ( int i = 0; i < 8; ++i )
        as[i] = { vs[i].id, i };
    std::sort( begin( as ), end( as ), []( const auto & a, const auto & b ) { return a.v < b.v; } );

    std::array<PointDegree, 8> res;
    std::int64_t d = 1;
    for ( int i = 0; i < 8; ++i )
    {
        const auto n = as[i].n;
        res[n] = { vs[n].pt, d };
        if ( i < 7 && as[i].v < as[i+1].v ) // skip to support triangles with shared vertices
            d *= 27;
    }
    return res;
}

/// 128 bits are enough to store all coefficients of one orient3d-polynomial (products of three coordinate differences),
/// while the products of two such polynomials are computed in 256 bits, see signOfProductsDiff<Int128Mul256>
template <std::int64_t M>
using VolumePoly = SparsePolynomial<FastInt128, std::int64_t, M>;

/// the volume of tetrahedron abcd (see orient3dVolume) of the perturbed points as a polynomial of eps, without the terms of degrees above M
template <std::int64_t M>
[[nodiscard]] VolumePoly<M> orient3dPoly( const PointDegree & a, const PointDegree & b, const PointDegree & c, const PointDegree & d )
{
    // every element of the matrix with the rows (a-d), (b-d), (c-d) is a polynomial: delta + eps^degP - eps^degD,
    // whose terms are stored below (skipping zero delta), the last term is always -eps^degD
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
            e.terms[e.numTerms++] = { rows[r]->degree( col ), 1 };
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
                const int last2 = k0 + 1 == e0.numTerms && k1 + 1 == e1.numTerms ? e2.numTerms - 1 : e2.numTerms;
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
