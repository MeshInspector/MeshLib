#pragma once

#include "MRVector2.h"
#include "MRVector3.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <memory>
#include <vector>

namespace MR
{

namespace MeshToStep
{

using V3 = Vector3d;
using V2 = Vector2d;
using Tri = std::array<int, 3>;

/// values of (deg+1) non-zero B-spline basis functions at t, and optionally their derivatives;
/// \return the index of the first of them
int bsplineBasis( const std::vector<double>& knots, int deg, int n, double t, double* vals, double* ders );

std::vector<double> clampedKnots( int n, int deg );

/// bicubic B-spline surface over [0,1]x[0,1]
struct BSplineSurf
{
    static constexpr int deg = 3;
    static constexpr int cGrid = 60;
    int nu = 0, nv = 0;
    std::vector<double> tu, tv;
    std::vector<V3> cp; ///< nu * nv points, [i * nv + j]
    double err = 0;     ///< maximum distance from fitted points
    std::vector<V3> grid;

    void eval( const V2& uv, V3& s, V3* su = nullptr, V3* sv = nullptr ) const
    {
        double bu[4], bv[4], du[4], dv[4];
        const int iu = bsplineBasis( tu, deg, nu, uv.x, bu, su ? du : nullptr );
        const int iv = bsplineBasis( tv, deg, nv, uv.y, bv, sv ? dv : nullptr );
        s = V3();
        if ( su )
            *su = V3();
        if ( sv )
            *sv = V3();
        for ( int a = 0; a <= deg; ++a )
            for ( int b = 0; b <= deg; ++b )
            {
                const V3& p = cp[( iu + a ) * nv + iv + b];
                s += bu[a] * bv[b] * p;
                if ( su )
                    *su += du[a] * bv[b] * p;
                if ( sv )
                    *sv += bu[a] * dv[b] * p;
            }
    }

    void buildGrid()
    {
        grid.resize( cGrid * cGrid );
        for ( int i = 0; i < cGrid; ++i )
            for ( int j = 0; j < cGrid; ++j )
                eval( gridUV( i * cGrid + j ), grid[i * cGrid + j] );
    }

    static V2 gridUV( int k ) { return V2( double( k / cGrid ) / ( cGrid - 1 ), double( k % cGrid ) / ( cGrid - 1 ) ); }

    /// parameters of the closest surface point; if no initial guess is given then starts from the closest grid sample
    V2 project( const V3& p, const V2* init = nullptr, int iters = 12 ) const
    {
        V2 uv;
        if ( init )
            uv = *init;
        else
        {
            int best = 0;
            double bestD = DBL_MAX;
            for ( int k = 0; k < (int)grid.size(); ++k )
                if ( auto d = ( grid[k] - p ).lengthSq(); d < bestD )
                {
                    bestD = d;
                    best = k;
                }
            uv = gridUV( best );
        }
        for ( int it = 0; it < iters; ++it )
        {
            V3 s, su, sv;
            eval( uv, s, &su, &sv );
            const V3 r = s - p;
            const double a = dot( su, su ), b = dot( su, sv ), c = dot( sv, sv ), e = -dot( su, r ), f = -dot( sv, r );
            double det = a * c - b * b;
            if ( std::abs( det ) < 1e-30 )
                det = 1e-30;
            uv.x = std::clamp( uv.x + ( c * e - b * f ) / det, 0.0, 1.0 );
            uv.y = std::clamp( uv.y + ( a * f - b * e ) / det, 0.0, 1.0 );
        }
        return uv;
    }

    V3 normalAt( const V2& uv ) const
    {
        V3 s, su, sv;
        eval( uv, s, &su, &sv );
        const V3 n = cross( su, sv );
        const double l = n.length();
        return l > 1e-30 ? n / l : V3( 0, 0, 1 );
    }

    double absDist( const V3& p ) const
    {
        V3 s;
        eval( project( p ), s );
        return ( p - s ).length();
    }
};

/// approximates a triangulation with disk topology by a B-spline surface passing near its vertices;
/// \return nullptr if the triangulation is not a disk or has too few vertices
std::shared_ptr<BSplineSurf> fitBSpline( const std::vector<V3>& pts, const std::vector<Tri>& tris, double tol );

} //namespace MeshToStep

} //namespace MR
