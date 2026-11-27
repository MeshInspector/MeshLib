#include "MRMeshToStepData.hpp"
#include "MRBitSet.h"
#include "MRFittedBSplineSurface.h"
#include "MRMatrix3.h"
#include "MRMeshTopology.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRSymMatrix3.h"
#include "MRTimer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <optional>

namespace MR
{

namespace MeshToStep
{

namespace
{

/// evenly chosen elements of v, at most maxNum of them
template<typename T>
Vector<T> subsample( const Vector<T>& v, size_t maxNum )
{
    if ( v.size() <= maxNum )
        return v;
    Vector<T> res;
    res.reserve( maxNum );
    for ( size_t i = 0; i < maxNum; ++i )
        res.push_back( v[i * ( v.size() - 1 ) / ( maxNum - 1 )] );
    return res;
}

/// computes the root-mean-square and the maximum distances from the points to the surface
void measure( Region& r, const Vector<V3>& pts )
{
    double sum = 0, mx = 0;
    for ( const auto& p : pts )
    {
        const double dist = std::abs( r.surf.dist( p ) );
        sum += sqr( dist );
        mx = std::max( mx, dist );
    }
    r.rms = pts.empty() ? 0 : std::sqrt( sum / double( pts.size() ) );
    r.maxDist = mx;
}

Vector<V3> pointsOf( const Data& d, const Vector<int>& faces )
{
    Vector<V3> res;
    for ( int v : d.vertsOf( faces ) )
        res.push_back( d.pts[v] );
    return res;
}

/// the normals of the triangles must mostly agree with the normals of a curved surface, otherwise it is fitted to a shape of different kind,
/// e.g. a thin cylinder to a small bump
bool normalsAgree( const Data& d, const Vector<int>& faces, const Surf& s )
{
    static const double cCos45 = std::cos( PI / 4 );
    double bad = 0, total = 0;
    for ( int f : faces )
    {
        total += d.areas[f];
        if ( std::abs( dot( s.normal( d.centers[f] ), d.normals[f] ) ) < cCos45 )
            bad += d.areas[f];
    }
    return bad <= 0.2 * total;
}

/// a face on a surface of revolution with exactly one boundary loop going around the axis would be unbounded on one side,
/// e.g. a small bump on a plane, fitted by a cylinder
bool boundedOnRevolution( const Data& d, const Vector<int>& faces, const Surf& s )
{
    if ( s.type != SurfType::Cyl && s.type != SurfType::Cone && s.type != SurfType::Torus )
        return true;
    FaceBitSet region( d.nf );
    for ( int f : faces )
        region.set( FaceId( f ) );
    const V3 x = s.a.perpendicular().first, y = cross( s.a, x );
    int around = 0;
    for ( const auto& loop : findLeftBoundary( d.mesh->topology, &region ) )
    {
        double total = 0, prev = 0;
        for ( size_t i = 0; i <= loop.size(); ++i )
        {
            const V3 q = d.pts[d.mesh->topology.org( loop[i % loop.size()] )] - s.c;
            const double ang = std::atan2( dot( q, y ), dot( q, x ) );
            if ( i > 0 )
                total += std::remainder( ang - prev, 2 * PI );
            prev = ang;
        }
        if ( std::abs( total ) > PI )
            ++around;
    }
    if ( around != 1 )
        return true;
    // a loop around the apex of a cone bounds it
    if ( s.type == SurfType::Cone )
        for ( int v : d.vertsOf( faces ) )
            if ( ( d.pts[v] - s.c ).length() < 5 * d.edgeTol )
                return true;
    return false;
}

/// the simplest of plane, cylinder, cone, sphere and torus, which fits the faces not much worse than any more complex of them
Region fitAnalytic( const Data& d, const Vector<int>& faces )
{
    const auto pts = pointsOf( d, faces );
    const auto sub = subsample( pts, 2000 );
    Vector<V3> ns, cs;
    for ( int f : subsample( faces, 2000 ) )
    {
        ns.push_back( d.normals[f] );
        cs.push_back( d.centers[f] );
    }

    // the best surface of every type
    constexpr int cNumTypes = int( SurfType::Torus ) + 1;
    std::array<std::optional<Region>, cNumTypes> best;
    auto consider = [&]( std::optional<Surf> s, bool refined )
    {
        if ( !s )
            return;
        if ( refined )
            s = refine( *s, sub );
        if ( !s->finite() || ( s->type != SurfType::Plane && !normalsAgree( d, faces, *s ) ) )
            return;
        Region r{ .surf = *s };
        measure( r, pts );
        auto& b = best[int( s->type )];
        if ( !b || r.rms < b->rms )
            b = std::move( r );
    };
    consider( fitPlane( pts ), false );
    if ( pts.size() >= 4 )
    {
        consider( fitCyl( sub, ns ), true );
        consider( fitConeByNormals( sub, cs, ns ), true );
        consider( fitRevolution( sub, cs, ns, SurfType::Cone ), true );
        consider( fitSphere( sub ), false );
        consider( fitRevolution( sub, cs, ns, SurfType::Torus ), true );
    }

    // more complex type is taken only if it reduces the distance to the points 1.5 times;
    // the distance below 1e-6 of the size of the mesh is not distinguished from zero
    std::optional<Region> res;
    for ( int t = cNumTypes - 1; t >= 0; --t )
        if ( best[t] && boundedOnRevolution( d, faces, best[t]->surf ) && ( !res || best[t]->rms <= std::max( 1.5 * res->rms, 1e-6 * d.diag ) ) )
            res = best[t];
    return *res; // the plane always exists and is bounded
}

/// B-spline surface fitted to the faces, if they form one disk
std::optional<Region> fitBSpline( const Data& d, const Vector<int>& faces )
{
    FaceBitSet region( d.nf );
    for ( int f : faces )
        region.set( FaceId( f ) );
    // only a disk can be parametrized: one boundary loop not touching itself in a vertex
    const auto loops = findLeftBoundary( d.mesh->topology, &region );
    if ( loops.size() != 1 )
        return {};
    Vector<VertId> loopVerts;
    for ( EdgeId e : loops[0] )
        loopVerts.push_back( d.mesh->topology.org( e ) );
    std::sort( loopVerts.begin(), loopVerts.end() );
    if ( std::adjacent_find( loopVerts.begin(), loopVerts.end() ) != loopVerts.end() )
        return {};
    auto bs = fitBSplineSurfaceToMesh( { *d.mesh, &region }, { .tolerance = d.tol } );
    if ( !bs )
        return {};
    Region res;
    res.surf.type = SurfType::BSpline;
    res.surf.bs = std::make_shared<FittedBSplineSurface>( std::move( *bs ) );
    measure( res, pointsOf( d, faces ) );
    return res;
}

/// moves the point to the closest one lying on all the surfaces (approximately if they are nearly tangent);
/// \return the maximum distance from the moved point to the surfaces
double snapPoint( V3& p, const Data& d, const Vector<int>& regions )
{
    // Gauss-Newton steps minimizing the sum of squared distances to the surfaces;
    // the pseudoinverse makes the step of the smallest length: no move along the surfaces,
    // and only along the common normal of nearly tangent surfaces, which keeps the point between them
    for ( int it = 0; it < 30; ++it )
    {
        SymMatrix3d jtj;
        V3 jtf;
        for ( int r : regions )
        {
            const auto& s = d.regions[r].surf;
            const V3 n = s.normal( p );
            jtj += outerSquare( n );
            jtf += n * s.dist( p );
        }
        // not SymMatrix3::pseudoinverse: its direct inverse branch takes a rank-1 matrix (a vertex on one surface) for a regular one
        Matrix3d vecs;
        const V3 vals = jtj.eigensJacobi( &vecs );
        V3 step;
        for ( int k = 0; k < 3; ++k )
            if ( vals[k] > 0.01 * vals[2] )
                step += vecs[k] * ( dot( vecs[k], jtf ) / vals[k] );
        p -= step;
        if ( step.lengthSq() <= sqr( 1e-12 * d.diag ) )
            break;
    }
    double res = 0;
    for ( int r : regions )
        res = std::max( res, std::abs( d.regions[r].surf.dist( p ) ) );
    return res;
}

} //anonymous namespace

bool fitRegions( Data& d, const ProgressCallback& cb )
{
    MR_TIMER;
    const auto faces = d.facesOfRegions();
    if ( !ParallelFor( size_t( 0 ), faces.size(), [&]( size_t r )
    {
        d.regions[r] = fitAnalytic( d, faces[r] );
    }, subprogress( cb, 0.0f, 0.5f ) ) )
        return false;

    if ( !( d.tol > 0 ) )
    {
        // the noise of the mesh: most faces are expected to have analytic surfaces
        Vector<double> rms;
        for ( const auto& r : d.regions )
            rms.push_back( r.rms );
        std::nth_element( rms.begin(), rms.begin() + rms.size() / 2, rms.end() );
        d.tol = std::max( 2.5 * rms[rms.size() / 2], 1e-6 * d.diag );
    }

    return ParallelFor( size_t( 0 ), faces.size(), [&]( size_t r )
    {
        auto& reg = d.regions[r];
        if ( reg.rms > d.tol )
            if ( auto bs = fitBSpline( d, faces[r] ); bs && 1.5 * bs->rms < reg.rms )
                reg = std::move( *bs );
    }, subprogress( cb, 0.5f, 1.0f ) );
}

void snapBoundaries( Data& d )
{
    MR_TIMER;
    struct BdVert
    {
        int v = -1;
        Vector<int> regions;  ///< around the vertex
        Vector<int> nbrs;     ///< indices of the boundary vertices connected with this one by boundary edges
    };
    Vector<BdVert> bv;
    std::map<int, int> index;
    auto bdVert = [&]( int v ) -> BdVert&
    {
        auto [it, inserted] = index.insert( { v, (int)bv.size() } );
        if ( inserted )
            bv.push_back( { .v = v } );
        return bv[it->second];
    };
    for ( int f = 0; f < d.nf; ++f )
    {
        if ( !d.valid( f ) )
            continue;
        for ( int i = 0; i < 3; ++i )
        {
            const int g = d.nbr[f][i];
            if ( g >= 0 && d.sub[g] == d.sub[f] )
                continue;
            const int u = d.tris[f][i], v = d.tris[f][( i + 1 ) % 3];
            for ( int w : { u, v } )
            {
                auto& rs = bdVert( w ).regions;
                if ( std::find( rs.begin(), rs.end(), d.sub[f] ) == rs.end() )
                    rs.push_back( d.sub[f] );
            }
            const int iu = index[u], iv = index[v];
            for ( auto [a, b] : { std::pair{ iu, iv }, std::pair{ iv, iu } } )
                if ( std::find( bv[a].nbrs.begin(), bv[a].nbrs.end(), b ) == bv[a].nbrs.end() )
                    bv[a].nbrs.push_back( b );
        }
    }

    // the selected edges go along the mesh, and zigzag around the curves where the surfaces meet;
    // the zigzags remain after snapping where the surfaces are nearly tangent, so the boundaries are smoothed:
    // the inner vertices of the chains between two regions move toward their neighbours (the boundaries of the mesh stay)
    Vector<V3> pos( bv.size() ), next( bv.size() );
    for ( size_t i = 0; i < bv.size(); ++i )
        pos[i] = d.pts[bv[i].v];
    Vector<double> residual( bv.size(), 0.0 );
    for ( int round = 0; round < 6; ++round )
    {
        if ( round > 0 )
            for ( int it = 0; it < 3; ++it )
            {
                ParallelFor( size_t( 0 ), bv.size(), [&]( size_t i )
                {
                    const auto& b = bv[i];
                    next[i] = b.nbrs.size() == 2 && b.regions.size() == 2 ? ( 2.0 * pos[i] + pos[b.nbrs[0]] + pos[b.nbrs[1]] ) / 4.0 : pos[i];
                } );
                std::swap( pos, next );
            }
        ParallelFor( size_t( 0 ), bv.size(), [&]( size_t i )
        {
            residual[i] = snapPoint( pos[i], d, bv[i].regions );
        } );
    }
    d.snapErr.assign( d.pts.size(), 0.0 );
    for ( size_t i = 0; i < bv.size(); ++i )
    {
        d.pts[bv[i].v] = pos[i];
        d.snapErr[bv[i].v] = residual[i];
    }
}

} //namespace MeshToStep

} //namespace MR
