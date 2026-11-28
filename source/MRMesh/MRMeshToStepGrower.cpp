#include "MRMeshToStepData.hpp"
#include "MRBestFit.h"
#include "MRMatrix3.h"
#include "MRTimer.h"

#include <initializer_list>
#include <random>
#include <tuple>
#include <utility>

namespace MR
{

namespace MeshToStep
{

namespace
{

/// finds regions of the mesh lying on planes, cylinders, cones, spheres and tori by growing them from seed triangles
class Grower
{
public:
    Grower( Data& d, const ProgressCallback& cb ) : d_( d ), cb_( cb )
    {
        avail_.resize( d.nf, 0 );
        fail_.resize( d.nf, 0 );
        faceStamp_.resize( d.nf, 0 );
        coveredStamp_.resize( d.nf, 0 );
    }

    /// \return false if the operation was canceled
    bool run();

private:
    struct Found
    {
        double area = 0, dev = 0;
        Surf surf;
        Vector<int> faces;
    };

    static constexpr size_t cMinVerts = 12;
    static constexpr int cMinStrongFaces = 20;

    std::optional<Vector<int>> growOnce_( Surf& s, int seed, std::initializer_list<double> sched );
    std::optional<Vector<int>> grow_( Surf& s, int seed );
    Vector<int> patch_( int seed, int rings ) const;
    Vector<int> sample_( const Vector<int>& from, size_t num );
    Vector<V3> normalsOf_( const Vector<int>& faces ) const;
    Vector<V3> centersOf_( const Vector<int>& faces ) const;
    void addIfFits_( Vector<Surf>& out, std::optional<Surf> s, const Vector<V3>& pts, bool allowRefine ) const;
    void candidates_( int seed, Vector<Surf>& out );
    std::optional<Surf> simpler_( const Surf& s, const Vector<int>& faces ) const;
    bool better_( const Found& x, const Found& y ) const;
    bool strong_( const Region& r ) const;
    void processSegment_( int seg, const Vector<int>& segFaces );

    Data& d_;
    ProgressCallback cb_;
    Vector<char> avail_;
    Vector<int> fail_;
    Vector<int> faceStamp_, coveredStamp_;
    int stamp_ = 0;
    std::mt19937 rng_{ 1 };
    Vector<Surf> known_; ///< curved surfaces found in the previous pass, they may continue into other segments
    Vector<Axis> lib_;   ///< axes of cylinders and cones found in the previous pass
    int pass_ = 1;
};

std::optional<Vector<int>> Grower::growOnce_( Surf& s, int seed, std::initializer_list<double> sched )
{
    Vector<int> reg;
    for ( int it = 0; it < 10; ++it )
    {
        const double tol = d_.tol * ( it < (int)sched.size() ? sched.begin()[it] : 1.0 );
        if ( !d_.faceOnSurf( s, seed, tol ) )
            return {};
        ++stamp_;
        Vector<int> cur{ seed };
        faceStamp_[seed] = stamp_;
        for ( size_t i = 0; i < cur.size(); ++i )
            for ( int k = 0; k < 3; ++k )
            {
                const int g = d_.nbrSameSeg( cur[i], k );
                if ( g < 0 || !avail_[g] || faceStamp_[g] == stamp_ )
                    continue;
                // a curved surface is not expected to be tessellated with large angles between adjacent facets:
                // it is rather a prism or a pyramid
                if ( s.type != SurfType::Plane && dot( d_.normals[cur[i]], d_.normals[g] ) < cCosFacets )
                    continue;
                faceStamp_[g] = stamp_;
                if ( d_.faceOnSurf( s, g, tol ) )
                    cur.push_back( g );
            }
        std::sort( cur.begin(), cur.end() );
        if ( it >= (int)sched.size() && cur == reg )
            break;
        reg = std::move( cur );
        if ( s.known )
            break;
        const auto numVerts = d_.vertsOf( reg ).size();
        if ( s.type != SurfType::Plane && numVerts < cMinVerts && it == 0 )
            return {};
        if ( numVerts < 4 )
            break;
        s = refine( s, d_.pointsOf( reg, 600 ) );
    }
    return reg;
}

std::optional<Vector<int>> Grower::grow_( Surf& s, int seed )
{
    const Surf s0 = s;
    auto res = growOnce_( s, seed, {} );
    if ( !res && s0.type != SurfType::Plane && !s0.known )
    {
        // coarse facets give a poor start: begin with a large tolerance and decrease it gradually
        s = s0;
        res = growOnce_( s, seed, { 50, 10, 3 } );
    }
    return res;
}

Vector<int> Grower::patch_( int seed, int rings ) const
{
    Vector<int> res{ seed };
    size_t begin = 0;
    for ( int r = 0; r < rings; ++r )
    {
        const size_t end = res.size();
        for ( size_t i = begin; i < end; ++i )
            for ( int k = 0; k < 3; ++k )
            {
                const int g = d_.nbrSameSeg( res[i], k );
                if ( g >= 0 && avail_[g] && std::find( res.begin(), res.end(), g ) == res.end() )
                    res.push_back( g );
            }
        begin = end;
    }
    return res;
}

Vector<int> Grower::sample_( const Vector<int>& from, size_t num )
{
    auto v = from;
    num = std::min( num, v.size() );
    for ( size_t i = 0; i < num; ++i )
        std::swap( v[i], v[i + rng_() % ( v.size() - i )] );
    v.resize( num );
    return v;
}

Vector<V3> Grower::normalsOf_( const Vector<int>& faces ) const
{
    Vector<V3> res;
    res.reserve( faces.size() );
    for ( int f : faces )
        res.push_back( d_.normals[f] );
    return res;
}

Vector<V3> Grower::centersOf_( const Vector<int>& faces ) const
{
    Vector<V3> res;
    res.reserve( faces.size() );
    for ( int f : faces )
        res.push_back( d_.centers[f] );
    return res;
}

void Grower::addIfFits_( Vector<Surf>& out, std::optional<Surf> s, const Vector<V3>& pts, bool allowRefine ) const
{
    if ( !s || !s->finite() )
        return;
    double e = s->maxDist( pts );
    if ( allowRefine && e > d_.tol && e <= 100 * d_.tol )
    {
        s = refine( *s, pts );
        e = s->maxDist( pts );
    }
    if ( s->finite() && e <= d_.tol )
        out.push_back( *s );
}

void Grower::candidates_( int seed, Vector<Surf>& out )
{
    const auto& d = d_;
    Surf plane;
    plane.c = d.centers[seed];
    plane.a = d.normals[seed];
    out.push_back( plane );
    for ( int rings : { 1, 2 } )
    {
        const auto pts = d.pointsOf( patch_( seed, rings ) );
        if ( pts.size() >= 4 )
            if ( auto s = fitPlane( pts ); s.maxDist( pts ) < d.tol )
                out.push_back( s );
    }

    // a cylinder through two triangles, a cone through three triangles
    auto others = patch_( seed, 3 );
    others.erase( others.begin() );
    if ( others.size() >= 2 )
    {
        for ( int q : sample_( others, 8 ) )
        {
            V3 a = cross( d.normals[seed], d.normals[q] );
            const double len = a.length();
            if ( len < 0.02 )
                continue;
            a /= len;
            const auto [u, v] = a.perpendicular();
            Vector<V2> proj;
            for ( int f : { seed, q } )
                for ( int vert : d.tris[f] )
                    proj.emplace_back( dot( d.pts[vert], u ), dot( d.pts[vert], v ) );
            if ( const auto circle = findBestCircle( proj ) )
            {
                Surf s;
                s.type = SurfType::Cyl;
                s.a = a;
                s.r = circle->radius;
                s.c = circle->center.x * u + circle->center.y * v;
                out.push_back( s );
            }
        }
        for ( int i = 0; i < 8; ++i )
        {
            const auto q = sample_( others, 2 );
            const int fs[3] = { seed, q[0], q[1] };
            const V3 n0 = d.normals[fs[0]], n1 = d.normals[fs[1]], n2 = d.normals[fs[2]];
            V3 a = cross( n0 - n1, n0 - n2 );
            const double len = a.length();
            if ( len < 1e-3 )
                continue;
            a /= len;
            const auto m = Matrix3d::fromRows( n0, n1, n2 );
            if ( std::abs( dot( n0, a ) ) < 0.02 || std::abs( m.det() ) < 1e-6 )
                continue;
            const V3 apex = m.inverse() * V3( dot( n0, d.centers[fs[0]] ), dot( n1, d.centers[fs[1]] ), dot( n2, d.centers[fs[2]] ) );
            Vector<double> hs;
            double sumTan = 0, minH = DBL_MAX;
            for ( int f : fs )
                for ( int vert : d.tris[f] )
                    hs.push_back( dot( d.pts[vert] - apex, a ) );
            auto sorted = hs;
            std::sort( sorted.begin(), sorted.end() );
            const double sign = sorted[sorted.size() / 2] < 0 ? -1 : 1;
            a *= sign;
            for ( int f : fs )
                for ( int vert : d.tris[f] )
                {
                    const V3 dv = d.pts[vert] - apex;
                    const double h = dot( dv, a );
                    minH = std::min( minH, h );
                    sumTan += ( dv - h * a ).length() / h;
                }
            if ( minH <= 0 )
                continue;
            Surf s;
            s.type = SurfType::Cone;
            s.c = apex;
            s.a = a;
            s.r = std::atan( sumTan / 9 );
            if ( s.finite() )
                out.push_back( s );
        }
    }

    // coarse facets: a cylinder through three triangles of similar size (two triangles give a poor radius)
    {
        Vector<int> big;
        for ( int f : patch_( seed, 6 ) )
            if ( f != seed && d.areas[f] >= 0.2 * d.areas[seed] )
                big.push_back( f );
        if ( big.size() >= 2 )
        {
            if ( big.size() > 12 )
            {
                std::sort( big.begin(), big.end(), [&]( int x, int y )
                {
                    return ( d.centers[x] - d.centers[seed] ).lengthSq() < ( d.centers[y] - d.centers[seed] ).lengthSq();
                } );
                big.resize( 12 );
            }
            const size_t numTries = std::min<size_t>( 12, big.size() * ( big.size() - 1 ) / 2 );
            for ( size_t i = 0; i < numTries; ++i )
            {
                const auto q = sample_( big, 2 );
                const Vector<int> fs{ seed, q[0], q[1] };
                const auto pts = d.pointsOf( fs );
                if ( auto s0 = fitCyl( pts, normalsOf_( fs ) ) )
                {
                    const Surf s = refine( *s0, pts );
                    if ( s.finite() && s.r < 100 * d.diag && s.maxDist( pts ) < d.tol )
                        out.push_back( s );
                }
            }
        }
    }

    // surfaces fitted to all triangles around the seed
    for ( int rings : { 2, 4, 8 } )
    {
        const auto p = patch_( seed, rings );
        if ( p.size() < 6 )
            continue;
        const auto pts = d.pointsOf( p );
        if ( pts.size() < cMinVerts )
            continue;
        const auto ns = normalsOf_( p );
        const auto cs = centersOf_( p );
        addIfFits_( out, fitCyl( pts, ns ), pts, true );
        addIfFits_( out, fitRevolution( pts, cs, ns, SurfType::Cone ), pts, true );
        addIfFits_( out, fitConeByNormals( pts, cs, ns ), pts, true );
        addIfFits_( out, fitSphere( pts ), pts, true );
        addIfFits_( out, fitRevolution( pts, cs, ns, SurfType::Torus ), pts, true );
    }

    // tori and cones around the known axes
    if ( !lib_.empty() )
        for ( int rings : { 1, 2, 4 } )
        {
            const auto pts = d.pointsOf( patch_( seed, rings ) );
            if ( pts.size() < 6 )
                continue;
            for ( const auto& ax : lib_ )
            {
                if ( auto s = axisTorus( pts, ax.dir, ax.pos ); s && s->maxDist( pts ) < d.tol )
                    out.push_back( *s );
                // the distance to the cone measured orthogonally to the axis is compared
                if ( auto s = axisCone( pts, ax.dir, ax.pos ); s && s->maxDist( pts ) < d.tol * std::cos( s->r ) && std::tan( s->r ) > 1e-4 && std::tan( s->r ) < 1e4 )
                    out.push_back( *s );
            }
        }

    for ( const auto& s : known_ )
    {
        bool fits = true;
        for ( int vert : d.tris[seed] )
            fits = fits && std::abs( s.dist( d.pts[vert] ) ) < d.tol;
        if ( fits )
            out.push_back( s );
    }
}

/// a simpler surface passing through all vertices of the region, if any
std::optional<Surf> Grower::simpler_( const Surf& s, const Vector<int>& faces ) const
{
    const auto pts = d_.pointsOf( faces, 600 );
    const auto ns = normalsOf_( faces );
    for ( int variant = 0; variant < 4; ++variant )
    {
        const auto type = variant == 0 ? SurfType::Plane : variant == 1 ? SurfType::Cyl : SurfType::Cone;
        if ( type >= s.type )
            break;
        std::optional<Surf> q;
        if ( variant == 0 )
            q = fitPlane( pts );
        else if ( variant == 1 )
            q = fitCyl( pts, ns );
        else if ( variant == 2 )
            q = fitConeByNormals( pts, centersOf_( faces ), ns );
        else
            q = fitRevolution( pts, centersOf_( faces ), ns, SurfType::Cone );
        if ( !q )
            continue;
        if ( type != SurfType::Plane )
            q = refine( *q, pts );
        if ( !q->finite() || !( q->maxDist( pts ) < d_.tol ) )
            continue;
        bool ok = true;
        for ( int f : faces )
            ok = ok && std::abs( dot( q->normal( d_.centers[f] ), d_.normals[f] ) ) > cCos40;
        if ( ok )
            return q;
    }
    return {};
}

/// for equal areas: the simpler type wins unless it leaves the facets much farther from the surface
bool Grower::better_( const Found& x, const Found& y ) const
{
    if ( x.surf.type == y.surf.type )
        return x.dev < y.dev;
    const bool xSimpler = x.surf.type < y.surf.type;
    const Found& lo = xSimpler ? x : y;
    const Found& hi = xSimpler ? y : x;
    const bool loWins = lo.dev <= 3 * hi.dev + 0.5 * d_.tol;
    return loWins == xSimpler;
}

bool Grower::strong_( const Region& r ) const
{
    if ( r.surf.type == SurfType::Plane )
        return r.numFaces >= 3;
    if ( r.numFaces < cMinStrongFaces )
        return false;
    return r.surf.type != SurfType::Torus || isCoaxial( r.surf, lib_, d_.tol );
}

void Grower::processSegment_( int seg, const Vector<int>& segFaces )
{
    auto& d = d_;
    Vector<int> faces;
    for ( int f : segFaces )
        if ( d.sub[f] < 0 )
        {
            faces.push_back( f );
            avail_[f] = 1;
        }
    auto addRegion = [&]( const Surf& s, const Vector<int>& fs )
    {
        Region r;
        r.surf = s;
        r.seg = seg;
        for ( int f : fs )
        {
            d.sub[f] = (int)d.regions.size();
            avail_[f] = 0;
            ++r.numFaces;
            r.area += d.areas[f];
        }
        d.regions.push_back( std::move( r ) );
    };
    Vector<Surf> cands;
    while ( !faces.empty() )
    {
        // first the largest triangles not failed before
        std::sort( faces.begin(), faces.end(), [&]( int x, int y )
        {
            return std::tuple( fail_[x], -d.areas[x], x ) < std::tuple( fail_[y], -d.areas[y], y );
        } );
        Vector<Found> found;
        bool anyTried = false;
        for ( int phase = 0; phase < 4; ++phase )
        {
            // phase 0: surfaces of all types from the seeds not failed before;
            // next phases, after all seeds have failed: only planes of at least minPlane triangles
            const bool full = phase == 0;
            const int minPlane = full ? 3 : 4 - phase;
            if ( !full && anyTried )
                break;
            const int coveredStamp = ++stamp_;
            int tried = 0;
            for ( int seed : faces )
            {
                if ( coveredStamp_[seed] == coveredStamp )
                    continue;
                if ( d.areas[seed] <= 0 )
                    continue;
                if ( full )
                {
                    if ( fail_[seed] > 0 )
                        break;
                    ++tried;
                    if ( tried > ( pass_ == 1 ? 300 : 60 ) || ( !found.empty() && tried > 40 ) )
                        break;
                    anyTried = true;
                }
                cands.clear();
                if ( full )
                    candidates_( seed, cands );
                else
                {
                    Surf plane;
                    plane.c = d.centers[seed];
                    plane.a = d.normals[seed];
                    cands.push_back( plane );
                }
                std::optional<Found> best;
                for ( auto& s : cands )
                {
                    auto reg = grow_( s, seed );
                    if ( !reg )
                        continue;
                    if ( s.type == SurfType::Plane )
                    {
                        if ( (int)reg->size() < minPlane )
                            continue;
                    }
                    else if ( !s.known && d.vertsOf( *reg ).size() < cMinVerts )
                        continue;
                    Found fnd;
                    for ( int f : *reg )
                        fnd.area += d.areas[f];
                    const double q = d.tol / 20;
                    fnd.dev = std::round( d.centersDeviation( s, *reg ) / q ) * q;
                    fnd.surf = s;
                    fnd.faces = std::move( *reg );
                    bool take = !best || fnd.area > best->area * 1.0001 || ( fnd.area > best->area * 0.9999 && better_( fnd, *best ) );
                    if ( best && fnd.surf.type != best->surf.type && std::min( fnd.area, best->area ) > 0.97 * std::max( fnd.area, best->area ) )
                    {
                        // a more complex surface can take a few more small triangles, but it must not win
                        // if it passes much farther from the facets, e.g. a huge torus instead of a cylinder
                        const bool simpler = fnd.surf.type < best->surf.type;
                        const Found& lo = simpler ? fnd : *best;
                        const Found& hi = simpler ? *best : fnd;
                        if ( 3 * lo.dev + 0.5 * d.tol < hi.dev )
                            take = simpler;
                    }
                    if ( take )
                        best = std::move( fnd );
                }
                if ( best && best->surf.type != SurfType::Plane && !best->surf.known )
                {
                    if ( auto q = simpler_( best->surf, best->faces ) )
                    {
                        auto reg = grow_( *q, seed );
                        if ( reg && reg->size() >= best->faces.size() )
                        {
                            best->area = 0;
                            for ( int f : *reg )
                                best->area += d.areas[f];
                            best->dev = 0;
                            best->surf = *q;
                            best->faces = std::move( *reg );
                        }
                    }
                }
                if ( best )
                {
                    for ( int f : best->faces )
                        coveredStamp_[f] = coveredStamp;
                    found.push_back( std::move( *best ) );
                }
                else if ( full )
                {
                    // do not start from the neighbours of the failed seed: most probably they are on the same freeform surface
                    for ( int f : patch_( seed, 3 ) )
                        ++fail_[f];
                }
            }
            if ( !found.empty() )
                break;
        }
        if ( found.empty() && anyTried )
            continue;
        if ( found.empty() )
        {
            // only degenerate triangles remain
            for ( int f : faces )
                addRegion( fitPlane( d.pointsOf( { f } ) ), { f } );
            break;
        }
        std::stable_sort( found.begin(), found.end(), []( const Found& x, const Found& y ) { return x.area > y.area; } );
        for ( const auto& fnd : found )
            if ( std::all_of( fnd.faces.begin(), fnd.faces.end(), [&]( int f ) { return avail_[f]; } ) )
                addRegion( fnd.surf, fnd.faces );
        erase_if( faces, [&]( int f ) { return !avail_[f]; } );
    }
}

bool Grower::run()
{
    MR_TIMER;
    auto& d = d_;
    Vector<Vector<int>> segFaces( d.numSegs );
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) )
            segFaces[d.seg[f]].push_back( f );
    for ( pass_ = 1; pass_ <= 2; ++pass_ )
    {
        if ( pass_ == 2 )
        {
            // keep well supported regions, redo the rest with the knowledge of found surfaces
            Vector<int> map( d.regions.size(), -1 );
            Vector<Region> kept;
            for ( size_t i = 0; i < d.regions.size(); ++i )
                if ( strong_( d.regions[i] ) )
                {
                    map[i] = (int)kept.size();
                    kept.push_back( d.regions[i] );
                }
            for ( int f = 0; f < d.nf; ++f )
                if ( d.valid( f ) )
                    d.sub[f] = map[d.sub[f]];
            d.regions = std::move( kept );
            std::fill( fail_.begin(), fail_.end(), 0 );
        }
        size_t doneFaces = 0;
        size_t numFaces = 0;
        for ( const auto& fs : segFaces )
            numFaces += fs.size();
        for ( int s = 0; s < d.numSegs; ++s )
        {
            processSegment_( s, segFaces[s] );
            doneFaces += segFaces[s].size();
            if ( !reportProgress( cb_, 0.5f * ( float( pass_ - 1 ) + float( doneFaces ) / float( numFaces ) ) ) )
                return false;
        }
        if ( pass_ == 1 )
        {
            for ( const auto& r : d.regions )
            {
                if ( r.surf.type == SurfType::Plane || r.numFaces < cMinStrongFaces )
                    continue;
                if ( std::none_of( known_.begin(), known_.end(), [&]( const Surf& k ) { return sameSurface( k, r.surf, 0.5 * d.tol ); } ) )
                {
                    known_.push_back( r.surf );
                    known_.back().known = true;
                }
                if ( r.surf.type == SurfType::Cyl || r.surf.type == SurfType::Cone )
                    addAxis( lib_, r.surf, d.tol );
            }
        }
    }
    for ( auto& r : d.regions )
        r.surf.known = false;
    return true;
}

} //anonymous namespace

bool findAnalyticRegions( Data& d, const ProgressCallback& cb )
{
    return Grower( d, cb ).run();
}

} //namespace MeshToStep

} //namespace MR
