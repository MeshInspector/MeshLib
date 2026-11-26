#include "MRMeshToStepData.hpp"
#include "MRBitSet.h"
#include "MRId.h"
#include "MRMeshComponents.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"
#include "MRUnionFind.h"

#include <map>
#include <numeric>
#include <set>
#include <utility>

namespace MR
{

namespace MeshToStep
{

namespace
{

/// improves the regions found by Grower, and replaces groups of unreliable regions with B-spline surfaces
class Cleaner
{
public:
    Cleaner( Data& d ) : d_( d ) {}
    void run();

private:
    using Pair = std::pair<int, int>;
    struct Border
    {
        double len = 0, angLen = 0;
        double angle() const { return len > 0 ? angLen / len : 0; }
    };

    bool strong_( const Region& r ) const;
    std::map<Pair, Border> borders_() const;
    void moveFaces_( int from, int to );
    void extendStrong_();
    std::optional<Surf> unionFit_( const std::vector<int>& fa, const std::vector<int>& fb, const Region& ra, const Region& rb ) const;
    void uniteUnderOneSurface_();
    void absorbLyingOnNeighbour_();
    void mergeSameSurface_();
    void fitFreeform_();
    void mergeZeroArea_();
    void splitOnComponents_();

    Data& d_;
    std::vector<Axis> axes_;
};

bool Cleaner::strong_( const Region& r ) const
{
    if ( r.numFaces == 0 )
        return false;
    if ( r.surf.type == SurfType::Plane )
        return r.numFaces >= 3;
    if ( r.surf.type == SurfType::BSpline )
        return true;
    if ( r.numFaces < 20 && r.area < 2e-4 * d_.total )
        return false;
    if ( r.surf.type != SurfType::Torus )
        return true;
    return isCoaxial( r.surf, axes_, d_.tol ) || ( r.surf.r <= 0.5 * r.surf.R && r.numFaces >= 100 );
}

/// common borders of all pairs of adjacent regions from the same segment
std::map<Cleaner::Pair, Cleaner::Border> Cleaner::borders_() const
{
    const auto& d = d_;
    std::map<Pair, Border> res;
    for ( int f = 0; f < d.nf; ++f )
    {
        if ( !d.valid( f ) )
            continue;
        for ( int i = 0; i < 3; ++i )
        {
            const int g = d.nbrSameSeg( f, i );
            if ( g < f || d.sub[f] == d.sub[g] )
                continue;
            auto& b = res[std::minmax( d.sub[f], d.sub[g] )];
            const double len = ( d.pts[d.tris[f][i]] - d.pts[d.tris[f][( i + 1 ) % 3]] ).length();
            b.len += len;
            b.angLen += len * std::acos( std::clamp( dot( d.normals[f], d.normals[g] ), -1.0, 1.0 ) ) * 180 / PI;
        }
    }
    return res;
}

void Cleaner::moveFaces_( int from, int to )
{
    for ( int f = 0; f < d_.nf; ++f )
        if ( d_.valid( f ) && d_.sub[f] == from )
            d_.sub[f] = to;
}

/// strong regions take over adjacent weak triangles that lie on their surface (looser distance, strict normal)
void Cleaner::extendStrong_()
{
    auto& d = d_;
    std::vector<char> isStrong( d.regions.size() );
    for ( size_t i = 0; i < d.regions.size(); ++i )
        isStrong[i] = strong_( d.regions[i] );
    std::vector<int> queue;
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) && isStrong[d.sub[f]] )
            queue.push_back( f );
    while ( !queue.empty() )
    {
        const int f = queue.back();
        queue.pop_back();
        const int r = d.sub[f];
        const Surf& s = d.regions[r].surf;
        for ( int i = 0; i < 3; ++i )
        {
            const int g = d.nbrSameSeg( f, i );
            if ( g < 0 || d.sub[g] == r || isStrong[d.sub[g]] )
                continue;
            double maxDist = 0, maxEdge = 0;
            for ( int k = 0; k < 3; ++k )
            {
                maxDist = std::max( maxDist, std::abs( s.dist( d.pts[d.tris[g][k]] ) ) );
                maxEdge = std::max( maxEdge, ( d.pts[d.tris[g][k]] - d.pts[d.tris[g][( k + 1 ) % 3]] ).length() );
            }
            if ( maxDist > 2.5 * d.tol )
                continue;
            const bool sliver = !( maxEdge > 0 ) || 2 * d.areas[g] / maxEdge < 0.5 * d.tol;
            if ( !sliver && std::abs( dot( s.normal( d.centers[g] ), d.normals[g] ) ) < cCos1 )
                continue;
            d.sub[g] = r;
            queue.push_back( g );
        }
    }
}

/// one surface passing through all vertices of two regions, if any
std::optional<Surf> Cleaner::unionFit_( const std::vector<int>& fa, const std::vector<int>& fb, const Region& ra, const Region& rb ) const
{
    const auto& d = d_;
    auto fs = fa;
    fs.insert( fs.end(), fb.begin(), fb.end() );
    const auto pts = d.pointsOf( fs );
    const auto sub = d.pointsOf( fs, 800 );
    std::vector<V3> ns, cs;
    double area = 0;
    for ( int f : fs )
    {
        ns.push_back( d.normals[f] );
        cs.push_back( d.centers[f] );
        area += d.areas[f];
    }
    const double own = ( d.centersDeviation( ra.surf, fa ) * ra.area + d.centersDeviation( rb.surf, fb ) * rb.area ) / area;
    auto accept = [&]( const Surf& q )
    {
        if ( !q.finite() || q.maxDist( pts ) > d.tol )
            return false;
        for ( int f : fs )
            if ( !d.normalsAgree( q, f, true ) )
                return false;
        return d.centersDeviation( q, fs ) <= 2 * own + 1e-4 * d.diag;
    };
    auto tryRefined = [&]( std::optional<Surf> q ) -> std::optional<Surf>
    {
        if ( !q )
            return {};
        q = refine( *q, sub );
        return accept( *q ) ? q : std::nullopt;
    };
    if ( auto q = tryRefined( fitCyl( pts, ns ) ) )
        return q;
    if ( auto q = tryRefined( fitConeByNormals( pts, cs, ns ) ) )
        return q;
    if ( auto q = tryRefined( fitRevolution( pts, cs, ns, SurfType::Cone ) ) )
        return q;
    if ( auto q = tryRefined( fitSphere( pts ) ) )
        return q;
    for ( const auto& ax : axes_ )
        if ( auto q = axisTorus( pts, ax.dir, ax.pos ); q && accept( *q ) )
            return q;
    return tryRefined( fitRevolution( pts, cs, ns, SurfType::Torus ) );
}

/// two smoothly adjacent curved regions that one surface explains are one region,
/// e.g. sectors of a torus that were accepted as short cylinders
void Cleaner::uniteUnderOneSurface_()
{
    auto& d = d_;
    std::set<std::array<int, 4>> tested;
    for ( ;; )
    {
        d.updateStats();
        const auto borders = borders_();
        const auto faces = d.facesOfRegions();
        std::vector<Pair> pairs;
        for ( const auto& [p, b] : borders )
        {
            const auto& ra = d.regions[p.first];
            const auto& rb = d.regions[p.second];
            if ( b.angle() > 15 || ra.surf.type == SurfType::Plane || rb.surf.type == SurfType::Plane || ra.numFaces < 6 || rb.numFaces < 6 )
                continue;
            if ( tested.count( { p.first, p.second, ra.numFaces, rb.numFaces } ) )
                continue;
            pairs.push_back( p );
        }
        std::stable_sort( pairs.begin(), pairs.end(), [&]( const Pair& x, const Pair& y )
        {
            return d.regions[x.first].area + d.regions[x.second].area > d.regions[y.first].area + d.regions[y.second].area;
        } );
        bool done = false;
        for ( const auto& p : pairs )
        {
            auto& ra = d.regions[p.first];
            const auto& rb = d.regions[p.second];
            tested.insert( { p.first, p.second, ra.numFaces, rb.numFaces } );
            if ( auto q = unionFit_( faces[p.first], faces[p.second], ra, rb ) )
            {
                ra.surf = *q;
                moveFaces_( p.second, p.first );
                done = true;
                break;
            }
        }
        if ( !done )
            break;
    }
}

/// a region lying entirely on the surface of a larger adjacent region belongs to it
void Cleaner::absorbLyingOnNeighbour_()
{
    auto& d = d_;
    for ( ;; )
    {
        d.updateStats();
        const auto faces = d.facesOfRegions();
        std::vector<Pair> pairs;
        for ( const auto& [p, b] : borders_() )
        {
            pairs.push_back( p );
            pairs.push_back( { p.second, p.first } );
        }
        std::stable_sort( pairs.begin(), pairs.end(), [&]( const Pair& x, const Pair& y ) { return d.regions[x.first].area > d.regions[y.first].area; } );
        bool done = false;
        for ( const auto& [a, b] : pairs )
        {
            const auto& ra = d.regions[a];
            if ( ra.area <= d.regions[b].area || !strong_( ra ) )
                continue;
            bool ok = true;
            for ( int f : faces[b] )
            {
                for ( int v : d.tris[f] )
                    ok = ok && std::abs( ra.surf.dist( d.pts[v] ) ) <= d.tol;
                ok = ok && std::abs( dot( ra.surf.normal( d.centers[f] ), d.normals[f] ) ) >= cCos40;
                if ( !ok )
                    break;
            }
            if ( !ok )
                continue;
            moveFaces_( b, a );
            done = true;
            break;
        }
        if ( !done )
            break;
    }
}

/// adjacent regions on the same surface are one region
void Cleaner::mergeSameSurface_()
{
    auto& d = d_;
    d.updateStats();
    UnionFind<RegionId> uf( d.regions.size() );
    // the surface of the largest region is kept for every united set
    std::vector<int> largest( d.regions.size() );
    std::iota( largest.begin(), largest.end(), 0 );
    for ( const auto& [p, b] : borders_() )
    {
        const RegionId ra = uf.find( RegionId( p.first ) ), rb = uf.find( RegionId( p.second ) );
        if ( ra == rb || !sameSurface( d.regions[p.first].surf, d.regions[p.second].surf, 0.5 * d.tol ) )
            continue;
        const int la = largest[ra], lb = largest[rb];
        largest[uf.unite( ra, rb ).first] = d.regions[la].area < d.regions[lb].area ? lb : la;
    }
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) )
            d.sub[f] = largest[uf.find( RegionId( d.sub[f] ) )];
}

/// groups of smoothly adjacent weak regions are approximated by B-spline surfaces
void Cleaner::fitFreeform_()
{
    MR_TIMER;
    auto& d = d_;
    d.updateStats();
    const int nr = (int)d.regions.size();
    std::vector<char> weak( nr );
    for ( int i = 0; i < nr; ++i )
        weak[i] = d.regions[i].numFaces > 0 && !strong_( d.regions[i] ) && d.regions[i].surf.type != SurfType::Plane; // one or two triangles can be a true plane
    const auto borders = borders_();
    std::vector<std::vector<int>> smoothNbr( nr );
    for ( const auto& [p, b] : borders )
        if ( b.angle() < 30 )
        {
            smoothNbr[p.first].push_back( p.second );
            smoothNbr[p.second].push_back( p.first );
        }
    std::vector<int> cluster;
    int numClusters = 0;
    auto findClusters = [&]
    {
        cluster.assign( nr, -1 );
        numClusters = 0;
        std::vector<int> stack;
        for ( int r0 = 0; r0 < nr; ++r0 )
        {
            if ( !weak[r0] || cluster[r0] >= 0 )
                continue;
            cluster[r0] = numClusters;
            stack.push_back( r0 );
            while ( !stack.empty() )
            {
                const int x = stack.back();
                stack.pop_back();
                for ( int y : smoothNbr[x] )
                    if ( weak[y] && cluster[y] < 0 )
                    {
                        cluster[y] = numClusters;
                        stack.push_back( y );
                    }
            }
            ++numClusters;
        }
    };
    findClusters();
    std::vector<double> clusterArea( numClusters, 0.0 ), lim( nr, 0.0 ), border( nr, 0.0 );
    for ( int r = 0; r < nr; ++r )
        if ( weak[r] )
            clusterArea[cluster[r]] += d.regions[r].area;
    for ( int r = 0; r < nr; ++r )
        if ( weak[r] )
            lim[r] = 0.03 * clusterArea[cluster[r]];
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) )
            for ( int i = 0; i < 3; ++i )
                if ( const int g = d.nbr[f][i]; g < 0 || d.sub[g] != d.sub[f] )
                    border[d.sub[f]] += ( d.pts[d.tris[f][i]] - d.pts[d.tris[f][( i + 1 ) % 3]] ).length();

    // accidental "strong" regions lying smoothly inside a cluster join it
    for ( ;; )
    {
        std::vector<std::pair<int, double>> add;
        for ( int y = 0; y < nr; ++y )
        {
            const auto& ry = d.regions[y];
            if ( weak[y] || ry.numFaces == 0 || ry.numFaces >= 120 || ry.surf.type == SurfType::Sphere || ry.surf.type == SurfType::BSpline )
                continue;
            double l = 0, shared = 0;
            bool any = false;
            for ( int r : smoothNbr[y] )
                if ( weak[r] )
                {
                    any = true;
                    l = std::max( l, lim[r] );
                    shared += borders.at( std::minmax( y, r ) ).len;
                }
            if ( any && ( ry.area < l || ( ry.area < 15 * l && ( shared > 0.4 * border[y] || !strong_( ry ) ) ) ) )
                add.push_back( { y, l } );
        }
        if ( add.empty() )
            break;
        for ( const auto& [y, l] : add )
        {
            weak[y] = 1;
            lim[y] = l;
        }
    }
    // a sphere with several weak pieces left around it only approximates a freeform surface, e.g. a lemon-shaped surface of revolution
    for ( int y = 0; y < nr; ++y )
    {
        if ( weak[y] || d.regions[y].numFaces == 0 || d.regions[y].surf.type != SurfType::Sphere )
            continue;
        int numWeak = 0;
        for ( int r : smoothNbr[y] )
            if ( weak[r] && d.regions[r].surf.type == SurfType::Sphere )
                ++numWeak;
        if ( numWeak >= 4 )
            weak[y] = 1;
    }
    findClusters();

    std::vector<std::vector<int>> clusterFaces( numClusters );
    for ( int f = 0; f < d.nf; ++f )
        if ( d.valid( f ) && cluster[d.sub[f]] >= 0 )
            clusterFaces[cluster[d.sub[f]]].push_back( f );
    for ( const auto& fs : clusterFaces )
    {
        if ( fs.size() < 30 )
            continue;
        FaceBitSet region( d.nf );
        for ( int f : fs )
            region.set( FaceId( f ) );
        // only a disk can be parametrized: one boundary loop not touching itself in a vertex
        const auto loops = findLeftBoundary( d.mesh->topology, &region );
        if ( loops.size() != 1 )
            continue;
        std::vector<VertId> loopVerts;
        for ( EdgeId e : loops[0] )
            loopVerts.push_back( d.mesh->topology.org( e ) );
        std::sort( loopVerts.begin(), loopVerts.end() );
        if ( std::adjacent_find( loopVerts.begin(), loopVerts.end() ) != loopVerts.end() )
            continue;
        const auto verts = d.vertsOf( fs );
        std::vector<V3> pts;
        pts.reserve( verts.size() );
        for ( int v : verts )
            pts.push_back( d.pts[v] );
        std::vector<Tri> tris;
        tris.reserve( fs.size() );
        for ( int f : fs )
        {
            Tri t;
            for ( int i = 0; i < 3; ++i )
                t[i] = int( std::lower_bound( verts.begin(), verts.end(), d.tris[f][i] ) - verts.begin() );
            tris.push_back( t );
        }
        auto bs = fitBSpline( pts, tris, 0.5 * d.tol );
        if ( !bs )
            continue;
        Region r;
        r.surf.type = SurfType::BSpline;
        r.surf.bs = bs;
        r.seg = d.seg[fs[0]];
        for ( int f : fs )
            d.sub[f] = (int)d.regions.size();
        d.regions.push_back( std::move( r ) );
    }
}

/// regions of (nearly) zero area carry no geometry: give them to the neighbour with the longest common border
void Cleaner::mergeZeroArea_()
{
    auto& d = d_;
    for ( ;; )
    {
        d.updateStats();
        const auto borders = borders_();
        bool done = false;
        for ( int r = 0; r < (int)d.regions.size(); ++r )
        {
            if ( d.regions[r].numFaces == 0 || d.regions[r].area >= 1e-8 * d.total )
                continue;
            int best = -1;
            double bestLen = -1;
            for ( const auto& [p, b] : borders )
                if ( ( p.first == r || p.second == r ) && b.len > bestLen )
                {
                    bestLen = b.len;
                    best = p.first == r ? p.second : p.first;
                }
            if ( best >= 0 )
            {
                moveFaces_( r, best );
                done = true;
                break;
            }
        }
        if ( !done )
            break;
    }
}

/// every region becomes one connected piece, and the regions are renumbered without gaps
void Cleaner::splitOnComponents_()
{
    auto& d = d_;
    const auto& topology = d.mesh->topology;
    UndirectedEdgeBitSet borders( topology.undirectedEdgeSize() );
    for ( UndirectedEdgeId ue{ 0 }; ue < borders.size(); ++ue )
    {
        const FaceId l = topology.left( ue ), r = topology.right( ue );
        if ( l && r && d.sub[l] != d.sub[r] )
            borders.set( ue );
    }
    const auto [compMap, numComps] = MeshComponents::getAllComponentsMap( *d.mesh, MeshComponents::PerEdge, &borders );
    std::vector<Region> regions( numComps );
    for ( FaceId f : topology.getValidFaces() )
    {
        const int c = compMap[f];
        regions[c] = d.regions[d.sub[f]];
        d.sub[f] = c;
    }
    d.regions = std::move( regions );
    d.updateStats();
}

void Cleaner::run()
{
    MR_TIMER;
    auto& d = d_;
    d.updateStats();
    for ( const auto& r : d.regions )
        if ( ( r.surf.type == SurfType::Cyl || r.surf.type == SurfType::Cone ) && r.numFaces >= 20 && r.area >= 2e-4 * d.total )
            addAxis( axes_, r.surf, d.tol );
    extendStrong_();
    uniteUnderOneSurface_();
    absorbLyingOnNeighbour_();
    mergeSameSurface_();
    fitFreeform_();
    mergeZeroArea_();
    splitOnComponents_();
}

} //anonymous namespace

void cleanRegions( Data& d )
{
    Cleaner( d ).run();
}

} //namespace MeshToStep

} //namespace MR
