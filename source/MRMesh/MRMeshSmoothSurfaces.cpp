#include "MRMeshSmoothSurfaces.h"
#include "MRSmoothSurfaceFit.hpp"
#include "MRBitSet.h"
#include "MRBox.h"
#include "MRLine3.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"

#include <array>

namespace MR
{

namespace
{

using namespace SmoothSurfaceFit;

/// the mesh in the form convenient for the fitting
struct Data
{
    const Mesh* mesh = nullptr;
    Vector<V3, VertId> pts;
    Vector<V3, FaceId> normals, centers;
    Vector<double, FaceId> areas;
    double diag = 0;
    double bsplineThreshold = 0;

    Vector<V3> pointsOf( const Vector<FaceId>& faces ) const
    {
        Vector<VertId> verts;
        verts.reserve( faces.size() * 3 );
        for ( FaceId f : faces )
            for ( VertId v : mesh->topology.getTriVerts( f ) )
                verts.push_back( v );
        std::sort( verts.begin(), verts.end() );
        verts.erase( std::unique( verts.begin(), verts.end() ), verts.end() );
        Vector<V3> res;
        res.reserve( verts.size() );
        for ( VertId v : verts )
            res.push_back( pts[v] );
        return res;
    }

    FaceBitSet bitSet( const Vector<FaceId>& faces ) const
    {
        FaceBitSet res( mesh->topology.faceSize() );
        for ( FaceId f : faces )
            res.set( f );
        return res;
    }
};

/// a surface with the distances from the vertices of its region
struct Fit
{
    Surf surf;
    double rms = 0;
    double maxDist = 0;
};

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
void measure( Fit& r, const Vector<V3>& pts )
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

/// the normals of the triangles must mostly agree with the normals of a curved surface, otherwise it is fitted to a shape of different kind,
/// e.g. a thin cylinder to a small bump
bool normalsAgree( const Data& d, const Vector<FaceId>& faces, const Surf& s )
{
    static const double cCos45 = std::cos( PI / 4 );
    double bad = 0, total = 0;
    for ( FaceId f : faces )
    {
        total += d.areas[f];
        if ( std::abs( dot( s.normal( d.centers[f] ), d.normals[f] ) ) < cCos45 )
            bad += d.areas[f];
    }
    return bad <= 0.2 * total;
}

/// a region on a surface of revolution with exactly one boundary loop going around the axis would be unbounded on one side,
/// e.g. a small bump on a plane, fitted by a cylinder
bool boundedOnRevolution( const Data& d, const Vector<FaceId>& faces, const Vector<V3>& pts, const Surf& s )
{
    if ( s.type != SurfType::Cyl && s.type != SurfType::Cone && s.type != SurfType::Torus )
        return true;
    const auto region = d.bitSet( faces );
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
        for ( const auto& p : pts )
            if ( ( p - s.c ).length() < 5e-6 * d.diag )
                return true;
    return false;
}

/// the simplest of plane, cylinder, cone, sphere and torus, which fits the faces not much worse than any more complex of them
Fit fitAnalytic( const Data& d, const Vector<FaceId>& faces )
{
    const auto pts = d.pointsOf( faces );
    const auto sub = subsample( pts, 2000 );
    Vector<V3> ns, cs;
    for ( FaceId f : subsample( faces, 2000 ) )
    {
        ns.push_back( d.normals[f] );
        cs.push_back( d.centers[f] );
    }

    // the best surface of every type
    constexpr int cNumTypes = int( SurfType::Torus ) + 1;
    std::array<std::optional<Fit>, cNumTypes> best;
    auto consider = [&]( std::optional<Surf> s, bool refined )
    {
        if ( !s )
            return;
        if ( refined )
            s = refine( *s, sub );
        if ( !s->finite() || ( s->type != SurfType::Plane && !normalsAgree( d, faces, *s ) ) )
            return;
        Fit r{ .surf = *s };
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
    std::optional<Fit> res;
    for ( int t = cNumTypes - 1; t >= 0; --t )
        if ( best[t] && boundedOnRevolution( d, faces, pts, best[t]->surf ) && ( !res || best[t]->rms <= std::max( 1.5 * res->rms, 1e-6 * d.diag ) ) )
            res = best[t];
    return *res; // the plane always exists and is bounded
}

/// B-spline surface fitted to the faces, if they have one boundary loop not touching itself
std::optional<Fit> fitBSpline( const Data& d, const Vector<FaceId>& faces )
{
    const auto region = d.bitSet( faces );
    const auto loops = findLeftBoundary( d.mesh->topology, &region );
    if ( loops.size() != 1 )
        return {};
    Vector<VertId> loopVerts;
    for ( EdgeId e : loops[0] )
        loopVerts.push_back( d.mesh->topology.org( e ) );
    std::sort( loopVerts.begin(), loopVerts.end() );
    if ( std::adjacent_find( loopVerts.begin(), loopVerts.end() ) != loopVerts.end() )
        return {};
    auto bs = fitBSplineSurfaceToMesh( { *d.mesh, &region }, { .tolerance = d.bsplineThreshold } );
    if ( !bs )
        return {};
    Fit res;
    res.surf.type = SurfType::BSpline;
    res.surf.bs = std::make_shared<FittedBSplineSurface>( std::move( *bs ) );
    measure( res, d.pointsOf( faces ) );
    return res;
}

/// the surface in the public form
SmoothSurface toSmoothSurface( const Surf& s, const Vector<V3>& pts )
{
    switch ( s.type )
    {
    case SurfType::Plane:
        return Plane3d::fromDirAndPt( s.a, s.c );
    case SurfType::Cyl:
    {
        MinMaxd range;
        for ( const auto& p : pts )
            range.include( dot( p - s.c, s.a ) );
        return Cylinder3d( s.c + range.center() * s.a, s.a, s.r, range.size() );
    }
    case SurfType::Cone:
    {
        double height = 0;
        for ( const auto& p : pts )
            height = std::max( height, dot( p - s.c, s.a ) );
        return Cone3d( Line3d( s.c, s.a ), s.r, height );
    }
    case SurfType::Sphere:
        return Sphere3d( s.c, s.r );
    case SurfType::Torus:
        return Torus3d( s.c, s.a, s.R, s.r );
    case SurfType::BSpline:
        return s.bs;
    }
    assert( false );
    return Plane3d();
}

} //anonymous namespace

Expected<MeshSmoothSurfaces> fitSmoothSurfaces( const Mesh& mesh, const Face2RegionMap& face2region,
    const FitSmoothSurfacesSettings& settings, const ProgressCallback& progress )
{
    MR_TIMER;
    const auto& topology = mesh.topology;
    if ( topology.numValidFaces() <= 0 )
        return unexpected( "Mesh is empty" );

    Data d;
    d.mesh = &mesh;
    d.pts.resize( mesh.points.size() );
    for ( VertId v = 0_v; v < d.pts.endId(); ++v )
        d.pts[v] = V3( mesh.points[v] );
    d.normals.resize( topology.faceSize() );
    d.centers.resize( topology.faceSize() );
    d.areas.resize( topology.faceSize(), 0.0 );
    Box3d box;
    Vector<Vector<FaceId>, RegionId> faces;
    for ( FaceId f : topology.getValidFaces() )
    {
        const auto [a, b, c] = topology.getTriVerts( f );
        const V3 n = cross( d.pts[b] - d.pts[a], d.pts[c] - d.pts[a] );
        const double len = n.length();
        d.areas[f] = len / 2;
        d.normals[f] = len > 0 ? n / len : V3();
        d.centers[f] = ( d.pts[a] + d.pts[b] + d.pts[c] ) / 3.0;
        for ( VertId v : { a, b, c } )
            box.include( d.pts[v] );
        const RegionId r = face2region[f];
        if ( r >= faces.size() )
            faces.resize( r + 1 );
        faces[r].push_back( f );
    }
    d.diag = box.valid() ? box.diagonal() : 0;
    if ( !( d.diag > 0 ) )
        return unexpected( "Mesh is degenerate" );

    Vector<Fit, RegionId> fits( faces.size() );
    if ( !ParallelFor( fits, [&]( RegionId r )
    {
        if ( !faces[r].empty() )
            fits[r] = fitAnalytic( d, faces[r] );
    }, subprogress( progress, 0.0f, 0.5f ) ) )
        return unexpectedOperationCanceled();

    d.bsplineThreshold = settings.bsplineThreshold;
    if ( !( d.bsplineThreshold > 0 ) )
    {
        // the noise of the mesh: most regions are expected to have analytic surfaces
        std::vector<double> rms;
        for ( RegionId r( 0 ); r < fits.endId(); ++r )
            if ( !faces[r].empty() )
                rms.push_back( fits[r].rms );
        std::nth_element( rms.begin(), rms.begin() + rms.size() / 2, rms.end() );
        d.bsplineThreshold = std::max( 2.5 * rms[rms.size() / 2], 1e-6 * d.diag );
    }

    if ( settings.allowBSplines && !ParallelFor( fits, [&]( RegionId r )
    {
        if ( fits[r].rms > d.bsplineThreshold )
            if ( auto bs = fitBSpline( d, faces[r] ); bs && 1.5 * bs->rms < fits[r].rms )
                fits[r] = std::move( *bs );
    }, subprogress( progress, 0.5f, 1.0f ) ) )
        return unexpectedOperationCanceled();

    MeshSmoothSurfaces res;
    res.bsplineThreshold = float( d.bsplineThreshold );
    res.surfaces.resize( faces.size(), { .surface = Plane3d() } );
    for ( RegionId r( 0 ); r < fits.endId(); ++r )
    {
        if ( faces[r].empty() )
            continue;
        res.surfaces[r] = {
            .surface = toSmoothSurface( fits[r].surf, d.pointsOf( faces[r] ) ),
            .rms = float( fits[r].rms ),
            .maxDist = float( fits[r].maxDist )
        };
    }
    reportProgress( progress, 1.0f );
    return res;
}

} //namespace MR
