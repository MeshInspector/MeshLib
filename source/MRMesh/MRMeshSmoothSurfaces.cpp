#include "MRMeshSmoothSurfaces.h"
#include "MRBestFit.h"
#include "MRBitSet.h"
#include "MRBox.h"
#include "MRFittedBSplineSurface.h"
#include "MRLine3.h"
#include "MRMatrix3.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"

#include <array>
#include <cmath>
#include <optional>

namespace MR
{

namespace
{

/// unit normal of the B-spline surface, or (0,0,1) where it is not defined
Vector3d bsplineNormal( const FittedBSplineSurface& bs, const Vector2d& uv )
{
    const Vector3d n = bs.surface.normal( uv );
    return n == Vector3d() ? Vector3d( 0, 0, 1 ) : n;
}

/// the radial vector from the axis to the point, and the position of the point along the axis
std::pair<Vector3d, double> radialAndHeight( const Line3d& axis, const Vector3d& p )
{
    const Vector3d d = p - axis.p;
    const double h = dot( d, axis.d );
    return { d - h * axis.d, h };
}

} //anonymous namespace

double signedDistance( const SmoothSurface& s, const Vector3d& p )
{
    if ( const auto* plane = std::get_if<Plane3d>( &s ) )
        return plane->distance( p );
    if ( const auto* cyl = std::get_if<Cylinder3d>( &s ) )
        return radialAndHeight( cyl->mainAxis, p ).first.length() - cyl->radius;
    if ( const auto* cone = std::get_if<Cone3d>( &s ) )
    {
        const auto [rv, h] = radialAndHeight( cone->axis, p );
        return ( rv.length() - h * std::tan( cone->angle ) ) * std::cos( cone->angle );
    }
    if ( const auto* sphere = std::get_if<Sphere3d>( &s ) )
        return sphere->distance( p );
    if ( const auto* torus = std::get_if<Torus3d>( &s ) )
        return torus->distance( p );
    const auto& bs = std::get<std::shared_ptr<const FittedBSplineSurface>>( s );
    assert( bs );
    const auto uv = bs->project( p );
    return dot( p - bs->surface.eval( uv ), bsplineNormal( *bs, uv ) );
}

Vector3d surfaceNormal( const SmoothSurface& s, const Vector3d& p )
{
    if ( const auto* plane = std::get_if<Plane3d>( &s ) )
        return plane->n;
    if ( const auto* sphere = std::get_if<Sphere3d>( &s ) )
    {
        const Vector3d d = p - sphere->center;
        return d / std::max( d.length(), 1e-12 );
    }
    if ( const auto* bs = std::get_if<std::shared_ptr<const FittedBSplineSurface>>( &s ) )
    {
        assert( *bs );
        return bsplineNormal( **bs, ( *bs )->project( p ) );
    }
    const auto* cyl = std::get_if<Cylinder3d>( &s );
    const auto* cone = std::get_if<Cone3d>( &s );
    const auto* torus = std::get_if<Torus3d>( &s );
    const Line3d axis = cyl ? cyl->mainAxis : cone ? cone->axis : Line3d( torus->center, torus->axis );
    const auto [rv, h] = radialAndHeight( axis, p );
    const double rho = rv.length();
    const Vector3d er = rv / std::max( rho, 1e-12 );
    if ( cyl )
        return er;
    if ( cone )
        return er * std::cos( cone->angle ) - axis.d * std::sin( cone->angle );
    const Vector3d n = er * ( rho - torus->majorRadius ) + h * axis.d;
    return n / std::max( n.length(), 1e-12 );
}

namespace
{

/// the mesh in the form convenient for the fitting
struct Data
{
    const Mesh* mesh = nullptr;
    Vector<Vector3d, VertId> pts;
    Vector<Vector3d, FaceId> normals, centers;
    Vector<double, FaceId> areas;
    double diag = 0;
    double bsplineThreshold = 0;

    Vector<Vector3d> pointsOf( const Vector<FaceId>& faces ) const
    {
        Vector<VertId> verts;
        verts.reserve( faces.size() * 3 );
        for ( FaceId f : faces )
            for ( VertId v : mesh->topology.getTriVerts( f ) )
                verts.push_back( v );
        std::sort( verts.begin(), verts.end() );
        verts.erase( std::unique( verts.begin(), verts.end() ), verts.end() );
        Vector<Vector3d> res;
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
    SmoothSurface surf;
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
void measure( Fit& r, const Vector<Vector3d>& pts )
{
    double sum = 0, mx = 0;
    for ( const auto& p : pts )
    {
        const double dist = std::abs( signedDistance( r.surf, p ) );
        sum += sqr( dist );
        mx = std::max( mx, dist );
    }
    r.rms = pts.empty() ? 0 : std::sqrt( sum / double( pts.size() ) );
    r.maxDist = mx;
}

/// the normals of the triangles must mostly agree with the normals of a curved surface, otherwise it is fitted to a shape of different kind,
/// e.g. a thin cylinder to a small bump
bool normalsAgree( const Data& d, const Vector<FaceId>& faces, const SmoothSurface& s )
{
    static const double cCos45 = std::cos( PI / 4 );
    double bad = 0, total = 0;
    for ( FaceId f : faces )
    {
        total += d.areas[f];
        if ( std::abs( dot( surfaceNormal( s, d.centers[f] ), d.normals[f] ) ) < cCos45 )
            bad += d.areas[f];
    }
    return bad <= 0.2 * total;
}

/// the axis of cylinder, cone or torus
std::optional<Line3d> revolutionAxis( const SmoothSurface& s )
{
    if ( const auto* cyl = std::get_if<Cylinder3d>( &s ) )
        return cyl->mainAxis;
    if ( const auto* cone = std::get_if<Cone3d>( &s ) )
        return cone->axis;
    if ( const auto* torus = std::get_if<Torus3d>( &s ) )
        return Line3d( torus->center, torus->axis );
    return {};
}

/// a region on a surface of revolution with exactly one boundary loop going around the axis would be unbounded on one side,
/// e.g. a small bump on a plane, fitted by a cylinder
bool boundedOnRevolution( const Data& d, const Vector<FaceId>& faces, const Vector<Vector3d>& pts, const SmoothSurface& s )
{
    const auto axis = revolutionAxis( s );
    if ( !axis )
        return true;
    const auto region = d.bitSet( faces );
    const Vector3d x = axis->d.perpendicular().first, y = cross( axis->d, x );
    int around = 0;
    for ( const auto& loop : findLeftBoundary( d.mesh->topology, &region ) )
    {
        double total = 0, prev = 0;
        for ( size_t i = 0; i <= loop.size(); ++i )
        {
            const Vector3d q = d.pts[d.mesh->topology.org( loop[i % loop.size()] )] - axis->p;
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
    if ( std::holds_alternative<Cone3d>( s ) )
        for ( const auto& p : pts )
            if ( ( p - axis->p ).length() < 5e-6 * d.diag )
                return true;
    return false;
}

bool isFinite( const Vector3d& v )
{
    return std::isfinite( v.x + v.y + v.z );
}

bool isFinite( const SmoothSurface& s )
{
    if ( const auto* plane = std::get_if<Plane3d>( &s ) )
        return isFinite( plane->n ) && std::isfinite( plane->d );
    if ( const auto* cyl = std::get_if<Cylinder3d>( &s ) )
        return isFinite( cyl->center() ) && isFinite( cyl->direction() ) && std::isfinite( cyl->radius );
    if ( const auto* cone = std::get_if<Cone3d>( &s ) )
        return isFinite( cone->center() ) && isFinite( cone->direction() ) && std::isfinite( cone->angle );
    if ( const auto* sphere = std::get_if<Sphere3d>( &s ) )
        return isFinite( sphere->center ) && std::isfinite( sphere->radius );
    if ( const auto* torus = std::get_if<Torus3d>( &s ) )
        return isFinite( torus->center ) && isFinite( torus->axis ) && std::isfinite( torus->majorRadius + torus->minorRadius );
    return true;
}

/// the best plane through the points
Plane3d fitPlane( const Vector<Vector3d>& pts )
{
    PointAccumulator acc;
    for ( const auto& p : pts )
        acc.addPoint( p );
    Vector3d c, vals;
    Matrix3d vecs;
    if ( !acc.getCenteredCovarianceEigen( c, vecs, vals ) )
        return Plane3d::fromDirAndPt( Vector3d( 0, 0, 1 ), c );
    return Plane3d::fromDirAndPt( vecs.x, c );
}

/// minimizes the sum of squared distances from the points to the cylinder, the cone or the torus starting from given one
SmoothSurface refine( const SmoothSurface& s, const Vector<Vector3d>& pts )
{
    if ( pts.empty() )
        return s;
    if ( const auto* torus = std::get_if<Torus3d>( &s ) )
    {
        const auto refined = refineTorus( pts, *torus );
        if ( refined == *torus )
            return s;
        // without major radius it is a sphere
        if ( refined.majorRadius < 1e-3 * refined.minorRadius )
            return Sphere3d( refined.center, refined.minorRadius );
        return refined;
    }
    // the refinements of cylinder and cone need at least 7 points, and a surface through fewer points (e.g. a cylinder through
    // two triangles) is found by them too if all the points are repeated: it does not change the closest surface
    Vector<Vector3d> repeated;
    while ( pts.size() < 7 && repeated.size() < 7 )
        for ( const auto& q : pts )
            repeated.push_back( q );
    const auto& ps = repeated.empty() ? pts : repeated;
    SmoothSurface res = s;
    if ( const auto* cyl = std::get_if<Cylinder3d>( &s ) )
        res = refineCylinder( ps, *cyl );
    else if ( const auto* cone = std::get_if<Cone3d>( &s ) )
        res = refineCone( ps, *cone );
    return isFinite( res ) ? res : s;
}

/// the simplest of plane, cylinder, cone, sphere and torus, which fits the faces not much worse than any more complex of them
Fit fitAnalytic( const Data& d, const Vector<FaceId>& faces )
{
    const auto pts = d.pointsOf( faces );
    const auto sub = subsample( pts, 2000 );
    Vector<Vector3d> ns;
    Vector<Plane3d> tangentPlanes;
    Vector<Line3d> normalLines;
    for ( FaceId f : subsample( faces, 2000 ) )
    {
        ns.push_back( d.normals[f] );
        tangentPlanes.push_back( Plane3d::fromDirAndPt( d.normals[f], d.centers[f] ) );
        normalLines.emplace_back( d.centers[f], d.normals[f] );
    }

    // the best surface of every analytic type, in the order of SmoothSurface alternatives
    constexpr int cNumTypes = 5;
    std::array<std::optional<Fit>, cNumTypes> best;
    auto consider = [&]( const auto& fitted, bool refined )
    {
        if ( !fitted )
            return;
        SmoothSurface s = *fitted;
        if ( refined )
            s = refine( s, sub );
        if ( !isFinite( s ) || ( !std::holds_alternative<Plane3d>( s ) && !normalsAgree( d, faces, s ) ) )
            return;
        Fit r{ .surf = std::move( s ) };
        measure( r, pts );
        auto& b = best[r.surf.index()];
        if ( !b || r.rms < b->rms )
            b = std::move( r );
    };
    consider( std::optional<Plane3d>( fitPlane( pts ) ), false );
    if ( pts.size() >= 4 )
    {
        const auto axis = findRevolutionAxis( normalLines );
        consider( findBestCylinderByNormals( sub, ns ), true );
        consider( findBestConeByTangentPlanes( sub, tangentPlanes ), true );
        if ( axis )
            consider( findBestConeWithAxis( sub, *axis ), true );
        consider( findBestSphere( sub ), false );
        if ( axis )
            consider( findBestTorusWithAxis( sub, *axis ), true );
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
    Fit res{ .surf = std::shared_ptr<const FittedBSplineSurface>( std::make_shared<FittedBSplineSurface>( std::move( *bs ) ) ) };
    measure( res, d.pointsOf( faces ) );
    return res;
}

/// the center and the length of the cylinder and the height of the cone span the points
SmoothSurface spanPoints( SmoothSurface s, const Vector<Vector3d>& pts )
{
    if ( auto* cyl = std::get_if<Cylinder3d>( &s ) )
    {
        MinMaxd range;
        for ( const auto& p : pts )
            range.include( dot( p - cyl->center(), cyl->direction() ) );
        cyl->center() += range.center() * cyl->direction();
        cyl->length = range.size();
    }
    else if ( auto* cone = std::get_if<Cone3d>( &s ) )
    {
        cone->height = 0;
        for ( const auto& p : pts )
            cone->height = std::max( cone->height, dot( p - cone->center(), cone->direction() ) );
    }
    return s;
}

/// the public form of the fit
FittedSmoothSurface toFitted( const Data& d, const Vector<FaceId>& faces, const Fit& fit )
{
    return {
        .surface = spanPoints( fit.surf, d.pointsOf( faces ) ),
        .rms = float( fit.rms ),
        .maxDist = float( fit.maxDist )
    };
}

/// replaces the fit with B-spline surface if the fit is farther than the threshold and the B-spline surface is 1.5 times closer
void improveByBSpline( const Data& d, const Vector<FaceId>& faces, Fit& fit )
{
    if ( fit.rms > d.bsplineThreshold )
        if ( auto bs = fitBSpline( d, faces ); bs && 1.5 * bs->rms < fit.rms )
            fit = std::move( *bs );
}

/// the mesh data for the fitting of the faces of the region (all valid faces if region is null);
/// the size of the whole mesh is computed in any case
Expected<Data> makeData( const Mesh& mesh, const FaceBitSet* region )
{
    const auto& topology = mesh.topology;
    if ( topology.numValidFaces() <= 0 )
        return unexpected( "Mesh is empty" );

    Data d;
    d.mesh = &mesh;
    d.pts.resize( mesh.points.size() );
    for ( VertId v = 0_v; v < d.pts.endId(); ++v )
        d.pts[v] = Vector3d( mesh.points[v] );
    d.normals.resize( topology.faceSize() );
    d.centers.resize( topology.faceSize() );
    d.areas.resize( topology.faceSize(), 0.0 );
    Box3d box;
    for ( FaceId f : topology.getValidFaces() )
    {
        const auto [a, b, c] = topology.getTriVerts( f );
        for ( VertId v : { a, b, c } )
            box.include( d.pts[v] );
        if ( region && !region->test( f ) )
            continue;
        const Vector3d n = cross( d.pts[b] - d.pts[a], d.pts[c] - d.pts[a] );
        const double len = n.length();
        d.areas[f] = len / 2;
        d.normals[f] = len > 0 ? n / len : Vector3d();
        d.centers[f] = ( d.pts[a] + d.pts[b] + d.pts[c] ) / 3.0;
    }
    d.diag = box.valid() ? box.diagonal() : 0;
    if ( !( d.diag > 0 ) )
        return unexpected( "Mesh is degenerate" );
    return d;
}

} //anonymous namespace

Expected<FittedSmoothSurface> fitSmoothSurface( const MeshPart& mp, const FitSmoothSurfacesSettings& settings )
{
    MR_TIMER;
    auto d = makeData( mp.mesh, mp.region );
    if ( !d )
        return unexpected( std::move( d.error() ) );

    Vector<FaceId> faces;
    for ( FaceId f : mp.mesh.topology.getFaceIds( mp.region ) )
        if ( mp.mesh.topology.hasFace( f ) )
            faces.push_back( f );
    if ( faces.empty() )
        return unexpected( "Region is empty" );

    auto fit = fitAnalytic( *d, faces );
    d->bsplineThreshold = settings.bsplineThreshold;
    if ( settings.allowBSplines && d->bsplineThreshold > 0 )
        improveByBSpline( *d, faces, fit );
    return toFitted( *d, faces, fit );
}

Expected<MeshSmoothSurfaces> fitSmoothSurfaces( const Mesh& mesh, const Face2RegionMap& face2region,
    const FitSmoothSurfacesSettings& settings, const ProgressCallback& progress )
{
    MR_TIMER;
    auto d = makeData( mesh, nullptr );
    if ( !d )
        return unexpected( std::move( d.error() ) );

    Vector<Vector<FaceId>, RegionId> faces;
    for ( FaceId f : mesh.topology.getValidFaces() )
    {
        const RegionId r = face2region[f];
        if ( r >= faces.size() )
            faces.resize( r + 1 );
        faces[r].push_back( f );
    }

    Vector<Fit, RegionId> fits( faces.size() );
    if ( !ParallelFor( fits, [&]( RegionId r )
    {
        if ( !faces[r].empty() )
            fits[r] = fitAnalytic( *d, faces[r] );
    }, subprogress( progress, 0.0f, 0.5f ) ) )
        return unexpectedOperationCanceled();

    d->bsplineThreshold = settings.bsplineThreshold;
    if ( !( d->bsplineThreshold > 0 ) )
    {
        // the noise of the mesh: most regions are expected to have analytic surfaces
        Vector<double> rms;
        for ( RegionId r( 0 ); r < fits.endId(); ++r )
            if ( !faces[r].empty() )
                rms.push_back( fits[r].rms );
        std::nth_element( rms.begin(), rms.begin() + rms.size() / 2, rms.end() );
        d->bsplineThreshold = std::max( 2.5 * rms[rms.size() / 2], 1e-6 * d->diag );
    }

    if ( settings.allowBSplines && !ParallelFor( fits, [&]( RegionId r )
    {
        if ( !faces[r].empty() )
            improveByBSpline( *d, faces[r], fits[r] );
    }, subprogress( progress, 0.5f, 1.0f ) ) )
        return unexpectedOperationCanceled();

    MeshSmoothSurfaces res;
    res.bsplineThreshold = float( d->bsplineThreshold );
    res.surfaces.resize( faces.size(), { .surface = Plane3d() } );
    for ( RegionId r( 0 ); r < fits.endId(); ++r )
        if ( !faces[r].empty() )
            res.surfaces[r] = toFitted( *d, faces[r], fits[r] );
    reportProgress( progress, 1.0f );
    return res;
}

} //namespace MR
