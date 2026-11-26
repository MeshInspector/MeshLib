#include "MRMeshAnalyticRegions.h"
#include "MRMeshAnalyticRegionsData.hpp"
#include "MRLine3.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRTimer.h"

namespace MR
{

namespace
{

using namespace AnalyticRegions;

/// the surface of a region with given vertices in the public form
AnalyticSurface toAnalyticSurface( const Surf& s, const Vector<V3>& pts )
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
    }
    assert( false );
    return Plane3d();
}

} //anonymous namespace

Expected<MeshAnalyticRegions> findAnalyticRegions( const Mesh& mesh, const FindAnalyticRegionsSettings& settings, const ProgressCallback& progress )
{
    MR_TIMER;
    if ( mesh.topology.numValidFaces() <= 0 )
        return unexpected( "Mesh is empty" );
    auto d = makeData( mesh, settings.faceBoundaries, settings.tolerance );
    if ( !( d.tol > 0 ) )
        return unexpected( "Mesh is degenerate" );
    if ( !growRegions( d, progress ) )
        return unexpectedOperationCanceled();

    MeshAnalyticRegions res;
    res.tolerance = float( d.tol );
    res.face2region.resize( mesh.topology.faceSize() );
    const auto regionFaces = d.facesOfRegions();
    for ( size_t i = 0; i < regionFaces.size(); ++i )
    {
        const auto& faces = regionFaces[i];
        if ( faces.empty() )
            continue;
        const RegionId id( res.surfaces.size() );
        res.surfaces.push_back( toAnalyticSurface( d.regions[i].surf, d.pointsOf( faces ) ) );
        for ( int f : faces )
            res.face2region[FaceId( f )] = id;
    }
    return res;
}

} //namespace MR
