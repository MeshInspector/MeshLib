#include "MRFillContours2D.h"
#include <limits>
#include "MRMesh.h"
#include "MRVector2.h"
#include "MR2DContoursTriangulation.h"
#include "MRRingIterator.h"
#include "MREdgePaths.h"
#include "MRAffineXf3.h"
#include "MRPch/MRSpdlog.h"
#include "MRTimer.h"

namespace MR
{

tl::expected<void, std::string> fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges, Vector<UVCoord, VertId>* uvCoords )
{
    MR_TIMER
    // check input
    assert( !holeRepresentativeEdges.empty() );
    if ( holeRepresentativeEdges.empty() )
        return tl::make_unexpected( "No hole edges are given" );

    // reorder to make edges ring with hole on left side
    bool badEdge = false;
    auto& meshTopology = mesh.topology;
    for ( const auto& edge : holeRepresentativeEdges )
    {
        if ( meshTopology.left( edge ) )
        {
            badEdge = true;
            break;
        }
    }
    assert( !badEdge );
    if ( badEdge )
        return tl::make_unexpected( "Some hole edges have left face" );

    // make border rings
    const auto paths = meshTopology.getLeftRings( holeRepresentativeEdges );

    // calculate plane normal
    Vector3f planeNormal;
    for ( const auto& path : paths )
    {
        for ( const auto& edge : path )
            planeNormal += cross( mesh.orgPnt( edge ), mesh.destPnt( edge ) );
    }
    planeNormal = planeNormal.normalized();

    // find transformation from world to plane space and back
    const auto planeXf = AffineXf3f( Matrix3f::rotation( Vector3f::plusZ(), planeNormal ), mesh.orgPnt( paths[0][0] ) );
    const auto planeXfInv = planeXf.inverse();

    // make contours2D (on plane) from border rings (in world)
    Contours2f contours2f;
    contours2f.reserve( paths.size() );
    for ( const auto& path : paths )
    {
        contours2f.emplace_back();
        auto& contour = contours2f.back();
        contour.reserve( path.size() + 1 );
        for ( const auto& edge : path )
        {
            const auto localPoint = planeXfInv( mesh.orgPnt( edge ) );
            contour.emplace_back( Vector2f( localPoint.x, localPoint.y ) );
        }
        contour.emplace_back( contour.front() );
    }

    auto holeVertIds = std::make_unique<PlanarTriangulation::HolesVertIds>(
        PlanarTriangulation::findHoleVertIdsByHoleEdges( mesh.topology, paths ) );
    // make patch surface
    auto fillResult = PlanarTriangulation::triangulateDisjointContours( contours2f, holeVertIds.get() );
    holeVertIds.reset();
    if ( !fillResult )
        return tl::make_unexpected( "Cannot triangulate contours with self-intersections" );
    Mesh& patchMesh = *fillResult;
    const auto holes = patchMesh.topology.findHoleRepresentiveEdges();

    // transform patch surface from plane to world space
    auto& patchMeshPoints = patchMesh.points;
    for ( auto& point : patchMeshPoints )
        point = planeXf( point );

    // make 
    std::vector<EdgePath> newPaths( holes.size() );
    for ( int i = 0; i < newPaths.size(); ++i )
        newPaths[i] = patchMesh.topology.getLeftRing( holes[i] );

    // check that patch surface borders size equal original mesh borders size
    if ( paths.size() != newPaths.size() )
        return tl::make_unexpected( "Patch surface borders size different from original mesh borders size" );
    for ( int i = 0; i < paths.size(); ++i )
    {
        if ( paths[i].size() != newPaths[i].size() )
            return tl::make_unexpected( "Patch surface borders size different from original mesh borders size" );
    }

    // reorder to make edges ring with hole on right side
    for ( int i = 0; i < newPaths.size(); ++i )
    {
        auto& newPath = newPaths[i];
        if ( !patchMesh.topology.right( newPath[0] ) )
            continue;
        
        newPath.push_back( newPath[0] );
        reverse( newPath );
        newPath.pop_back();
    }

    // move patch surface border points to original position (according original mesh)
    auto& patchMeshTopology = patchMesh.topology;
    auto& meshPoints = mesh.points;
    for ( int i = 0; i < paths.size(); ++i )
    {
        auto& path = paths[i];
        auto& newPath = newPaths[i];
        for ( int j = 0; j < path.size(); ++j )
            patchMeshPoints[patchMeshTopology.org( newPath[j] )] = meshPoints[meshTopology.org( path[j] )];
    }
    
    PartMapping pmap;
    VertHashMap vertHashMap;
    if ( uvCoords )
        pmap.src2tgtVerts = &vertHashMap;

    // add patch surface to original mesh
    mesh.addPartByMask( patchMesh, patchMesh.topology.getValidFaces(), false, paths, newPaths, pmap );
    if ( uvCoords )
    {
        Vector<UVCoord, VertId> newUVCoords( *uvCoords );

        if ( mesh.points.size() > uvCoords->size() )
            newUVCoords.resize( mesh.points.size() );

        for ( const auto& [fromVert, thisVert] : vertHashMap )
            newUVCoords[thisVert] = (*uvCoords)[fromVert];

        *uvCoords = std::move( newUVCoords );
    }
    return {};
}

}
