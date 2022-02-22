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

namespace
{
const float maxError = std::numeric_limits<float>::epsilon() * 10.f;
}

namespace MR
{

bool fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holesEdges )
{
    MR_TIMER
    // check input
    assert( !holesEdges.empty() );
    if ( holesEdges.empty() )
    {
        spdlog::warn( "Holes edges size empty!" );
        return false;
    }

    // reorder to make edges ring with hole on left side
    bool badEdge = false;
    auto& meshTopology = mesh.topology;
    for ( const auto& edge : holesEdges )
    {
        if ( meshTopology.left( edge ) )
        {
            badEdge = true;
            break;
        }
    }
    assert( !badEdge );
    if ( badEdge )
    {
        spdlog::warn( "Holes edges have edge with face on left side" );
        return false;
    }

    // make border rings
    std::vector<EdgePath> paths( holesEdges.size() );
    for ( int i = 0; i < paths.size(); ++i )
    {
        auto& path = paths[i];
        for ( const auto& edge : leftRing( meshTopology, holesEdges[i] ) )
            path.push_back( edge );
    }

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
    float maxZ = 0.f;
    for ( const auto& path : paths )
    {
        Contour2f contour;
        for ( const auto& edge : path )
        {
            const auto localPoint = planeXfInv( mesh.orgPnt( edge ) );
            contour.push_back( Vector2f( localPoint.x, localPoint.y ) );
            if ( std::fabs( localPoint.z ) > maxZ ) maxZ = std::fabs( localPoint.z );
        }
        contour.push_back( contour[0] );
        contours2f.push_back( contour );
    }
    if ( maxZ > maxError )
    {
        spdlog::warn("Edges aren't in the same plane. Max Z = {}", maxZ );
        return false;
    }

    // make patch surface
    auto fillResult = PlanarTriangulation::triangulateDisjointContours( contours2f, false );
    if ( !fillResult )
    {
        spdlog::warn( "Cant triangulate contours" );
        return false;
    }
    Mesh& patchMesh = *fillResult;
    const auto holes = patchMesh.topology.findHoleRepresentiveEdges();

    // transform patch surface from plane to world space
    auto& patchMeshPoints = patchMesh.points;
    for ( auto& point : patchMeshPoints )
        point = planeXf( point );

    // make 
    std::vector<EdgePath> newPaths( holes.size() );
    for ( int i = 0; i < newPaths.size(); ++i )
    {
        EdgePath newPath;
        EdgeId edge = holes[i];
        if ( patchMesh.topology.left( edge ) ) edge = edge.sym();
        auto ring = leftRing( patchMesh.topology, edge );
        for ( const auto& e : ring )
            newPath.push_back( e );
        newPaths[i] = newPath;
    }

    // check that patch surface borders size equal original mesh borders size
    if ( paths.size() != newPaths.size() )
    {
        spdlog::warn( "Patch surface borders size different from original mesh borders size" );
        return false;
    }
    for ( int i = 0; i < paths.size(); ++i )
    {
        if ( paths[i].size() != newPaths[i].size() )
        {
            spdlog::warn( "Patch surface borders size different from original mesh borders size" );
            return false;
        }
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
    
    // add patch surface to original mesh
    mesh.addPartByMask( patchMesh, patchMesh.topology.getValidFaces(), false, paths, newPaths );
    return true;
}

}
