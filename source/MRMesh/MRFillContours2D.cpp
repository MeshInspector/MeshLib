#include "MRFillContours2D.h"
#include "MRMesh.h"
#include "MRVector2.h"
#include "MR2DContoursTriangulation.h"

#include "MRMesh/MREdgeIterator.h"
#include "MRMesh/MRIteratorRange.h"
#include "MRMesh/MRRingIterator.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include "MRBestFit.h"

namespace MR
{

bool MR::fillContours2D( Mesh& mesh, std::vector<EdgePath>& paths )
{
    if ( paths.empty() )
    {
        spdlog::warn( "Paths size empty!" );
        return false;
    }

    // reorder to make edges ring with hole on left side
    for ( int i = 0; i < paths.size(); ++i )
    {
        auto& path = paths[i];
        if ( !mesh.topology.left( path[0] ) )
            continue;

        const auto size = path.size();
        EdgePath reversPath( size );
        reversPath[0] = path[0].sym();
        for ( int j = 1; j < size; ++j )
            reversPath[j] = path[size - j].sym();
        path = std::move( reversPath );
    }

    // make Contours from EdgePaths
    Contours3f contours3f;
    for ( const auto& path : paths )
    {
        Contour3f contour;
        for ( const auto& edge : path )
        {
            const Vector3f point = mesh.orgPnt( edge );
            contour.push_back( point );
        }
        contour.push_back( contour[0] );
        contours3f.push_back( contour );
    }

    // calculate center point
    Vector3f centerPoint;
    size_t pointsNumber = 0;
    for ( const auto& contour : contours3f )
    {
        const auto max = contour.size() - 1;
        pointsNumber += max;
        for ( int i = 0; i < max; ++i )
            centerPoint += contour[i];
    }
    centerPoint = centerPoint / float( pointsNumber );

    // calculate plane normal
    Vector3f planeNormal;
    for ( const auto& contour : contours3f )
    {
        const auto max = contour.size() - 1;
        for ( int i = 0; i < max; ++i )
        {
            planeNormal += cross( ( contour[i] - centerPoint ), ( contour[i + 1] - centerPoint ) );
        }
    }
    planeNormal = planeNormal.normalized();

    // find transformation from world to plane space and back
    const auto planeXf = AffineXf3f( Matrix3f::rotation( Vector3f::plusZ(), planeNormal ), centerPoint );
    const auto planeXfInv = planeXf.inverse();

    // make contours2D (on plane) from contours3D (in world)
    Contours2f contours2f;
    float maxZ = 0.f; // FOR DEBUG
    for ( const auto& contour3f : contours3f )
    {
        Contour2f contour;
        for ( const auto& point : contour3f )
        {
            const auto localPoint = planeXfInv( point );
            contour.push_back( Vector2f( localPoint.x, localPoint.y ) );
            if ( std::fabs( localPoint.z ) > maxZ ) maxZ = std::fabs( localPoint.z ); // DEBUG
        }
        contours2f.push_back( contour );
    }
    spdlog::info( "DEBUG maxZ = {}", maxZ ); // DEBUG

    // make patch surface
    auto fillResult = PlanarTriangulation::triangulateContours( contours2f, true );
    if ( !fillResult )
    {
        spdlog::warn( "Cant triangulate contours" );
        return false;
    }
    Mesh& additionalMesh = *fillResult;
    auto holes = additionalMesh.topology.findHoleRepresentiveEdges();
    auto holesForSearch = holes;

    // transform patch surface from plane to world space
    auto& addMeshPoints = additionalMesh.points;
    for ( auto& point : addMeshPoints )
        point = planeXf( point );
    
    
    // find 
    std::vector<Vector2i> mapToSourceMesh( paths.size(), Vector2i( -1, -1 ) );
    for ( int i = 0; i < paths.size(); ++i )
    {
        auto& path = paths[i];
        for ( int j = 0; j < path.size(); ++j )
        {
            auto sourceOrg = mesh.orgPnt( path[j] );
            auto sourceDest = mesh.destPnt( path[j] );
            bool found = false;
            for ( int k = int( holesForSearch.size() ) - 1; k >= 0; --k )
            {
                auto addOrg = additionalMesh.orgPnt( holesForSearch[k] );
                auto addDest = additionalMesh.destPnt( holesForSearch[k] );
                found = ( ( sourceOrg - addOrg ).lengthSq() < 1.e-6f && ( sourceDest - addDest ).lengthSq() < 1.e-6f ) ||
                    ( ( sourceOrg - addDest ).lengthSq() < 1.e-6f && ( sourceDest - addOrg ).lengthSq() < 1.e-6f );
//                 std::string text = fmt::format( "DEBUG i = {:2d} j = {:2d} k = {:2d} sOrg = ( {:4f} , {:4f} ) sDest = ( {:4f} , {:4f} ) aOrg = ( {:4f} , {:4f} ) aDest = ( {:4f} , {:4f} )",
//                     i, j, k, sourceOrg.x, sourceOrg.y, sourceDest.x, sourceDest.y, addOrg.x, addOrg.y, addDest.x, addDest.y );
//                 std::cout << text << std::endl;
                if ( found )
                {
                    mapToSourceMesh[k] = Vector2i( i, j );
                    break;
                }
            }
            if ( found )
                break;
        }
    }

    // check that all holes edges founded
    for ( const auto& item : mapToSourceMesh )
    {
        if ( item == Vector2i( -1, -1 ) )
        {
            spdlog::warn( "Some hole edge not found pair" );
            return false;
        }
    }

    std::vector<EdgePath> newPaths( holes.size() );
    for ( int i = 0; i < newPaths.size(); ++i )
    {
        EdgePath newPath;
        EdgeId edge = holes[i];
        if ( additionalMesh.topology.left( edge ) ) edge = edge.sym();
        auto ring = leftRing( additionalMesh.topology, edge );
        for ( const auto& e : ring )
            newPath.push_back( e );
        newPaths[mapToSourceMesh[i].x] = newPath;
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

    //make true ring direction
    for ( int i = 0; i < newPaths.size(); ++i )
    {
        auto& newPath = newPaths[i];
        if ( !additionalMesh.topology.right( newPath[0] ) )
            continue;

        const auto size = newPath.size();
        EdgePath reversPath( size );
        for ( int j = 1; j < size; ++j )
            reversPath[size - j] = newPath[j].sym();
        reversPath[0] = newPath[0].sym();
        newPath = reversPath;
    }

    // move patch surface border points to original position (according original mesh)
    auto& addMeshTopology = additionalMesh.topology;
    auto& meshPoints = mesh.points;
    auto& meshTopology = mesh.topology;
    for ( int i = 0; i < paths.size(); ++i )
    {
        auto& path = paths[i];
        auto& newPath = newPaths[i];
        for ( int j = 0; j < path.size(); ++j )
        {
            addMeshPoints[addMeshTopology.org( newPath[j] )] = meshPoints[meshTopology.org( path[j] )];
        }
    }
    
    // add patch surface to original mesh
    mesh.addPartByMask( additionalMesh, additionalMesh.topology.getValidFaces(), false, paths, newPaths );
    return true;
}

}
