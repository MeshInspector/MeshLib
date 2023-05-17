#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )

#include "MRToolPath.h"
#include "MRSurfacePath.h"
#include "MRFixUndercuts.h"
#include "MROffset.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRMesh/MRExtractIsolines.h"
#include <sstream>

namespace MR
{
ToolPathResult getToolPath( Mesh& mesh, float millRadius, float voxelSize, float sectionsStep, float critLength,
    float plungeLength, float retractLength,
    float plungeFeed, float retractFeed )
{
    const Vector3f normal = Vector3f::plusZ();
    FixUndercuts::fixUndercuts( mesh, normal, voxelSize );

    OffsetParameters offsetParams;
    offsetParams.voxelSize = voxelSize;

    auto resMesh = offsetMesh( mesh, millRadius, offsetParams );
    if ( !resMesh.has_value() )
        return {};

    const auto box = mesh.getBoundingBox();
    const float safeZ = box.max.z + millRadius;

    const float dragSpeed = millRadius * 0.001f;
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min.z ) / sectionsStep ) );

    Contour3f toolPath{ { 0, 0, safeZ } };
    std::ostringstream gcode;
    gcode << "G0 Z" << safeZ << "\t(rapid down to safe height)" << std::endl;

    MeshEdgePoint prevEdgePoint;

    const float critLengthSq = critLength * critLength;
    for ( int step = 0; step < steps; ++step )
    {        
        for ( const auto& section : extractPlaneSections( mesh, Plane3f{ plane.n, plane.d - sectionsStep * step } ) )
        {
            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, section );
            const auto contours = polyline.contours().front();
            
            auto nearestPointIt = section.begin();

            float minDistSq = FLT_MAX;

            if ( prevEdgePoint.e.valid() )
            {
                for ( auto it = section.begin(); it < section.end(); ++it )
                {
                    float distSq = ( mesh.edgePoint( *it ) - mesh.edgePoint( prevEdgePoint ) ).lengthSq();
                    if ( distSq < minDistSq )
                    {
                        minDistSq = distSq;
                        nearestPointIt = it;
                    }
                }
            }
            auto nextEdgePointIt = nearestPointIt;

            do
            {
                std::next( nextEdgePointIt ) != section.end() ? ++nextEdgePointIt : nextEdgePointIt = section.begin();
            } 
            while ( nextEdgePointIt != nearestPointIt && ( mesh.edgePoint( *nextEdgePointIt ) - mesh.edgePoint( *nearestPointIt ) ).lengthSq() < sectionsStep * sectionsStep );

            const auto pivotIt = contours.begin() + std::distance( section.begin(), nextEdgePointIt );
            
            if ( !prevEdgePoint.e.valid() || minDistSq > critLengthSq )
            {
                const auto lastPoint = toolPath.back();
                if ( lastPoint.z < safeZ )
                {
                    if ( safeZ - lastPoint.z > retractLength )
                    {
                        const float zRetract = lastPoint.z + retractLength;
                        toolPath.push_back( { toolPath.back().x, toolPath.back().y, zRetract } );
                        gcode << "G1 Z" << zRetract << " F" << retractFeed << "\t(retract)" << std::endl;
                        toolPath.push_back( { toolPath.back().x, toolPath.back().y, safeZ } );
                        gcode << "G0 Z" << safeZ << "\t(rapid up to safe height)" << std::endl;
                    }
                    else
                    {
                        toolPath.push_back( { toolPath.back().x, toolPath.back().y, safeZ } );
                        gcode << "G1 Z" << safeZ << " F" << retractFeed << "\t(retract)" << std::endl;
                    }
                }

                toolPath.push_back( { pivotIt->x, pivotIt->y, safeZ } );
                gcode << "G0 X" << pivotIt->x << " Y" << pivotIt->y << std::endl;

                if ( safeZ - pivotIt->z > plungeLength )
                {
                    const float zPlunge = pivotIt->z + plungeLength;
                    toolPath.push_back( { toolPath.back().x, toolPath.back().y, zPlunge } );
                    gcode << "G0 Z" << zPlunge << "\t(rapid down)" << std::endl;
                    toolPath.push_back( { toolPath.back().x, toolPath.back().y, pivotIt->z } );
                    gcode << "G1 Z" << pivotIt->z << " F" << plungeFeed << "\t(plunge)" << std::endl;
                }
                else
                {
                    toolPath.push_back( { toolPath.back().x, toolPath.back().y, pivotIt->z } );
                    gcode << "G1 Z" << pivotIt->z << " F" << plungeFeed << "\t(plunge)" << std::endl;
                }
            }
            else
            {
                Polyline3 transit;
                const auto sp = computeSurfacePath( mesh, prevEdgePoint, *nextEdgePointIt );
                if ( sp.has_value() )
                {
                    if ( sp->size() == 1 )
                    {
                        const auto p = mesh.edgePoint( sp->front() );
                        toolPath.push_back( p );
                        gcode << "G1 X" << p.x << " Y" << p.y << " Z" << p.z << "\t(transit)" << std::endl;
                    }
                    else
                    {
                        transit.addFromSurfacePath( mesh, *sp );
                        const auto transitContours = transit.contours().front();
                        toolPath.insert( toolPath.end(), transitContours.begin(), transitContours.end() );

                        for ( const auto& p : transitContours )
                            gcode << "G1 X" << p.x << " Y" << p.y << " Z" << p.z << "\t(transit)" << std::endl;
                    }
                }

                gcode << "G1 X" << pivotIt->x << " Y" << pivotIt->y << " Z" << pivotIt->z << "\t(transit)" << std::endl;
            }
            
            toolPath.insert( toolPath.end(),  pivotIt, contours.end() );
            for ( auto it = pivotIt + 1; it < contours.end(); ++it )
                gcode << "G1 X" << it->x << " Y" << it->y << std::endl;

            toolPath.insert( toolPath.end(), contours.begin(), pivotIt + 1 );
            for ( auto it = contours.begin(); it < pivotIt + 1; ++it )
                gcode << "G1 X" << it->x << " Y" << it->y << std::endl;

            prevEdgePoint = *nextEdgePointIt;
        }        
    }

    return { .toolPath = std::make_shared<Polyline3>( Contours3f{ toolPath } ), .gcode = gcode.str() };
}
}
#endif
