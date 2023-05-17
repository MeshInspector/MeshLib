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
ToolPathResult getToolPath( const Mesh& inputMesh, const AffineXf3f& xf, const ToolPathParams& params )
{
    const Vector3f normal = Vector3f::plusZ();

    Mesh meshCopy( inputMesh );
    meshCopy.transform( xf );
    FixUndercuts::fixUndercuts( meshCopy, normal, params.voxelSize );

    OffsetParameters offsetParams;
    offsetParams.voxelSize = params.voxelSize;

    auto resMesh = offsetMesh( meshCopy, params.millRadius, offsetParams );
    if ( !resMesh.has_value() )
        return {};

    ToolPathResult  res{ .modifiedMesh = std::make_shared<Mesh>( *resMesh ) };
    const auto& mesh = *res.modifiedMesh;

    const auto box = mesh.getBoundingBox();
    const float safeZ = box.max.z + params.millRadius;

    const float dragSpeed = params.millRadius * 0.001f;
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min.z ) / params.sectionStep ) );

    Contour3f toolPath{ { 0, 0, safeZ } };
    std::ostringstream gcode;
    gcode << "G0 Z" << safeZ << "\t(rapid down to safe height)" << std::endl;

    MeshEdgePoint prevEdgePoint;

    const float critTransitionLengthSq = params.critTransitionLength * params.critTransitionLength;
    for ( int step = 0; step < steps; ++step )
    {        
        for ( const auto& section : extractPlaneSections( mesh, Plane3f{ plane.n, plane.d - params.sectionStep * step } ) )
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
            const float sectionStepSq = params.sectionStep * params.sectionStep;
            do
            {
                std::next( nextEdgePointIt ) != section.end() ? ++nextEdgePointIt : nextEdgePointIt = section.begin();
            } 
            while ( nextEdgePointIt != nearestPointIt && ( mesh.edgePoint( *nextEdgePointIt ) - mesh.edgePoint( *nearestPointIt ) ).lengthSq() < sectionStepSq );

            const auto pivotIt = contours.begin() + std::distance( section.begin(), nextEdgePointIt );
            
            if ( !prevEdgePoint.e.valid() || minDistSq > critTransitionLengthSq )
            {
                const auto lastPoint = toolPath.back();
                if ( lastPoint.z < safeZ )
                {
                    if ( safeZ - lastPoint.z > params.retractLength )
                    {
                        const float zRetract = lastPoint.z + params.retractLength;
                        toolPath.push_back( { toolPath.back().x, toolPath.back().y, zRetract } );
                        gcode << "G1 Z" << zRetract << " F" << params.retractFeed << "\t(retract)" << std::endl;
                        toolPath.push_back( { toolPath.back().x, toolPath.back().y, safeZ } );
                        gcode << "G0 Z" << safeZ << "\t(rapid up to safe height)" << std::endl;
                    }
                    else
                    {
                        toolPath.push_back( { toolPath.back().x, toolPath.back().y, safeZ } );
                        gcode << "G1 Z" << safeZ << " F" << params.retractFeed << "\t(retract)" << std::endl;
                    }
                }

                toolPath.push_back( { pivotIt->x, pivotIt->y, safeZ } );
                gcode << "G0 X" << pivotIt->x << " Y" << pivotIt->y << std::endl;

                if ( safeZ - pivotIt->z > params.plungeLength )
                {
                    const float zPlunge = pivotIt->z + params.plungeLength;
                    toolPath.push_back( { toolPath.back().x, toolPath.back().y, zPlunge } );
                    gcode << "G0 Z" << zPlunge << "\t(rapid down)" << std::endl;
                    toolPath.push_back( { toolPath.back().x, toolPath.back().y, pivotIt->z } );
                    gcode << "G1 Z" << pivotIt->z << " F" << params.plungeFeed << "\t(plunge)" << std::endl;
                }
                else
                {
                    toolPath.push_back( { toolPath.back().x, toolPath.back().y, pivotIt->z } );
                    gcode << "G1 Z" << pivotIt->z << " F" << params.plungeFeed << "\t(plunge)" << std::endl;
                }
            }
            else
            {
                Polyline3 transit;
                const auto sp = computeSurfacePath( mesh, prevEdgePoint, *nextEdgePointIt );
                if ( sp.has_value() && !sp->empty() )
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

    res.toolPath = std::make_shared<Polyline3>( Contours3f{ toolPath } );
    res.gcode = gcode.str();
    return res;
}
}
#endif
