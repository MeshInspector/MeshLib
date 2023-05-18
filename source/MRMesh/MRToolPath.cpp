#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )

#include "MRToolPath.h"
#include "MRSurfacePath.h"
#include "MRFixUndercuts.h"
#include "MROffset.h"
#include "MRBox.h"
#include "MRMesh/MRExtractIsolines.h"
#include <sstream>

namespace MR
{
ToolPathResult constantZToolPath( const Mesh& inputMesh, const AffineXf3f& xf, const ToolPathParams& params )
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

    ToolPathResult  res{ .modifiedMesh = *resMesh };
    const auto& mesh = res.modifiedMesh;

    const auto box = mesh.getBoundingBox();
    const float safeZ = box.max.z + params.millRadius;

    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min.z ) / params.sectionStep ) );

    Contour3f toolPath{ { 0, 0, safeZ } };
    res.commands.push_back( { 0, 0 } );

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
            const auto nearestPoint = mesh.edgePoint( *nearestPointIt );
            do
            {
                std::next( nextEdgePointIt ) != section.end() ? ++nextEdgePointIt : nextEdgePointIt = section.begin();
            } 
            while ( nextEdgePointIt != nearestPointIt && ( mesh.edgePoint( *nextEdgePointIt ) - nearestPoint ).lengthSq() < sectionStepSq );

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
                        res.commands.push_back( { 1, params.retractFeed } );
                        toolPath.push_back( { toolPath.back().x, toolPath.back().y, safeZ } );
                        res.commands.push_back( { 0, 0 } );
                    }
                    else
                    {
                        toolPath.push_back( { toolPath.back().x, toolPath.back().y, safeZ } );
                        res.commands.push_back( { 1, params.retractFeed } );
                    }
                }

                toolPath.push_back( { pivotIt->x, pivotIt->y, safeZ } );
                res.commands.push_back( { 0, 0 } );

                if ( safeZ - pivotIt->z > params.plungeLength )
                {
                    const float zPlunge = pivotIt->z + params.plungeLength;
                    toolPath.push_back( { pivotIt->x, pivotIt->y, zPlunge } );
                    res.commands.push_back( { 0, 0 } );
                }
                 
                toolPath.push_back( *pivotIt );
                res.commands.push_back( { 1, params.plungeFeed } );                
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
                        res.commands.push_back( {} );
                    }
                    else
                    {
                        transit.addFromSurfacePath( mesh, *sp );
                        const auto transitContours = transit.contours().front();
                        toolPath.insert( toolPath.end(), transitContours.begin(), transitContours.end() );
                        res.commands.insert( res.commands.end(), transitContours.size(), {} );
                    }
                }

                toolPath.push_back( *pivotIt );
                res.commands.push_back( {} );
            }
            
            toolPath.insert( toolPath.end(),  pivotIt + 1, contours.end() );
            toolPath.insert( toolPath.end(), contours.begin() + 1, pivotIt + 1 );
            res.commands.insert( res.commands.end(), contours.size() - 1, {} );

            prevEdgePoint = *nextEdgePointIt;
        }        
    }

    res.toolPath = Polyline3( Contours3f{ toolPath } );
    return res;
}

std::string exportToolPathToGCode( const Polyline3& toolPath, const std::vector<GCommand>& commands )
{
    const auto contours = toolPath.contours();
    if ( contours.empty() )
        return {};

    const auto& contour = contours.front();
    assert( contour.size() == commands.size() );

    std::ostringstream gcode;    

    for ( size_t i = 0; i < contour.size(); ++i )
    {
        const auto& p = contour[i];
        const Vector3f prev = ( i > 0 ) ? contour[i - 1] : Vector3f{};
        
        gcode << "G" << commands[i].type;

        if ( p.x != prev.x )
            gcode << " X" << p.x;
        if ( p.y != prev.y )
            gcode << " Y" << p.y;
        if ( p.z != prev.z )
            gcode << " Z" << p.z;

        if ( commands[i].feed != 0 )
            gcode << " F" << commands[i].feed;

        gcode << std::endl;
    }

    return gcode.str();
}
}
#endif
