#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )

#include "MRToolPath.h"
#include "MRSurfacePath.h"
#include "MRFixUndercuts.h"
#include "MROffset.h"
#include "MRBox.h"
#include "MRExtractIsolines.h"
#include "MRSurfaceDistance.h"
#include "MRMeshDirMax.h"
#include "MRParallelFor.h"
#include "MRObjectGcode.h"
#include "MRExpected.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRRegionBoundary.h"
#include "MRMeshDecimate.h"
#include "MRBitSetParallelFor.h"

#include "MRPch/MRTBB.h"
#include <sstream>
#include <span>

namespace MR
{

Vector2f rotate90( const Vector2f& v )
{
    return { v.y, -v.x };
}

Vector2f rotateMinus90( const Vector2f& v )
{
    return { -v.y, v.x };
}

bool calcCircleCenter( const Vector2f& p0, const Vector2f& p1, const Vector2f& p2, Vector2f& center )
{
    const auto dif1 = p1 - p0;
    const auto dif2 = p2 - p0;

    const auto proj1 = dot( dif1, p0 + p1 );
    const auto proj2 = dot( dif2, p0 + p2 );
    const auto det = cross( dif1, p2 - p1 ) * 2;

    if ( fabs( det ) < 1e-10 )
        return false;

    // calc center coords
    center.x = ( dif2.y * proj1 - dif1.y * proj2 ) / det;
    center.y = ( dif1.x * proj2 - dif2.x * proj1 ) / det;

    return true;
}

// get coordinate along specified axis
float coord( const GCommand& command, Axis axis )
{
    return ( axis == Axis::X ) ? command.x :
        ( axis == Axis::Y ) ? command.y : command.z;
}
// get projection to the plane orthogonal to the specified axis
Vector2f project( const GCommand& command, Axis axis )
{
    return ( axis == Axis::X ) ? Vector2f{ command.y, command.z } :
        ( axis == Axis::Y ) ? Vector2f{ command.x, command.z } : Vector2f{ command.x, command.y };
}

Expected<Mesh, std::string> preprocessMesh( const Mesh& inputMesh, const ToolPathParams& params, bool needToDecimate )
{
    OffsetParameters offsetParams;
    offsetParams.voxelSize = params.voxelSize;
    offsetParams.callBack = subprogress( params.cb, 0.0f, 0.15f );
    const Vector3f normal = Vector3f::plusZ();

    const auto offsetRes = offsetMesh( inputMesh, params.millRadius, offsetParams );
    if ( !offsetRes )
        return unexpectedOperationCanceled();

    Mesh meshCopy = *offsetRes;
    if ( params.xf )
        meshCopy.transform( *params.xf );
    
    FixUndercuts::fixUndercuts( meshCopy, normal, params.voxelSize );
    if ( !reportProgress( params.cb, 0.20f ) )
        return unexpectedOperationCanceled();

    if ( needToDecimate )
    {
        const auto decimateResult = decimateMesh( meshCopy, { .progressCallback = subprogress( params.cb, 0.20f, 0.25f ) } );
        if ( decimateResult.cancelled )
            return unexpectedOperationCanceled();
    }
        
    return meshCopy;
}

// compute surface path between given edge points
void addSurfacePath( std::vector<GCommand>& gcode, const Mesh& mesh, const MeshEdgePoint& start, const MeshEdgePoint& end )
{
    const auto sp = computeSurfacePath( mesh, start, end );
    if ( !sp.has_value() || sp->empty() )
        return;

    if ( sp->size() == 1 )
    {
        const auto p = mesh.edgePoint( sp->front() );
        gcode.push_back( { .x = p.x, .y = p.y, .z = p.z } );
    }
    else
    {
        Polyline3 transit;
        transit.addFromSurfacePath( mesh, *sp );
        const auto transitContours = transit.contours().front();
        for ( const auto& p : transitContours )
            gcode.push_back( { .x = p.x, .y = p.y, .z = p.z } );
    }

    const auto p = mesh.edgePoint( end );
    gcode.push_back( { .x = p.x, .y = p.y, .z = p.z } );
}

// computes all sections of modified mesh along the given axis
// if a selected area is specified in the original mesh, then only points projected on it will be taken into consideration
std::vector<PlaneSections> extractAllSections( const Mesh& mesh, const MeshPart& origMeshPart, const Box3f& box, Axis axis, float sectionStep, int steps, BypassDirection bypassDir, ProgressCallback cb )
{
    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone{ 0 };

    std::vector<PlaneSections> sections( steps );

    const int axisIndex = int( axis );
    constexpr Vector3f normals[3] = { {1, 0, 0}, {0, 1, 0}, {0, 0 ,1} };
    const Vector3f normal = normals[axisIndex];
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );

    tbb::parallel_for( tbb::blocked_range<int>( 0, steps ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int step = range.begin(); step < range.end(); ++step )
        {
            if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            const float currentCoord = plane.d - sectionStep * step;

            auto stepSections = extractPlaneSections( mesh, Plane3f{ plane.n, currentCoord } );
            // if there is not selected area just move the sections as is
            if ( !origMeshPart.region && bypassDir == BypassDirection::Clockwise )
            {
                sections[step] = std::move( stepSections );
                continue;
            }

            sections[step].reserve( stepSections.size() );

            for ( auto& section : stepSections )
            {
                if ( bypassDir == BypassDirection::CounterClockwise )
                    std::reverse( section.begin(), section.end() );

                if ( !origMeshPart.region )
                {
                    sections[step].push_back( std::move( section ) );
                    continue;
                }

                auto startIt = section.begin();
                auto endIt = startIt;

                for ( auto it = section.begin(); it < section.end(); ++it )
                {
                    // try to project point on the original mesh
                    Vector3f rayStart = mesh.edgePoint( *it );
                    rayStart[axisIndex] = box.max[axisIndex];

                    auto intersection = rayMeshIntersect( origMeshPart.mesh, Line3f{ rayStart, -normal } );

                    // in case of success expand the interval
                    if ( intersection )
                    {
                        const auto faceId = origMeshPart.mesh.topology.left( intersection->mtp.e );
                        if ( origMeshPart.region->test( faceId ) )
                        {
                            ++endIt;
                            continue;
                        }
                    }
                    // otherwise add current interval to the result (if it is not empty)
                    if ( startIt < endIt )
                    {
                        sections[step].push_back( SurfacePath{ startIt, endIt } );
                    }
                    // reset the interval from the last point
                    startIt = it + 1;
                    endIt = startIt;
                }
                // add the last interval (if it is not empty)
                if ( startIt < section.end() )
                {
                    sections[step].push_back( SurfacePath{ startIt, section.end() } );
                }
            }            
        }

        if ( cb )
            numDone += range.size();

        if ( cb && std::this_thread::get_id() == mainThreadId )
        {
            if ( !cb( float( numDone ) / float( steps ) ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    if ( !keepGoing.load( std::memory_order_relaxed ) || !reportProgress( cb, 1.0f ) )
        return {};

    return sections;
}

std::vector<IsoLines> extractAllIsolines( const Mesh& mesh, const SurfacePath& startSurfacePath, float sectionStep, BypassDirection bypassDir, ProgressCallback cb )
{
    MR::VertScalars distances;
   
    VertBitSet startVertices( mesh.topology.vertSize() );
    for ( const auto& ep : startSurfacePath )
    {
        startVertices.set( mesh.topology.org( ep.e ) );
    }

    distances = computeSurfaceDistances(mesh, startVertices);    

    const float topExcluded = FLT_MAX;
    const auto [min, max] = parallelMinMax( distances.vec_, &topExcluded );

    const size_t numIsolines = size_t( ( max - min ) / sectionStep ) - 1;

    const auto& topology = mesh.topology;
    std::vector<IsoLines> isoLines( numIsolines );

    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone{ 0 };

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, isoLines.size() ),
                       [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            isoLines[i] = extractIsolines( topology, distances, sectionStep * ( i + 1 ) );

            if ( bypassDir == BypassDirection::CounterClockwise )
            {
                for ( auto& isoLine : isoLines[i] )
                {
                    std::reverse( isoLine.begin(), isoLine.end() );
                }
            }
        }

        if ( cb )
            numDone += range.size();

        if ( cb && std::this_thread::get_id() == mainThreadId )
        {
            if ( !cb( float( numDone ) / float( numIsolines ) ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    if ( !keepGoing.load( std::memory_order_relaxed ) || !reportProgress( cb, 1.0f ) )
        return {};

    return isoLines;
}

using V3fIt = std::vector<Vector3f>::const_iterator;
using Intervals = std::vector<std::pair<V3fIt, V3fIt>>;

// compute intervals of the slice which are projected on the selected area in the mesh (skip if not specified)
Intervals getIntervals( const MeshPart& mp, const V3fIt startIt, const V3fIt endIt, const V3fIt beginVec, const V3fIt endVec, bool moveForward )
{
    Intervals res;
    if ( startIt == endIt )
        return res;

    auto startInterval = moveForward ? startIt : endIt;
    auto endInterval = startInterval;

    // if the point is projected on the selected area expand the current interval (forward or backward in depending of the parameter )
    // otherwise add interval to the result
    const auto processPoint = [&] ( V3fIt it )
    {
        const auto mpr = mp.mesh.projectPoint( *it );
        const auto faceId = mp.mesh.topology.left( mpr->mtp.e );

        if ( !mp.region || ( mpr && mp.region->test( faceId ) ) )
        {
            if ( moveForward || endInterval > beginVec )
                moveForward ? ++endInterval : --endInterval;
            return;
        }

        if ( startInterval != endInterval )
        {
            if ( !moveForward && startInterval == endVec )
                res.emplace_back( startInterval - 1, endInterval );
            else
                res.emplace_back( startInterval, endInterval );
        }

        if ( moveForward )
            startInterval = endInterval = it + 1;
        else
            startInterval = endInterval = it - 1;
    };

    // we might be able to pass from the start point to the end directly directly,
    //otherwise we have to reach end ( beginning ) of the entire section and continue from beginning (end)
    if ( moveForward )
    {
        if ( startIt < endIt )
        {
            for ( auto it = startIt; it < endIt; ++it )
                processPoint( it );

            if ( startInterval < endInterval )
                res.emplace_back( startInterval, endInterval );

            return res;
        }

        for ( auto it = startIt; it < endVec; ++it )
            processPoint( it );

        if ( startInterval < endInterval )
            res.emplace_back( startInterval, endInterval );

        startInterval = beginVec;
        endInterval = beginVec;

        for ( auto it = beginVec; it < endIt; ++it )
            processPoint( it );

        if ( startInterval != endInterval )
            res.emplace_back( startInterval, endInterval );
    }
    else
    {
        if ( startIt < endIt )
        {
            for ( auto it = endIt - 1; it >= startIt; --it )
                processPoint( it );

            if ( startInterval != endInterval )
                res.emplace_back( startInterval, endInterval );

            return res;
        }

        for ( auto it = endIt - 1; it > beginVec; --it )
            processPoint( it );

        processPoint( beginVec );

        if ( startInterval != endInterval )
            res.emplace_back( startInterval, endInterval );

        startInterval = endVec;
        endInterval = endVec;

        for ( auto it = endVec - 1; it >= startIt; --it )
            processPoint( it );

        if ( startInterval == endVec )
            --startInterval;

        if ( startInterval != endInterval )
            res.emplace_back( startInterval, endInterval );
    }

    return res;
}

// if distance between the last point and the given one is more than critical distance
// we should make a transit on the safe height
void transitOverSafeZ( V3fIt it, ToolPathResult& res, const ToolPathParams& params, float safeZ, float currentZ, float& lastFeed )
{
    // retract the tool fast if possible
    if ( safeZ - currentZ > params.retractLength )
    {
        const float zRetract = currentZ + params.retractLength;
        res.commands.push_back( { .feed = params.retractFeed, .z = zRetract } );
        res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
    }
    else
    {
        res.commands.push_back( { .feed = params.retractFeed, .z = safeZ } );
    }

    res.commands.push_back( { .type = MoveType::FastLinear, .x = it->x, .y = it->y } );

    // plunge the tool fast if possible
    if ( safeZ - it->z > params.plungeLength )
    {
        const float zPlunge = it->z + params.plungeLength;
        res.commands.push_back( { .type = MoveType::FastLinear, .z = zPlunge } );
    }
    res.commands.push_back( { .feed = params.plungeFeed, .x = it->x, .y = it->y, .z = it->z } );
    lastFeed = params.plungeFeed;
}

Expected<ToolPathResult, std::string> lacingToolPath( const MeshPart& mp, const ToolPathParams& params, Axis cutDirection )
{
    if ( cutDirection != Axis::X && cutDirection != Axis::Y )
        return unexpected( "Lacing can be done along the X or Y axis" );

    const auto cutDirectionIdx = int( cutDirection );
    const auto sideDirection = ( cutDirection == Axis::X ) ? Axis::Y : Axis::X;
    const auto sideDirectionIdx = int( sideDirection );
    auto preprocessedMesh = preprocessMesh( mp.mesh, params, false );
    if ( !preprocessedMesh )
        return unexpected( preprocessedMesh.error() );

    ToolPathResult  res{ .modifiedMesh = std::move( *preprocessedMesh ) };
    const auto& mesh = res.modifiedMesh;

    const auto box = mp.mesh.computeBoundingBox( params.xf );
    const float safeZ = std::max( box.max.z + 10.0f * params.millRadius, params.safeZ );

    const Vector3f normal = (cutDirection == Axis::X) ? Vector3f::plusX() : Vector3f::plusY();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min[cutDirectionIdx] ) / params.sectionStep ) );

    MeshEdgePoint lastEdgePoint = {};
    // bypass direction is not meaningful for this toolpath, so leave it as default
    const auto allSections = extractAllSections( mesh, mp.mesh, box, cutDirection, params.sectionStep, steps, BypassDirection::Clockwise, subprogress( params.cb, 0.25f, 0.5f ) );
    if ( allSections.empty() )
        return unexpectedOperationCanceled();
    const auto sbp = subprogress( params.cb, 0.5f, 1.0f );

    float lastFeed = 0;  

    Vector3f lastPoint;
    // if the last point is equal to parameter, do nothing
    // otherwise add new move
    const auto addPoint = [&] ( const Vector3f& point )
    {
        if ( lastPoint == point )
            return;

        if ( lastFeed == params.baseFeed )
        {
            ( cutDirection == Axis::X ) ?
            res.commands.push_back( { .y = point.y, .z = point.z } ) :
            res.commands.push_back( { .x = point.x, .z = point.z } );
        }
        else
        {
            ( cutDirection == Axis::X ) ?
            res.commands.push_back( { .feed = params.baseFeed, .y = point.y, .z = point.z } ) :
            res.commands.push_back( { .feed = params.baseFeed, .x = point.x, .z = point.z } );

            lastFeed = params.baseFeed;
        }
            
        lastPoint = point;
    };

    const float critDistSq = params.critTransitionLength * params.critTransitionLength;

    for ( int step = 0; step < steps; ++step )
    {
        if ( !reportProgress( sbp, float( step ) / steps ) )
            return unexpectedOperationCanceled();

        const auto sections = allSections[step];
        if ( sections.empty() )
            continue;

        // there could be many sections in one slice
        for ( const auto& section : sections )
        {
            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, section );
            const auto contours = polyline.contours();            
            const auto contour = contours.front();

            if ( contour.size() < 3 )
                continue;

            // we need to find the most left and the most right point on the mesh
            // and move tol from one side to another
            auto bottomLeftIt = contour.end();
            auto bottomRightIt = contour.end();

            for ( auto it = contour.begin(); it < contour.end(); ++it )
            {
                if ( bottomLeftIt == contour.end() || ( *it )[sideDirectionIdx] < ( *bottomLeftIt )[sideDirectionIdx] || ( ( *it )[sideDirectionIdx] == ( *bottomLeftIt )[sideDirectionIdx] && it->z < bottomLeftIt->z ) )
                    bottomLeftIt = it;

                if ( bottomRightIt == contour.end() || ( *it )[sideDirectionIdx] > ( *bottomRightIt )[sideDirectionIdx] || ( ( *it )[sideDirectionIdx] == ( *bottomRightIt )[sideDirectionIdx] && it->z < bottomRightIt->z ) )
                    bottomRightIt = it;
            }

            // move from left to right and then  from right to left to make the smoothest path
            const bool moveForward = step & 1;

            if ( cutDirection == Axis::Y )
            {
                std::swap( bottomLeftIt, bottomRightIt );
                if ( !moveForward && bottomLeftIt != contour.begin() )
                    --bottomLeftIt;
            }
            
            const auto intervals = getIntervals( mp, bottomLeftIt, bottomRightIt, contour.begin(), contour.end(), moveForward );
            if ( intervals.empty() )
                continue;

            // go to the first point through the safe height
            if ( res.commands.empty() )
            {
                res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
                res.commands.push_back( { .type = MoveType::FastLinear, .x = intervals[0].first->x, .y = intervals[0].first->y } );
                res.commands.push_back( { .type = MoveType::FastLinear, .z = intervals[0].first->z } );
            }
            else
            {
                // otherwise compute distance from the last point to a new one and decide how to get to it
                const auto nextEdgePoint = section[intervals[0].first - contour.begin()];
                const auto distSq = ( mesh.edgePoint( lastEdgePoint ) - mesh.edgePoint( nextEdgePoint ) ).lengthSq();

                if ( distSq > critDistSq )
                    transitOverSafeZ( intervals[0].first, res, params, safeZ, res.commands.back().z, lastFeed );
                else
                    addSurfacePath( res.commands, mesh, lastEdgePoint, nextEdgePoint );
            }
            
            // process all the intervals except the last one and transit to the next
            for ( size_t i = 0; i < intervals.size() - 1; ++i )
            {
                const auto& interval = intervals[i];

                if ( moveForward )
                {
                    for ( auto it = interval.first; it < interval.second; ++it )
                        addPoint( *it );
                }
                else
                {
                    if ( interval.first == contour.begin() )
                        continue;

                    for ( auto it = interval.first - 1; it >= interval.second; --it )
                        addPoint( *it );
                }

                if ( *intervals[i + 1].first != lastPoint )
                    transitOverSafeZ( intervals[i + 1].first, res, params, safeZ, res.commands.back().z, lastFeed );
            }
            // process the last interval
            if ( moveForward )
            {
                for ( auto it = intervals.back().first; it < intervals.back().second; ++it )
                    addPoint( *it );
            }
            else
            {
                for ( auto it = intervals.back().first - 1; it >= intervals.back().second; --it )
                    addPoint( *it );
            }

            const auto dist = ( intervals.back().second - contour.begin() ) % contour.size();
            lastEdgePoint = section[dist];
        }
    }

    if ( !reportProgress( params.cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}


Expected<ToolPathResult, std::string>  constantZToolPath( const MeshPart& mp, const ToolPathParams& params )
{
    auto preprocessedMesh = preprocessMesh( mp.mesh, params, false );
    if ( !preprocessedMesh )
        return unexpected( preprocessedMesh.error() );

    ToolPathResult  res{ .modifiedMesh = std::move( *preprocessedMesh ) };
    const auto& mesh = res.modifiedMesh;

    const auto box = mp.mesh.computeBoundingBox( params.xf );
    const float safeZ = std::max( box.max.z + 10.0f * params.millRadius, params.safeZ );
    float currentZ = 0;

    const Vector3f normal = Vector3f::plusZ();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min.z ) / params.sectionStep ) );

    res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );

    MeshEdgePoint prevEdgePoint;

    const float critTransitionLengthSq = params.critTransitionLength * params.critTransitionLength;

    std::vector<PlaneSections> sections = extractAllSections( mesh, mp, box, Axis::Z, params.sectionStep, steps, params.bypassDir, subprogress( params.cb, 0.25f, 0.5f ) );
    if ( sections.empty() )
        return unexpectedOperationCanceled();

    const auto sbp = subprogress( params.cb, 0.5f, 1.0f );

    float lastFeed = 0;

    Vector3f lastPoint;
    // if the last point is equal to parameter, do nothing
    // otherwise add new move
    const auto addPoint = [&] ( const Vector3f& point )
    {
        if ( lastPoint == point )
            return;

        if ( lastFeed == params.baseFeed )
        {
                res.commands.push_back( { .x = point.x, .y = point.y } );
        }
        else
        {
            res.commands.push_back( { .feed = params.baseFeed, .x = point.x, .y = point.y } );
            lastFeed = params.baseFeed;
        }

        lastPoint = point;
    };

    for ( int step = 0; step < steps; ++step )
    {
        if ( !reportProgress( sbp, float( step ) / steps ) )
            return unexpectedOperationCanceled();

        auto& commands = res.commands;

        for ( const auto& section : sections[step] )
        {   
            if ( section.size() < 2 )
                continue;

            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, section );
            const auto contours = polyline.contours().front();

            auto nearestPointIt = section.begin();
            auto nextEdgePointIt = section.begin();
            float minDistSq = FLT_MAX;            

            if ( prevEdgePoint.e.valid() && !mp.region )
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

                const float sectionStepSq = params.sectionStep * params.sectionStep;
                const auto nearestPoint = mesh.edgePoint( *nearestPointIt );
                nextEdgePointIt = nearestPointIt;
                do
                {
                    std::next( nextEdgePointIt ) != section.end() ? ++nextEdgePointIt : nextEdgePointIt = section.begin();
                } while ( nextEdgePointIt != nearestPointIt && ( mesh.edgePoint( *nextEdgePointIt ) - nearestPoint ).lengthSq() < sectionStepSq );
            }

            const auto pivotIt = contours.begin() + std::distance( section.begin(), nextEdgePointIt );

            if ( !prevEdgePoint.e.valid() || minDistSq > critTransitionLengthSq )
            {
                transitOverSafeZ( pivotIt, res, params, safeZ, currentZ, lastFeed );               
            }
            else
            {
                const auto sp = computeSurfacePath( mesh, prevEdgePoint, *nextEdgePointIt );
                if ( sp.has_value() && !sp->empty() )
                {
                    if ( sp->size() == 1 )
                    {
                        const auto p = mesh.edgePoint( sp->front() );
                        res.commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
                    }
                    else
                    {
                        Polyline3 transit;
                        transit.addFromSurfacePath( mesh, *sp );
                        const auto transitContours = transit.contours().front();
                        for ( const auto& p : transitContours )
                            commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
                    }
                }

                commands.push_back( { .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );
            }

            auto startIt = pivotIt + 1;

            for ( auto it = startIt; it < contours.end(); ++it )
            {
                addPoint( *it );
            }

            for ( auto it = contours.begin() + 1; it < pivotIt + 1; ++it )
            {
                addPoint( *it );
            }

            prevEdgePoint = *nextEdgePointIt;
            currentZ = pivotIt->z;
        }
    }

    if ( !reportProgress( params.cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}


Expected<ToolPathResult, std::string> constantCuspToolPath( const MeshPart& mp, const ConstantCuspParams& params )
{
    auto preprocessedMesh = preprocessMesh( mp.mesh, params, true );
    if ( !preprocessedMesh )
        return unexpected( preprocessedMesh.error() );

    ToolPathResult  res{ .modifiedMesh = std::move( *preprocessedMesh ) };
    
    const auto& mesh = res.modifiedMesh;
    const auto box = mp.mesh.computeBoundingBox( params.xf );

    const Vector3f normal = Vector3f::plusZ();
    float minZ = box.min.z + params.sectionStep;
    float safeZ = std::max( box.max.z + params.millRadius, params.safeZ );

    const auto undercutPlane = MR::Plane3f::fromDirAndPt( normal, { 0.0f, 0.0f, minZ } );
    
    //compute the lowest contour that might be processed
    const auto undercutSection = extractPlaneSections( mesh, undercutPlane ).front();
    Polyline3 undercutPolyline;
    undercutPolyline.addFromSurfacePath( mesh, undercutSection );
    const auto undercutContour = undercutPolyline.contours().front();

    // if there are multiple independent zones selected we need to process them separately
    const auto processZone = [&] ( const SurfacePath& bounds, Vector3f lastPoint, ProgressCallback cb ) -> bool
    {
        //compute isolines based on the start point or the bounding contour
        std::vector<IsoLines> isoLines = extractAllIsolines( mesh, bounds, params.sectionStep, params.bypassDir, subprogress( cb, 0.0f, 0.4f ) );

        if ( isoLines.empty() )
            return false;

        //go to the start point through safe height
        res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
        lastPoint.z = safeZ;
        float lastFeed = 0.0f;

        MeshEdgePoint prevEdgePoint;

        std::optional<Vector3f> startSkippedRegion;
        
        // skip point if it is the same as the last one
        const auto addPointToTheToolPath = [&] ( const Vector3f& p )
        {
            if ( p == lastPoint )
                return;

            res.commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
            lastPoint = p;
        };

        //returns the nearest point in the contour to the given point
        const auto findNearestPoint = [] ( const Contour3f& contour, const Vector3f& p )
        {
            auto res = contour.begin();
            if ( res == contour.end() )
                return res;

            float minDistSq = ( p - *res ).lengthSq();

            for ( auto it = std::next( contour.begin() ); it != contour.end(); ++it )
            {
                const float distSq = ( p - *it ).lengthSq();
                if ( distSq < minDistSq )
                {
                    minDistSq = distSq;
                    res = it;
                }
            }

            return res;
        };

        // compute bitset of the vertices that are not belonged to the undercut
        VertBitSet noUndercutVertices( mesh.points.size() );
        BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
        {
            noUndercutVertices.set( v, mesh.points[v].z >= minZ );
        } );

        ToolPathParams paramsCopy = params;
        paramsCopy.critTransitionLength = 0;

        //append a part of a contour to the result
        const auto addSliceToTheToolPath = [&] ( const Contour3f::const_iterator startIt, Contour3f::const_iterator endIt )
        {
            for ( auto it = startIt; it < endIt; ++it )
            {
                //skip point if it lies in the undercut
                bool isPointAppropriate = ( it->z >= minZ );
                if ( !isPointAppropriate )
                {
                    if ( !startSkippedRegion )
                        startSkippedRegion = *it;

                    continue;
                }
                if ( startSkippedRegion )
                {
                    transitOverSafeZ( it, res, paramsCopy, safeZ, res.commands.back().z, lastFeed );
                    startSkippedRegion.reset();
                    continue;
                }
                
                addPointToTheToolPath( *it );
            }
        };

        if ( !reportProgress( cb, 0.5f ) )
            return false;

        const auto sbp = subprogress( cb, 0.5f, 1.0f );
        const size_t numIsolines = isoLines.size();

        if ( params.fromCenterToBoundary )
            std::reverse( isoLines.begin(), isoLines.end() );
        
        // go on in the inverse order (from the highest isoline to the lowest )
        for ( size_t i = 0; i < numIsolines; ++i )
        {
            if ( sbp && !sbp( float( i ) / numIsolines ) )
                return false;

            if ( isoLines[i].empty() )
                continue;

            for ( const auto& surfacePath : isoLines[i] )
            {
                Polyline3 polyline;
                polyline.addFromSurfacePath( mesh, surfacePath );
                const auto contour = polyline.contours().front();

                auto nearestPointIt = surfacePath.begin();
                float minDistSq = FLT_MAX;

                if ( mp.region )
                {
                    //skip isoline if nore than half of its point are not lying in the selection
                    size_t pointsInside = 0;
                    for ( const auto& p : contour )
                    {
                        auto mpr = mp.mesh.projectPoint( p );
                        if ( mpr && mp.region->test( mp.mesh.topology.left( mpr->mtp.e ) ) )
                            ++pointsInside;
                    }

                    if ( pointsInside < ( contour.size() >> 1 ) )
                        continue;
                }

                // find the nearest point to the last processed one
                if ( prevEdgePoint.e.valid() )
                {
                    for ( auto it = surfacePath.begin(); it < surfacePath.end(); ++it )
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

                // make severals steps ahead for smoother transit
                Vector3f tmp;
                bool skipPoint = false;
                do
                {
                    std::next( nextEdgePointIt ) != surfacePath.end() ? ++nextEdgePointIt : nextEdgePointIt = surfacePath.begin();
                    tmp = mesh.edgePoint( *nextEdgePointIt );
                    skipPoint = ( tmp.z < minZ );
                } while ( nextEdgePointIt != nearestPointIt && ( ( ( tmp - nearestPoint ).lengthSq() < sectionStepSq ) || skipPoint ) );

                const auto pivotIt = contour.begin() + std::distance( surfacePath.begin(), nextEdgePointIt );
                if ( skipPoint )
                    continue;

                if ( !prevEdgePoint.e.valid() )
                {
                    // go through the safe height to the first point
                    res.commands.push_back( { .type = MoveType::FastLinear, .x = pivotIt->x, .y = pivotIt->y } );

                    if ( safeZ - pivotIt->z > params.plungeLength )
                    {
                        const float zPlunge = pivotIt->z + params.plungeLength;
                        res.commands.push_back( { .type = MoveType::FastLinear, .z = zPlunge } );
                    }

                    res.commands.push_back( { .feed = params.plungeFeed, .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );
                }
                else
                {
                    // go along the undercut if the part of section is below
                    const auto p1 = mesh.edgePoint( prevEdgePoint );
                    const auto p2 = mesh.edgePoint( *nextEdgePointIt );
                    if ( p1.z == minZ )
                    {
                        const auto sectionStartIt = findNearestPoint( undercutContour, p1 );
                        const auto sectionEndIt = findNearestPoint( undercutContour, p2 );

                        if ( sectionStartIt < sectionEndIt )
                        {
                            for ( auto sectionIt = sectionStartIt; sectionIt <= sectionEndIt; ++sectionIt )
                                addPointToTheToolPath( *sectionIt );
                        }
                        else
                        {
                            for ( auto sectionIt = sectionStartIt; sectionIt < undercutContour.end(); ++sectionIt )
                                addPointToTheToolPath( *sectionIt );

                            for ( auto sectionIt = std::next( undercutContour.begin() ); sectionIt <= sectionEndIt; ++sectionIt )
                                addPointToTheToolPath( *sectionIt );
                        }

                        addPointToTheToolPath( p2 );
                    }
                    else
                    {
                        // go along the mesh in other cases
                        const auto sp = computeSurfacePath( mesh, prevEdgePoint, *nextEdgePointIt, 5, &noUndercutVertices );
                        if ( sp.has_value() && !sp->empty() )
                        {
                            if ( sp->size() == 1 )
                            {
                                addPointToTheToolPath( mesh.edgePoint( sp->front() ) );
                            }
                            else
                            {
                                Polyline3 transit;
                                transit.addFromSurfacePath( mesh, *sp );
                                const auto transitContours = transit.contours().front();
                                for ( const auto& p : transitContours )
                                    addPointToTheToolPath( p );
                            }
                        }
                    }

                    res.commands.push_back( { .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );
                }

                addSliceToTheToolPath( pivotIt + 1, contour.end() );
                addSliceToTheToolPath( contour.begin() + 1, pivotIt + 1 );
                prevEdgePoint = *nextEdgePointIt;
            }
        }

        return true;
    };
    
    //if selection is not specified then process all the vertices above the undercut
    if ( !mp.region )
    {
        if ( !processZone( undercutSection, {}, subprogress( params.cb, 0.25f, 1.0f ) ) || !reportProgress( params.cb, 1.0f ) )
            return unexpectedOperationCanceled();

        return res;
    }

    const auto edgeLoops = findLeftBoundary( mp.mesh.topology, mp.region );
    //otherwise process each selected zone separately
    const size_t loopCount = edgeLoops.size();
    for ( size_t i = 0; i < loopCount; ++i )
    {
        const auto& edgeLoop = edgeLoops[i];
        SurfacePath selectionBound;
        for ( auto edgeId : edgeLoop )
        {
            // project all the vertices in the selected contour to the modified mesh
            const auto origPoint = mp.mesh.edgePoint( MeshEdgePoint( edgeId, 0 ) );
            const auto mpr = mesh.projectPoint( origPoint );
            if ( !mpr )
                continue;

            if ( selectionBound.empty() )
            {
                selectionBound.push_back( MeshEdgePoint( mpr->mtp.e, 0 ) );
                continue;
            }
            // compute path from the previous point to the current ( there may be additional points in the modified mesh )
            const auto sp = computeSurfacePath( mesh, selectionBound.back(), MeshEdgePoint( mpr->mtp.e, 0 ) );
            if ( sp )
                selectionBound.insert( selectionBound.end(), sp->begin(), sp->end() );

            selectionBound.push_back( MeshEdgePoint( mpr->mtp.e, 0 ) );
        }
        // unite the last point with the first one
        const auto sp = computeSurfacePath( mesh, selectionBound.back(), selectionBound.front() );
        if ( sp )
            selectionBound.insert( selectionBound.end(), sp->begin(), sp->end() );

        selectionBound.push_back( selectionBound.front() );

        if ( !processZone( selectionBound,
            res.commands.empty() ? Vector3f{} : Vector3f{ res.commands.back().x, res.commands.back().y, res.commands.back().z },
            subprogress( params.cb, 0.25f + 0.75f * float( i ) / loopCount, 0.25f + 0.75f * float( i + 1 ) / loopCount ) ) )
        {
            return unexpectedOperationCanceled();
        }
    }

    return res;
}

std::vector<GCommand> replaceLineSegmentsWithCircularArcs( const std::span<GCommand>& path, float eps, float maxRadius, Axis axis )
{
    if ( path.size() < 5 )
        return {};

    std::vector<GCommand> res;

    int startIdx = 0, endIdx = 0;
    Vector2f bestArcCenter, bestArcStart, bestArcEnd;
    bool CCWrotation = false;
    double bestArcR = 0;
    for ( int i = startIdx + 2; i < path.size(); ++i )
    {
        const GCommand& d2 = path[i];
        const int middleI = ( i + startIdx ) / 2;
        const GCommand& d1 = path[middleI];

        const Vector2f p0 = project( path[startIdx], axis );
        const Vector2f p1 = project( d1, axis );
        const Vector2f p2 = project( d2, axis );

        const Vector2f dif1 = p1 - p0;
        const Vector2f dif2 = p2 - p1;

        Vector2f pCenter;
        if ( dot( dif1, dif2 ) > 0
            && calcCircleCenter( p0, p1, p2, pCenter ) )
        {
            const double rArc = ( pCenter - p0 ).length();            
            const double r2Max = sqr( rArc + eps );
            const double r2Min = sqr( rArc - eps );

            const bool ccwRotation = cross( dif1, dif2 ) > 0;

            Vector2f dirStart = rotate90( p0 - pCenter );
            Vector2f dirEnd = rotateMinus90( p2 - pCenter );
            if ( ccwRotation )
            {
                dirStart = -dirStart;
                dirEnd = -dirEnd;
            }

            bool allInTolerance = true;
            Vector2f pPrev = p0;
            for ( int k = startIdx + 1; k <= i; ++k )
            {
                const Vector2f pk = project( path[k], axis );
                double r2k = ( pCenter - pk ).lengthSq();
                const Vector2f pkMiddle = ( pk + pPrev ) * 0.5f; 
                double r2kMiddle = ( pCenter - pkMiddle ).lengthSq();
                if ( r2k < r2Min || r2k > r2Max
                    || r2kMiddle < r2Min || r2kMiddle > r2Max )
                {
                    allInTolerance = false;
                    break;
                }
                bool insideArc = dot( dirStart, pk - p0 ) >= 0 && dot( dirEnd, pk - p2 ) >= 0;
                if ( !insideArc )
                {
                    allInTolerance = false;
                    break;
                }
                pPrev = pk;
            }
            if ( allInTolerance )
            {
                endIdx = i;
                bestArcCenter = pCenter;
                bestArcStart = p0;
                bestArcEnd = p2;
                CCWrotation = ccwRotation;
                if ( axis == Axis::Y )
                    CCWrotation = !CCWrotation;
                bestArcR = rArc;

                if ( i < path.size() - 1 )
                    continue;
            }
        }

        if ( endIdx - startIdx >= 3 && bestArcR < maxRadius )
        {
            const auto& d0a = path[startIdx];
            res.push_back( {} );
            auto& d1a = res.back();

            switch ( axis )
            {
            case MR::Axis::X:
                d1a.y = path[endIdx].y;
                d1a.z = path[endIdx].z;
                d1a.arcCenter.y = bestArcCenter.x - d0a.y;
                d1a.arcCenter.z = bestArcCenter.y - d0a.z;
                break;
            case MR::Axis::Y:
                d1a.x = path[endIdx].x;
                d1a.z = path[endIdx].z;
                d1a.arcCenter.x = bestArcCenter.x - d0a.x;
                d1a.arcCenter.z = bestArcCenter.y - d0a.z;
                break;
            case MR::Axis::Z:
                d1a.x = path[endIdx].x;
                d1a.y = path[endIdx].y;
                d1a.arcCenter.x = bestArcCenter.x - d0a.x;
                d1a.arcCenter.y = bestArcCenter.y - d0a.y;
                break;
            default:
                assert( false );
                break;
            }
            
            d1a.type = CCWrotation ? MoveType::ArcCCW : MoveType::ArcCW;

            startIdx = endIdx;
            i = startIdx + 1;
        }
        else
        {
            ++startIdx;
            endIdx = startIdx;
            res.push_back( path[startIdx] );
            i = startIdx + 1;
        }
    }

    for ( size_t i = endIdx; i < path.size(); ++i )
        res.push_back( path[i] );

    return res;
}

void interpolateArcs( std::vector<GCommand>& commands, const ArcInterpolationParams& params, Axis axis )
{
    const ArcPlane arcPlane = ( axis == Axis::X ) ? ArcPlane::YZ :
        ( axis == Axis::Y ) ? ArcPlane::XZ :
        ArcPlane::XY;

    commands.insert( commands.begin(), { .arcPlane = arcPlane } );
    size_t startIndex = 1u;

    while ( startIndex < commands.size() )
    {
        while ( startIndex != commands.size() && ( commands[startIndex].type != MoveType::Linear || std::isnan( coord( commands[startIndex], axis ) ) ) )
            ++startIndex;

        if ( ++startIndex >= commands.size() )
            return;

        auto endIndex = startIndex + 1;
        while ( endIndex != commands.size() && std::isnan( coord( commands[endIndex], axis ) ) )
            ++endIndex;

        const size_t segmentSize = endIndex - startIndex;
        const auto interpolatedSegment = replaceLineSegmentsWithCircularArcs( std::span<GCommand>( &commands[startIndex], segmentSize ), params.eps, params.maxRadius, axis );
        if ( interpolatedSegment.empty() )
        {
            startIndex = endIndex;
            continue;
        }

        if ( interpolatedSegment.size() != segmentSize )
        {
            commands.erase( commands.begin() + startIndex + 1, commands.begin() + endIndex );
            commands.insert( commands.begin() + startIndex + 1, interpolatedSegment.begin(), interpolatedSegment.end() );
        }

        startIndex = startIndex + interpolatedSegment.size() + 1;
    }
}

std::shared_ptr<ObjectGcode> exportToolPathToGCode( const std::vector<GCommand>& commands )
{
    auto gcodeSource = std::make_shared<std::vector<std::string>>();

    for ( const auto& command : commands )
    {
        std::ostringstream gcode;
        gcode << "G";
        gcode << ( ( command.arcPlane != ArcPlane::None ) ? int( command.arcPlane ) : int( command.type ) );

        if ( !std::isnan( command.x ) )
            gcode << " X" << command.x;

        if ( !std::isnan( command.y ) )
            gcode << " Y" << command.y;

        if ( !std::isnan( command.z ) )
            gcode << " Z" << command.z;

        if ( !std::isnan( command.arcCenter.x ) )
            gcode << " I" << command.arcCenter.x;

        if ( !std::isnan( command.arcCenter.y ) )
            gcode << " J" << command.arcCenter.y;

        if ( !std::isnan( command.arcCenter.z ) )
            gcode << " K" << command.arcCenter.z;

        if ( !std::isnan( command.feed ) )
            gcode << " F" << command.feed;

        gcode << std::endl;
        gcodeSource->push_back( gcode.str() );
    }

    auto res = std::make_shared<ObjectGcode>();
    res->setGcodeSource( gcodeSource );
    res->setName( "Tool Path" );
    res->setLineWidth( 1.0f );
    return res;
}

float distSqrToLineSegment( const MR::Vector2f p, const MR::Vector2f& seg0, const MR::Vector2f& seg1 )
{
    const auto segDir = seg1 - seg0;
    const auto len2 = segDir.lengthSq();
    constexpr float epsSq = std::numeric_limits<float>::epsilon() * std::numeric_limits<float>::epsilon();
    if ( len2 <  epsSq )
    {
        return ( seg0 - p ).lengthSq();
    }

    const auto tmp = cross( p - seg0, segDir );
    return ( tmp * tmp ) / len2;
}


std::vector<GCommand> replaceStraightSegmentsWithOneLine( const std::span<GCommand>& path, float eps, float maxLength, Axis axis )
{
    if ( path.size() < 3 )
        return {};

    std::vector<GCommand> res;

    const float epsSq = eps * eps;
    const float maxLengthSq = maxLength * maxLength;
    int startIdx = 0, endIdx = 0;
    for ( int i = startIdx + 2; i < path.size(); ++i )
    {
        const auto& d0 = path[startIdx];
        const auto& d2 = path[i];

        const Vector2f p0 = project( d0, axis );
        const Vector2f p2 = project( d2, axis );

        if ( ( p0 - p2 ).lengthSq() < maxLengthSq ) // don't merge too long lines
        {
            bool allInTolerance = true;
            for ( int k = startIdx + 1; k < i; ++k )
            {
                const auto& dk = path[k];

                const Vector2f pk = project( dk, axis );
                const float dist2 = distSqrToLineSegment( pk, p0, p2 );
                if ( dist2 > epsSq )
                {
                    allInTolerance = false;
                    break;
                }
            }
            if ( allInTolerance )
            {
                endIdx = i;
                if ( i < path.size() - 1 ) // don't continue on last point, do interpolation
                    continue;
            }
        }

        res.push_back( path[endIdx] );
        startIdx = ( startIdx <= endIdx ) ? endIdx + 1 : endIdx;
        endIdx = startIdx;
        i = startIdx + 1;
    }

    for ( int i = startIdx; i < path.size(); ++i )
    {
        res.push_back( path[i] );
    }

    return res;
}

void interpolateLines( std::vector<GCommand>& commands, const LineInterpolationParams& params, Axis axis )
{
    size_t startIndex = 0u;

    while ( startIndex < commands.size() )
    {
        while ( startIndex != commands.size() && ( commands[startIndex].type != MoveType::Linear || std::isnan( coord( commands[startIndex], axis ) ) ) )
            ++startIndex;

        if ( ++startIndex >= commands.size() )
            return;

        auto endIndex = startIndex + 1;
        while ( endIndex != commands.size() && std::isnan( coord( commands[endIndex], axis ) ) &&  commands[endIndex].type == MoveType::Linear )
            ++endIndex;

        const size_t segmentSize = endIndex - startIndex;
        const auto interpolatedSegment = replaceStraightSegmentsWithOneLine( std::span<GCommand>( &commands[startIndex], segmentSize ), params.eps, params.maxLength, axis );
        if ( interpolatedSegment.empty() )
        {
            startIndex = endIndex;
            continue;
        }

        if ( interpolatedSegment.size() != segmentSize )
        {
            commands.erase( commands.begin() + startIndex + 1, commands.begin() + endIndex );
            commands.insert( commands.begin() + startIndex + 1, interpolatedSegment.begin(), interpolatedSegment.end() );
        }

        startIndex = startIndex + interpolatedSegment.size() + 1;
    }
}

}
#endif
