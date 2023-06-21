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

Mesh preprocessMesh( const Mesh& inputMesh, const ToolPathParams& params, const AffineXf3f* xf )
{
    OffsetParameters offsetParams;
    offsetParams.voxelSize = params.voxelSize;

    Mesh meshCopy = *offsetMesh( inputMesh, params.millRadius, offsetParams );
    if ( xf )
        meshCopy.transform( *xf );

    const Vector3f normal = Vector3f::plusZ();
    FixUndercuts::fixUndercuts( meshCopy, normal, params.voxelSize );    

    return meshCopy;
}

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

tl::expected<ToolPathResult, std::string> lacingToolPath( const Mesh& inputMesh, const ToolPathParams& params, const AffineXf3f* xf, ProgressCallback cb )
{
    ToolPathResult  res{ .modifiedMesh = preprocessMesh( inputMesh, params, xf ) };
    const auto& mesh = res.modifiedMesh;

    const auto box = mesh.getBoundingBox();
    const float safeZ = box.max.z + params.millRadius;

    const Vector3f normal = Vector3f::plusX();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min.x ) / params.sectionStep ) );

    MeshEdgePoint lastEdgePoint = {};

    for ( int step = 0; step < steps; ++step )
    {
        if ( cb && !cb( float( step ) / steps ) )
            return unexpectedOperationCanceled();

        const auto sections = extractPlaneSections( mesh, Plane3f{ plane.n, plane.d - params.sectionStep * step } );
        if ( sections.empty() )
            continue;

        Polyline3 polyline;
        const auto& section = sections[0];
        polyline.addFromSurfacePath( mesh, section );
        const auto contour = polyline.contours().front();

        if ( contour.size() < 3 )
            continue;

        auto bottomLeftIt = contour.begin();
        auto bottomRightIt = contour.begin();

        for ( auto it = std::next( contour.begin() ); it < contour.end(); ++it )
        {
            if ( it->y < bottomLeftIt->y || ( it->y == bottomLeftIt->y && it->z < bottomLeftIt->z ) )
                bottomLeftIt = it;

            if ( it->y > bottomRightIt->y || ( it->y == bottomRightIt->y && it->z < bottomRightIt->z ) )
                bottomRightIt = it;
        }

        if ( step & 1 )
        {
            if ( res.commands.empty() )
            {
                res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
                res.commands.push_back( { .type = MoveType::FastLinear, .x = bottomLeftIt->x, .y = bottomLeftIt->y } );
                res.commands.push_back( { .x = bottomLeftIt->x, .y = bottomLeftIt->y, .z = bottomLeftIt->z } );
            }
            else
            {
                const auto nextEdgePoint = section[bottomLeftIt - contour.begin()];
                addSurfacePath( res.commands, mesh, lastEdgePoint, nextEdgePoint );
            }

            if ( bottomLeftIt < bottomRightIt )
            {
                res.commands.reserve( res.commands.size() + std::distance( bottomLeftIt, bottomRightIt ) + 1 );
                for ( auto it = bottomLeftIt + 1; it <= bottomRightIt; ++it )
                {
                    res.commands.push_back( { .y = it->y, .z = it->z } );
                }
            }
            else
            {
                res.commands.reserve( res.commands.size() + std::distance( bottomLeftIt, contour.end() ) + std::distance( contour.begin(), bottomRightIt ) );
                for ( auto it = bottomLeftIt + 1; it < contour.end(); ++it )
                {
                    res.commands.push_back( { .y = it->y, .z = it->z } );
                }

                for ( auto it = std::next( contour.begin() ); it <= bottomRightIt; ++it )
                {
                    res.commands.push_back( { .y = it->y, .z = it->z } );
                }
            }

            lastEdgePoint = section[bottomRightIt - contour.begin()];
        }
        else
        {
            if ( res.commands.empty() )
            {
                res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
                res.commands.push_back( { .type = MoveType::FastLinear, .x = bottomRightIt->x, .y = bottomRightIt->y } );
                res.commands.push_back( { .x = bottomRightIt->x, .y = bottomRightIt->y, .z = bottomRightIt->z } );
            }
            else
            {
                const auto nextEdgePoint = section[bottomRightIt - contour.begin()];
                addSurfacePath( res.commands, mesh, lastEdgePoint, nextEdgePoint );
            }

            if ( bottomLeftIt < bottomRightIt )
            {
                for ( auto it = bottomRightIt - 1; it >= bottomLeftIt; --it )
                {
                    res.commands.push_back( { .y = it->y, .z = it->z } );
                }
            }
            else
            {
                for ( auto it = bottomRightIt - 1; it > contour.begin(); --it )
                {
                    res.commands.push_back( { .y = it->y, .z = it->z } );
                }

                for ( auto it = std::prev( contour.end() ); it >= bottomLeftIt; --it )
                {
                    res.commands.push_back( { .y = it->y, .z = it->z } );
                }
            }

            lastEdgePoint = section[bottomLeftIt - contour.begin()];
        }
    }

    if ( cb && !cb( 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}

tl::expected<ToolPathResult, std::string>  constantZToolPath( const Mesh& inputMesh, const ToolPathParams& params, const AffineXf3f* xf, ProgressCallback cb )
{
    ToolPathResult  res{ .modifiedMesh = preprocessMesh( inputMesh, params, xf ) };
    const auto& mesh = res.modifiedMesh;

    const auto box = mesh.getBoundingBox();
    const float safeZ = box.max.z + params.millRadius;

    const Vector3f normal = Vector3f::plusZ();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min.z ) / params.sectionStep ) );

    float currentZ = safeZ;
    res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );

    MeshEdgePoint prevEdgePoint;

    const float critTransitionLengthSq = params.critTransitionLength * params.critTransitionLength;
    bool needToRestoreBaseFeed = true;

    for ( int step = 0; step < steps; ++step )
    {
        if ( cb && !cb( float( step ) / steps ) )
            return unexpectedOperationCanceled();

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
                if ( currentZ < safeZ )
                {
                    if ( safeZ - currentZ > params.retractLength )
                    {
                        const float zRetract = currentZ + params.retractLength;
                        res.commands.push_back( { .feed = params.retractFeed, .z = zRetract } );
                        res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
                    }
                    else
                    {
                        res.commands.push_back( { .feed =params.retractFeed, .z = safeZ } );
                    }
                }

                res.commands.push_back( { .type = MoveType::FastLinear, .x = pivotIt->x, .y = pivotIt->y } );

                if ( safeZ - pivotIt->z > params.plungeLength )
                {
                    const float zPlunge = pivotIt->z + params.plungeLength;
                    res.commands.push_back( { .type = MoveType::FastLinear, .z = zPlunge } );
                }
                res.commands.push_back( { .feed = params.plungeFeed, .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );
                needToRestoreBaseFeed = true;
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
                            res.commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
                    }
                }

                res.commands.push_back( { .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );                
            }

            currentZ = pivotIt->z;
            auto startIt = pivotIt + 1;
            if ( needToRestoreBaseFeed )
            {
                res.commands.push_back( { .feed = params.baseFeed, .x = startIt->x, .y = startIt->y } );
                ++startIt;
            }

            for ( auto it = startIt; it < contours.end(); ++it )
            {
                res.commands.push_back( { .x = it->x, .y = it->y } );
            }

            for ( auto it = contours.begin() + 1; it < pivotIt + 1; ++it )
            {
                res.commands.push_back( { .x = it->x, .y = it->y } );
            }

            needToRestoreBaseFeed = false;
            prevEdgePoint = *nextEdgePointIt;
        }        
    }

    if ( cb && !cb( 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}


tl::expected<ToolPathResult, std::string> constantCuspToolPath( const Mesh& inputMesh, const ToolPathParams& params, VertId startPoint, const AffineXf3f* xf, ProgressCallback cb )
{
    ToolPathResult  res{ .modifiedMesh = preprocessMesh( inputMesh, params, xf ) };
    
    const auto& mesh = res.modifiedMesh;
    const auto box = mesh.getBoundingBox();
    const float safeZ = box.max.z + params.millRadius;
    
    if ( !startPoint.valid() )
        startPoint = findDirMax( Vector3f::plusZ(), mesh );

    const MeshTriPoint mtp( mesh.topology, startPoint );

    const auto distances = computeSurfaceDistances( mesh, mtp );
    const auto [min, max] = parallelMinMax( distances.vec_ );
    
    const size_t numIsolines = size_t( ( max - min ) / params.sectionStep ) - 1;  

    const auto& topology = mesh.topology;
    std::vector<IsoLines> isoLines( numIsolines );

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, isoLines.size() ),
                       [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            isoLines[i] = extractIsolines( topology, distances, params.sectionStep * ( i + 1  ) );
        }
    } );

    if ( cb && !cb( 0.4f ) )
        return unexpectedOperationCanceled();

    res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
    Vector3f lastPoint{ 0.0f, 0.0f, safeZ };

    MeshEdgePoint prevEdgePoint;

    std::optional<Vector3f> startUndercut;

    const auto addPointToTheToolPath = [&] ( const Vector3f& p )
    {
        if ( p == lastPoint )
            return;
      
        res.commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
        lastPoint = p;
    };

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

    const float minZ = box.min.z + params.sectionStep;

    VertBitSet noUndercutVertices( mesh.points.size() );
    tbb::parallel_for( tbb::blocked_range<VertId>( VertId{ 0 }, VertId{ noUndercutVertices.size() } ),
                      [&] ( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId i = range.begin(); i < range.end(); ++i )
        {
            noUndercutVertices.set( i, mesh.points[i].z >= minZ );
        }
    } );

    const Vector3f normal = Vector3f::plusZ();
    const auto undercutPlane = MR::Plane3f::fromDirAndPt( normal, { 0.0f, 0.0f, minZ } );
    const auto undercutSections = extractPlaneSections( mesh, undercutPlane );
    Polyline3 undercutPolyline;
    undercutPolyline.addFromSurfacePath( mesh, undercutSections[0] );
    const auto undercutContour = undercutPolyline.contours().front();

    const auto addSliceToTheToolPath = [&] ( const Contour3f::const_iterator startIt, Contour3f::const_iterator endIt )
    {
        auto it = startIt;
        while ( it < endIt )
        {
            if ( it->z >= minZ )
            {
                addPointToTheToolPath( *it++ );
                continue;
            }

            if ( !startUndercut )
                startUndercut = *it;

            while ( it < endIt && it->z < minZ )
                ++it;

            if ( it < endIt )
            {
                Vector3f endUndercut = *it;
                const auto sectionStartIt = findNearestPoint( undercutContour, *startUndercut );
                const auto sectionEndIt = findNearestPoint( undercutContour, endUndercut );
                startUndercut.reset();

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

                    addPointToTheToolPath( *it++ );
                }
            }
        }
    };

    if ( cb && !cb( 0.5f ) )
        return unexpectedOperationCanceled();

    auto sbp = subprogress( cb, 0.5f, 1.0f );
    for ( size_t i = 0; i < numIsolines; ++i )
    {
        if ( sbp && !sbp( float( i ) / numIsolines ) )
            return unexpectedOperationCanceled();

        if ( isoLines[i].empty() )
            continue;

        Polyline3 polyline;
        const auto& surfacePath = isoLines[i][0];
        polyline.addFromSurfacePath( mesh, surfacePath );
        const auto contour = polyline.contours().front();

        auto nearestPointIt = surfacePath.begin();
        float minDistSq = FLT_MAX;

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

        Vector3f tmp;

        do
        {
            std::next( nextEdgePointIt ) != surfacePath.end() ? ++nextEdgePointIt : nextEdgePointIt = surfacePath.begin();
            tmp = mesh.edgePoint( *nextEdgePointIt );
        } while ( nextEdgePointIt != nearestPointIt && ( ( ( tmp - nearestPoint ).lengthSq() < sectionStepSq ) || ( tmp.z < minZ ) ) );

        const auto pivotIt = contour.begin() + std::distance( surfacePath.begin(), nextEdgePointIt );
        if ( pivotIt->z < minZ )
            continue;

        if ( i == 0 )
        {
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

    if ( cb && !cb( 1.0f ) )
        return unexpectedOperationCanceled();

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

        const Vector2f p0 = path[startIdx].project( axis );
        const Vector2f p1 = d1.project( axis );
        const Vector2f p2 = d2.project( axis );

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
                const Vector2f pk = path[k].project( axis );
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
                d1a.j = bestArcCenter.x - d0a.y;
                d1a.k = bestArcCenter.y - d0a.z;
                break;
            case MR::Axis::Y:
                d1a.x = path[endIdx].x;
                d1a.z = path[endIdx].z;
                d1a.i = bestArcCenter.x - d0a.x;
                d1a.k = bestArcCenter.y - d0a.z;
                break;
            case MR::Axis::Z:
                d1a.x = path[endIdx].x;
                d1a.y = path[endIdx].y;
                d1a.i = bestArcCenter.x - d0a.x;
                d1a.j = bestArcCenter.y - d0a.y;
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
            res.push_back( path[startIdx] );
            i = startIdx + 1;
        }
    }

    for ( size_t i = endIdx + 1; i < path.size(); ++i )
        res.push_back( path[i] );

    return res;
}

void interpolateArcs( std::vector<GCommand>& commands, const ArcInterpolationParams& params, Axis axis )
{
    const MoveType planeSelection = ( axis == Axis::X ) ? MoveType::PlaneSelectionYZ :
        ( axis == Axis::Y ) ? MoveType::PlaneSelectionXZ :
        MoveType::PlaneSelectionXY;

    commands.insert( commands.begin(), { .type = planeSelection } );
    size_t startIndex = 1u;

    while ( startIndex < commands.size() )
    {
        while ( startIndex != commands.size() && ( commands[startIndex].type != MoveType::Linear || std::isnan( commands[startIndex].coord( axis ) ) ) )
            ++startIndex;

        if ( startIndex == commands.size() )
            return;

        auto endIndex = startIndex + 1;
        while ( endIndex != commands.size() && std::isnan( commands[endIndex].coord( axis ) ) )
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
        gcode << "G" << int( command.type );

        if ( !std::isnan( command.x ) )
            gcode << " X" << command.x;

        if ( !std::isnan( command.y ) )
            gcode << " Y" << command.y;

        if ( !std::isnan( command.z ) )
            gcode << " Z" << command.z;

        if ( !std::isnan( command.i ) )
            gcode << " I" << command.i;

        if ( !std::isnan( command.j ) )
            gcode << " J" << command.j;

        if ( !std::isnan( command.k ) )
            gcode << " K" << command.k;

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
    for ( int i = startIdx + 1; i < path.size(); ++i )
    {
        const auto& d0 = path[startIdx];
        const auto& d2 = path[i];

        const Vector2f p0 = d0.project( axis );
        Vector2f p2 = d2.project( axis );

        bool canInterpolate = ( p0 - p2 ).lengthSq() < maxLengthSq // don't merge too long lines
            && d2.type == MoveType::Linear; // don't merge arcs

        if ( canInterpolate )
        {
            bool allInTolerance = true;
            for ( int k = startIdx + 1; k < i; ++k )
            {
                const auto& dk = path[k];

                if ( dk.type != MoveType::Linear ) // don't merge arcs
                {
                    allInTolerance = false;
                    break;
                }

                const Vector2f pk = dk.project( axis );
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
        startIdx = endIdx;
        i = startIdx;
    }

    return res;
}

void interpolateLines( std::vector<GCommand>& commands, const LineInterpolationParams& params, Axis axis )
{
    size_t startIndex = 0u;

    while ( startIndex < commands.size() )
    {
        while ( startIndex != commands.size() && ( commands[startIndex].type != MoveType::Linear || std::isnan( commands[startIndex].coord( axis ) ) ) )
            ++startIndex;

        if ( startIndex == commands.size() )
            return;

        auto endIndex = startIndex + 1;
        while ( endIndex != commands.size() && std::isnan( commands[endIndex].coord( axis ) ) )
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

float GCommand::coord( Axis axis ) const
{
    return ( axis == Axis::X ) ? x :
        ( axis == Axis::Y ) ? y : z;
}

Vector2f GCommand::project( Axis axis ) const
{
    return ( axis == Axis::X ) ? Vector2f{ y, z } :
        ( axis == Axis::Y ) ? Vector2f{ x, z } : Vector2f{ x, y };
}
}
#endif
