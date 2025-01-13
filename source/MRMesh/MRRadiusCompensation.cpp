#include "MRRadiusCompensation.h"
#include "MRDistanceMapParams.h"
#include "MRDistanceMap.h"
#include "MRParallelFor.h"
#include "MRMesh.h"
#include "MRMakeSphereMesh.h"
#include "MRRegionBoundary.h"
#include "MRMeshComponents.h"
#include "MRPositionVertsSmoothly.h"
#include "MRBitSetParallelFor.h"
#include "MR2to3.h"
#include "MRExpandShrink.h"
#include "MRMeshRelax.h"
#include "MRMeshDelone.h"

namespace MR
{

Expected<void> compensateRadius( Mesh& mesh, const CompensateRadiusParams& params )
{
    const auto& faceRegion = mesh.topology.getFaceIds( params.region );
    VertBitSet vertRegion = getIncidentVerts( mesh.topology, faceRegion );

    if ( MeshComponents::hasFullySelectedComponent( mesh, vertRegion - mesh.topology.findBoundaryVerts( &vertRegion ) ) )
        return unexpected( "MeshPart should not contain closed components" );

    MeshPart mp = MeshPart( mesh, params.region );

    MeshToDistanceMapParams dmParams;
    if ( params.pixelSize > 0 )
        dmParams = MeshToDistanceMapParams( params.direction, Vector2f::diagonal( params.pixelSize ), mp, true );
    else
        dmParams = MeshToDistanceMapParams( params.direction, Vector2i::diagonal( 200 ), mp, true );

    if ( !reportProgress( params.callback, 0.1f ) )
        return unexpectedOperationCanceled();
    
    DistanceMapToWorld wParams = DistanceMapToWorld( dmParams );

    auto dm = computeDistanceMap( mp, dmParams, subprogress( params.callback, 0.1f, 0.2f ) );
    if ( dm.size() == 0 )
        return unexpectedOperationCanceled();

    Vector2f realPixelSize;
    realPixelSize.x = dmParams.xRange.length() / dmParams.resolution.x;
    realPixelSize.y = dmParams.yRange.length() / dmParams.resolution.y;

    Vector2i pixelsInDiameter;
    pixelsInDiameter.x = int( std::ceil( 2 * params.toolRadius / realPixelSize.x ) ) + 1;
    pixelsInDiameter.y = int( std::ceil( 2 * params.toolRadius / realPixelSize.y ) ) + 1;

    constexpr std::array<Vector2i, 9> cNeigborsOrder =
    {
        Vector2i( -1,1 ),
        Vector2i( 0,1 ),
        Vector2i( 1,1 ),
        Vector2i( 1,0 ),
        Vector2i( 1,-1 ),
        Vector2i( 0,-1 ),
        Vector2i( -1,-1 ),
        Vector2i( -1,0 ),
        Vector2i( -1,1 )
    };

    auto getToolSphereCenter = [&] ( int x0, int y0 )->std::optional<Vector3f>
    {
        if ( !dm.isValid( x0, y0 ) )
            return std::nullopt;
        auto val0 = dm.getValue( x0, y0 );
        Vector3f sumNorm;
        auto pos0 = wParams.toWorld( x0 + 0.5f, y0 + 0.5f, val0 );
        Vector3f prevPos;
        bool hasPrev{ false };
        for ( const auto& neighShift : cNeigborsOrder )
        {
            int xi = x0 + neighShift.x;
            int yi = y0 + neighShift.y;
            if ( xi < 0 || xi >= dmParams.resolution.x || yi < 0 || yi >= dmParams.resolution.x )
            {
                hasPrev = false;
                continue;
            }
            auto vali = dm.get( xi, yi );
            if ( !vali )
            {
                hasPrev = false;
                continue;
            }
            auto posi = wParams.toWorld( xi + 0.5f, yi + 0.5f, *vali );
            if ( !hasPrev )
            {
                prevPos = posi;
                hasPrev = true;
                continue;
            }
            sumNorm -= cross( posi - pos0, prevPos - pos0 ).normalized();
            prevPos = posi;
        }
        if ( sumNorm == Vector3f() )
            return std::nullopt;
        sumNorm = sumNorm.normalized();
        return pos0 + sumNorm * params.toolRadius;
    };

    std::vector<Vector3f> dmToolCenters( dm.size() );
    bool keepGoing = ParallelFor( size_t( 0 ), dm.size(), [&] ( size_t i )
    {
        auto pos = dm.toPos( i );
        auto toolPos = getToolSphereCenter( pos.x, pos.y );
        if ( !toolPos )
            dmToolCenters[i] = Vector3f::diagonal( FLT_MAX );
        else
            dmToolCenters[i] = *toolPos;
    }, subprogress( params.callback, 0.2f, 0.5f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    const auto radiusSq = sqr( params.toolRadius );
    const auto diameterSq = 4 * radiusSq;
    auto dirNormed = params.direction.normalized();
    auto getHeightAt = [&] ( int x, int y, const Vector3f& toolSphereCenter )->float
    {
        auto val = dm.getValue( x, y ); // should be OK if we got here
        auto pos = wParams.toWorld( x + 0.5f, y + 0.5f, val );
        auto rVec = pos - toolSphereCenter;
        auto projection = dot( rVec, dirNormed );
        auto distSq = rVec.lengthSq() - sqr( projection );

        if ( distSq >= radiusSq )
            return -FLT_MAX;

        auto shift = std::sqrt( radiusSq - distSq );
        return val - projection + shift;
    };

    auto newDm = dm;
    auto compensate = [&] ( int x0, int y0 )
    {
        auto val0 = dm.get( x0, y0 );
        if ( !val0 )
            return;

        auto pos0 = wParams.toWorld( x0 + 0.5f, y0 + 0.5f, 0 );
        float maxVal = *val0;
        for ( int xi = x0 - pixelsInDiameter.x; xi <= x0 + pixelsInDiameter.x; ++xi )
        {
            if ( xi < 0 || xi >= dmParams.resolution.x )
                continue;
            for ( int yi = y0 - pixelsInDiameter.y; yi <= y0 + pixelsInDiameter.y; ++yi )
            {
                if ( yi < 0 || yi >= dmParams.resolution.x )
                    continue;

                auto posi = wParams.toWorld( xi + 0.5f, yi + 0.5f, 0 );
                if ( ( posi - pos0 ).lengthSq() >= diameterSq )
                    continue;

                auto vali = dm.get( xi, yi );
                if ( !vali )
                    continue;
                if ( *val0 - *vali > params.toolRadius )
                    continue;

                auto toolCenter = dmToolCenters[dm.toIndex( { xi,yi } )];
                if ( toolCenter.x == FLT_MAX )
                    continue;
                auto height = getHeightAt( x0, y0, toolCenter );
                if ( height > maxVal )
                    maxVal = height;
            }
        }
        newDm.set( x0, y0, maxVal );
    };

    keepGoing = ParallelFor( size_t( 0 ), dm.size(), [&] ( size_t i )
    {
        auto pos = dm.toPos( i );
        compensate( pos.x, pos.y );
    }, subprogress( params.callback, 0.5f, 0.8f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    MR_WRITER( mesh );

    auto invXf = wParams.xf().inverse();
    VertBitSet bounds = vertRegion - getInnerVerts( mesh.topology, faceRegion );
    Contour3f backupBounds( bounds.count() );
    int i = 0;
    for ( auto v : vertRegion )
    {
        if ( bounds.test( v ) )
            backupBounds[i++] = mesh.points[v];
        mesh.points[v] = to3dim( to2dim( invXf( mesh.points[v] ) ) );
    }

    vertRegion -= bounds;
    MeshEqualizeTriAreasParams etParams;
    etParams.region = &vertRegion;
    etParams.iterations = std::max( int( faceRegion.count() ) / 10, 50 ); // some weird estimation
    keepGoing = equalizeTriAreas( mesh, etParams, subprogress( params.callback, 0.8f, 0.9f ) );
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    DeloneSettings dParams;
    dParams.region = &faceRegion;
    dParams.maxAngleChange = PI_F / 6;
    makeDeloneEdgeFlips( mesh, dParams, etParams.iterations * 20 );

    i = 0;
    for ( auto v : bounds )
        mesh.points[v] = backupBounds[i++];
    
    if ( !reportProgress( params.callback, 0.85f ) )
        return unexpectedOperationCanceled();
    
    keepGoing = BitSetParallelFor( vertRegion, [&] ( VertId v )
    {
        auto pos = to2dim( mesh.points[v] );
        auto value = newDm.getInterpolated( pos.x, pos.y );
        if ( !value )
            return;
        mesh.points[v] = wParams.toWorld( pos.x, pos.y, *value );
        vertRegion.reset( v );
    }, subprogress( params.callback, 0.9f, 1.0f ) );
    
    
    if ( vertRegion.any() )
        positionVertsSmoothlySharpBd( mesh, vertRegion );
    
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    return {};
}

}