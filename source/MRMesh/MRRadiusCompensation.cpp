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
#include "MRMeshDecimate.h"
#include "MRPch/MRTBB.h"
#include "MRTimer.h"
#include "MRAABBTreePoints.h"
#include "MRPointsInBall.h"
#include "MRBall.h"

namespace MR
{

class RadiusCompensator
{
public:
    RadiusCompensator( Mesh& mesh, const CompensateRadiusParams& params ):
        mesh_{ mesh }, params_{ params }
    {
        if ( params.projectToOriginalMesh )
            meshCpy_ = mesh;
        params_.direction = params_.direction.normalized();
        radiusSq_ = sqr( params_.toolRadius );
    }

    // prepares raw distance map
    Expected<void> init();

    // finds tool locations for each pixel and summary compensation cost for each
    Expected<void> calcCompensations();

    // creates compensation distance map
    Expected<void> compensateDistanceMap();

    // apply compensation
    Expected<void> applyCompensation();

    // remesh and smooth final result
    Expected<void> postprocessMesh();
private:
    /*
    // calculates weighted normal for pixel in distance map
    // outPixelWorldPos returns world position of the pixel within this argument
    Vector3f calcNormalAtPixel_( const Vector2i& pixelCoord, Vector3f* outPixelWorldPos );
    */

    // find world tool center location applying it by its normal at given vert
    // returns Vector3f::diagonal(FLT_MAX) if invalid
    Vector3f findToolCenterAtVertId_( VertId pixelCoord );

    // returns compensated shift for given vert out of given toolCenter
    // returns -FLT_MAX if invalid
    Vector3f calcCompensationMovementInVertId_( VertId v, const Vector3f& plneToolCenter );

    /*
    // calls callback for each valid pixel in tool radius around pixelCoord
    // if callback returns false iterations stops
    void iteratePixelsInRadius_( const Vector2i& pixelCoord, const std::function<bool( const Vector2i& )>& callback );
    */

    // calculates summary compensation cost for given tool location
    float sumCompensationCost_( const Vector3f& toolCenter );

    Mesh& mesh_;
    Mesh meshCpy_;
    CompensateRadiusParams params_;
    const FaceBitSet* faceRegion_{ nullptr };
    VertBitSet vertRegion_;
    float radiusSq_{ 0.0f };

    // raw distance map with no compensation
    std::unique_ptr<AABBTreePoints> planeTree_;
    AffineXf3f toWorldXf_;
    AffineXf3f toPlaneXf_;
    Vector2i pixelsInRadius_;

    // speedup caches
    Vector<Vector3f, VertId> toolCenters_; // per pixel
    std::vector<Vector3f> unprojectedPixes_; // per pixel

    // cost is (approx compensation Volume)/(approx compensation Projected Area) - less is better
    Vector<std::pair<float, VertId>, VertId> costs_; // <cost, id> per vert (this array will be sorted, thats why we need id as value and not only as key)

    DistanceMap compensatedDm_;

    // tolerance of distance map height for "touching" compensations
    static constexpr float cDMTolearance = 1e-5f;
};

Expected<void> RadiusCompensator::init()
{
    MR_TIMER;
    faceRegion_ = &mesh_.topology.getFaceIds( params_.region );
    assert( faceRegion_ );
    vertRegion_ = getIncidentVerts( mesh_.topology, *faceRegion_ );

    if ( MeshComponents::hasFullySelectedComponent( mesh_, vertRegion_ - mesh_.topology.findBoundaryVerts( &vertRegion_ ) ) )
        return unexpected( "MeshPart should not contain closed components" );

    auto [xvec, yvec] = params_.direction.perpendicular();
    toWorldXf_ = AffineXf3f::linear( Matrix3f::fromColumns( xvec, yvec, params_.direction ) );
    toPlaneXf_ = toWorldXf_.inverse();

    VertCoords planeVerts( vertRegion_.endId() );
    BitSetParallelFor( vertRegion_, [&] ( VertId v )
    {
        planeVerts[v] = to3dim( to2dim( toPlaneXf_( mesh_.points[v] ) ) );
    } );

    planeTree_ = std::make_unique<AABBTreePoints>( planeVerts, vertRegion_ );

    /*
    MeshToDistanceMapParams dmParams;
    dmParams = MeshToDistanceMapParams( params_.direction, params_.distanceMapResolution, MeshPart( mesh_, params_.region ), true );

    dm_ = computeDistanceMap( MeshPart( mesh_, params_.region ), dmParams, subprogress( params_.callback, 0.0f, 0.05f ) );

    if ( dm_.size() == 0 )
        return unexpectedOperationCanceled();

    toWorldXf_ = dmParams.xf();
    toDmXf_ = toWorldXf_.inverse();

    unprojectedPixes_.resize( dm_.size() );
    auto keepGoing = ParallelFor( unprojectedPixes_, [&] ( size_t i )
    {
        auto pos = dm_.toPos( i );
        auto worldPos = dm_.unproject( pos.x, pos.y, toWorldXf_ );
        if ( worldPos )
            unprojectedPixes_[i] = *worldPos;
    }, subprogress( params_.callback, 0.05f, 0.1f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    Vector2f realPixelSize = div( Vector2f( dmParams.xRange.length(), dmParams.yRange.length() ), Vector2f( dmParams.resolution ) );
    pixelsInRadius_ = Vector2i( div( Vector2f::diagonal( params_.toolRadius ), realPixelSize ) ) + Vector2i::diagonal( 1 );
    */
    return {};
}

Expected<void> RadiusCompensator::calcCompensations()
{
    MR_TIMER;
    toolCenters_.resize( vertRegion_.endId(), Vector3f::diagonal( FLT_MAX ) );
    costs_.resize( vertRegion_.endId(), std::make_pair( -1.0f, VertId() ) );
    BitSetParallelFor( vertRegion_, [&] ( VertId v )
    {
        auto tc = findToolCenterAtVertId_( v );
        toolCenters_[v] = tc;
        if ( tc.x != FLT_MAX )
            costs_[v] = std::make_pair( sumCompensationCost_( tc ), v );
    } );

    /*
    costs_.resize( vertRegion_.endId(), std::make_pair( -1.0f, -1 ) );
    bool keepGoing = ParallelFor( size_t( 0 ), dm_.size(), [&] ( size_t i )
    {
        auto pixelCoord = dm_.toPos( i );
        auto& toolCenter = toolCenters_[i];
        toolCenter = findToolCenterAtPixel_(pixelCoord);
        if ( toolCenter.x != FLT_MAX )
            costs_[i] = std::make_pair( sumCompensationCost_( toolCenter ), int( i ) );
    }, subprogress( params_.callback, 0.01f, 0.15f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();
    */
    tbb::parallel_sort( begin( costs_ ), end( costs_ ), [] ( const auto& l, const auto& r )
    {
        return l.first < r.first;
    } );

    if ( !reportProgress( params_.callback, 0.2f ) )
        return unexpectedOperationCanceled();

    return {};
}

Expected<void> RadiusCompensator::compensateDistanceMap()
{
    MR_TIMER;
    /*
    auto sb = subprogress( params_.callback, 0.2f, 0.5f );
    compensatedDm_ = DistanceMap( dm_.resX(), dm_.resY() );
    size_t i = 0;
    const float cTolerance = params_.toolRadius * cDMTolearance;
    for ( auto [cost, cId] : costs_ )
    {
        if ( ( ++i % 1024 == 0 ) && !reportProgress( sb, float( i ) / costs_.size() ) )
            return unexpectedOperationCanceled();

        if ( cId < 0 || cost < 0.0f )
            continue;

        const auto& toolCenter = toolCenters_[cId];
        if ( toolCenter.x == FLT_MAX )
            continue;

        auto pixelCoord = Vector2i( to2dim( toDmXf_( toolCenter ) ) );

        bool needCompensate = false;
        iteratePixelsInRadius_( pixelCoord, [&] ( const Vector2i& pixeli )->bool
        {
            needCompensate = !compensatedDm_.isValid( pixeli.x, pixeli.y );
            return !needCompensate;
        } );
        if ( !needCompensate )
            continue;

        // iterate in parallel
        int xZoneLength = ( 2 * pixelsInRadius_.x + 1 );
        int numPixesInArea = xZoneLength * ( 2 * pixelsInRadius_.y + 1 );
        ParallelFor( 0, numPixesInArea, [&] ( int i )
        {
            int xShift = ( i % xZoneLength ) - pixelsInRadius_.x;
            int yShift = ( i / xZoneLength ) - pixelsInRadius_.y;
            int xi = pixelCoord.x + xShift;
            int yi = pixelCoord.y + yShift;
            if ( xi < 0 || xi >= dm_.resX() )
                return;
            if ( yi < 0 || yi >= dm_.resY() )
                return;
            if ( !dm_.isValid( xi, yi ) )
                return;
            auto realValue = dm_.getValue( xi, yi );
            auto& compValue = compensatedDm_.getValue( xi, yi );
            auto newCompValue = realValue + calcCompensatedHeightAtPixel_( Vector2i( xi, yi ), toolCenter);
            if ( newCompValue > compValue && newCompValue + cTolerance >= realValue )
                compValue = std::max( newCompValue, realValue );
        } );
    }

    if ( !reportProgress( sb, 1.0f ) )
        return unexpectedOperationCanceled();

    ParallelFor( size_t( 0 ), dm_.size(), [&] ( size_t i )
    {
        auto& cv = compensatedDm_.getValue( i );
        auto v = dm_.getValue( i );
        if ( v > cv )
            cv = v;
    } );
    */
    return {};
}

Expected<void> RadiusCompensator::applyCompensation()
{
    MR_TIMER;
    MR_WRITER( mesh_ );

    //Mesh aggregateMesh;
    //Mesh sphere = makeUVSphere( params_.toolRadius );
    //int i = 0;
    VertBitSet updatedVerts( vertRegion_.size() );

    for ( auto [cost, cId] : costs_ )
    {
        if ( cId < 0 || cost < 0.0f )
            continue;
        const auto& toolCenter = toolCenters_[cId];
        if ( toolCenter.x == FLT_MAX )
            continue;
        auto planeToolCenter = toPlaneXf_( toolCenter );

        //if ( i < params_.distanceMapResolution.x )
        //{
        //    Mesh cpySph = sphere;
        //    cpySph.transform( AffineXf3f::translation( toolCenter ) );
        //    aggregateMesh.addMesh( cpySph );
        //}
        //++i;
        findPointsInBall( *planeTree_, { .center = to3dim( to2dim( planeToolCenter ) ),.radiusSq = sqr( params_.toolRadius ) },
            [&] ( VertId v, const Vector3f& )
        {
            auto shift = calcCompensationMovementInVertId_( v, planeToolCenter );
            if ( shift == Vector3f() )
                return;
            mesh_.points[v] += 0.1f * shift;
            if ( shift.lengthSq() > sqr( params_.toolRadius * 0.2f ) )
                updatedVerts.set( v );
        } );
    }

    expand( mesh_.topology, updatedVerts, 2 );
    updatedVerts &= vertRegion_;
    relaxKeepVolume( mesh_, { {.iterations = 5, .region = &updatedVerts,.force = 0.2f } } );

    //mesh_ = aggregateMesh;
    /*
    // transform verts into distance map space
    VertBitSet bounds = vertRegion_ - getInnerVerts( mesh_.topology, faceRegion_ );
    Contour3f backupBounds( bounds.count() );
    int i = 0;
    for ( auto v : vertRegion_ )
    {
        if ( bounds.test( v ) )
            backupBounds[i++] = mesh_.points[v];
        mesh_.points[v] = to3dim( to2dim( toDmXf_( mesh_.points[v] ) ) );
    }

    // fix inverted faces (undercuts on original mesh)
    for ( int iFlipped = 0; iFlipped < 10; ++iFlipped ) // repeat until no flipped faces left, 10 - max iters
    {
        // only fix flipped areas
        auto sumDirArea = Vector3f( mesh_.dirArea( faceRegion_ ).normalized() ); // we only care about direction

        auto flippedFaces = *faceRegion_;
        BitSetParallelFor( flippedFaces, [&] ( FaceId f )
        {
            if ( dot( mesh_.normal( f ), sumDirArea ) > 0.0f )
                flippedFaces.reset( f );
        } );
        expand( mesh_.topology, flippedFaces, 4 );
        flippedFaces &= *faceRegion_;

        if ( flippedFaces.none() )
            break;

        auto equalizingVertRegion = getInnerVerts( mesh_.topology, flippedFaces );
        assert( ( equalizingVertRegion & bounds ).none() );
        MeshEqualizeTriAreasParams etParams;
        etParams.region = &equalizingVertRegion;
        etParams.iterations = std::max( int( flippedFaces.count() ), 50 ); // some weird estimation
        bool keepGoing = equalizeTriAreas( mesh_, etParams, subprogress( params_.callback, 0.5f, 0.6f ) );
        if ( !keepGoing )
            return unexpectedOperationCanceled();

        DeloneSettings dParams;
        dParams.region = faceRegion_;
        dParams.maxAngleChange = PI_F / 6;
        makeDeloneEdgeFlips( mesh_, dParams, etParams.iterations * 20 );
    }

    i = 0;
    for ( auto v : bounds )
        mesh_.points[v] = backupBounds[i++];

    if ( !reportProgress( params_.callback, 0.65f ) )
        return unexpectedOperationCanceled();

    vertRegion_ -= bounds;
    bool keepGoing = BitSetParallelFor( vertRegion_, [&] ( VertId v )
    {
        auto pos = to2dim( mesh_.points[v] );
        auto value = compensatedDm_.getInterpolated( pos.x, pos.y );
        if ( !value )
            return;
        mesh_.points[v] = toWorldXf_( Vector3f( pos.x, pos.y, *value ) );
        vertRegion_.reset( v );
    }, subprogress( params_.callback, 0.65f, 0.8f ) );
    
    if ( vertRegion_.any() )
        positionVertsSmoothlySharpBd( mesh_, vertRegion_ );
    
    if ( !keepGoing )
        return unexpectedOperationCanceled();
    */
    return {};
}

Expected<void> RadiusCompensator::postprocessMesh()
{
    MR_TIMER;

    DeloneSettings dParams;
    dParams.region = faceRegion_;
    dParams.maxAngleChange = PI_F / 3;
    makeDeloneEdgeFlips( mesh_, dParams, int( faceRegion_->count() ) );

    if ( !reportProgress( params_.callback, 0.85f ) )
        return unexpectedOperationCanceled();

    auto edgeBounds = findRegionBoundaryUndirectedEdgesInsideMesh( mesh_.topology, *faceRegion_ );
    RemeshSettings rParams;
    rParams.finalRelaxIters = 2;
    rParams.targetEdgeLen = params_.remeshTargetEdgeLength <= 0.0f ? mesh_.averageEdgeLength() : params_.remeshTargetEdgeLength;
    rParams.region = params_.region;
    rParams.notFlippable = &edgeBounds;
    rParams.progressCallback = subprogress( params_.callback, 0.85f, params_.projectToOriginalMesh ? 0.92f : 1.0f );
    
    if ( !remesh( mesh_, rParams ) )
        return unexpectedOperationCanceled();

    if ( !params_.projectToOriginalMesh )
        return {};

    auto verts = getInnerVerts( mesh_.topology, *faceRegion_ );
    auto keepGoing = BitSetParallelFor( verts, [&] ( VertId v )
    {
        auto proj = findSignedDistance( mesh_.points[v], meshCpy_ );
        if ( !proj )
            return;
        if ( proj->dist >= 0.0f )
            return;
        mesh_.points[v] = proj->proj.point;
    }, subprogress( params_.callback, 0.92f,  1.0f ) );

    mesh_.invalidateCaches();

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    return {};
}
/*
Vector3f RadiusCompensator::calcNormalAtPixel_( const Vector2i& coord0, Vector3f* outPixelWorldPos )
{
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

    Vector3f sumNorm;
    auto i0 = dm_.toIndex( coord0 );
    if ( !dm_.isValid( i0 ) )
        return {};
    auto pos0 = unprojectedPixes_[i0];
    if ( outPixelWorldPos )
        *outPixelWorldPos = pos0;
    Vector3f prevPos;
    bool hasPrev{ false };
    for ( const auto& neighShift : cNeigborsOrder )
    {
        auto coordi = coord0 + neighShift;
        if ( coordi.x < 0 || coordi.x >= dm_.resX() || coordi.y < 0 || coordi.y >= dm_.resY() )
        {
            hasPrev = false;
            continue;
        }
        auto ii = dm_.toIndex( coordi );
        if ( !dm_.isValid( ii ) )
        {
            hasPrev = false;
            continue;
        }
        auto posi = unprojectedPixes_[ii];
        if ( !hasPrev )
        {
            prevPos = posi;
            hasPrev = true;
            continue;
        }
        auto veci = posi - pos0;
        auto vecPrev = prevPos - pos0;
        sumNorm -= ( MR::angle( veci, vecPrev ) * ( cross( veci, vecPrev ).normalized() ) );
        prevPos = posi;
    }
    if ( sumNorm == Vector3f() )
        return {};
    sumNorm = sumNorm.normalized();
    return sumNorm;
}
*/

/*
Vector3f RadiusCompensator::findToolCenterAtPixel_( const Vector2i& coord0 )
{
    Vector3f pos0;
    auto normAtPixel = calcNormalAtPixel_( coord0, &pos0 );
    if ( normAtPixel == Vector3f() )
        return Vector3f::diagonal( FLT_MAX );

    //// this block is not useful but lets keep it as comment for possible improvements
    //
    //auto normCos = dot( -normAtPixel, params_.direction );
    //if ( normCos <= params_.criticalToolAngleCos )
    //{
    //    normAtPixel += normCos * params_.direction; // plus here because we used -normAtPixel for dot product
    //    normAtPixel = normAtPixel.normalized();
    //}

    return pos0 + normAtPixel * params_.toolRadius;
}
*/

Vector3f RadiusCompensator::findToolCenterAtVertId_( VertId v )
{
    auto norm = mesh_.normal( v );
    //if ( dot( norm, params_.direction ) < 0.0f )
    //    norm = -norm;
    return mesh_.points[v] + norm * params_.toolRadius;
}

Vector3f RadiusCompensator::calcCompensationMovementInVertId_( VertId v, const Vector3f& planeToolCenter )
{
    auto point = toPlaneXf_( mesh_.points[v] );
    if ( point.z <= planeToolCenter.z )
    {
        auto point2d = to2dim( point );
        auto center2d = to2dim( planeToolCenter );
        auto vec = point2d - center2d;
        auto vecLenSq = vec.lengthSq();
        if ( vecLenSq > sqr( params_.toolRadius ) || vecLenSq == 0 )
            return {}; // fast return for updated/non-determined points
        vec = vec / std::sqrt( vecLenSq );

        auto newPos = center2d + vec * params_.toolRadius;
        return toWorldXf_.A * to3dim( newPos - point2d );
    }
    auto vec = point - planeToolCenter;
    auto vecLenSq = vec.lengthSq();
    if ( vecLenSq > sqr( params_.toolRadius ) || vecLenSq == 0 )
        return {}; // fast return for updated/non-determined points
    vec = vec / std::sqrt( vecLenSq );
    auto newPos = planeToolCenter + vec * params_.toolRadius;
    return toWorldXf_.A * ( newPos - point );
}

/*
float RadiusCompensator::calcCompensatedHeightAtPixel_( const Vector2i& pixelCoord, const Vector3f& worldToolCenter )
{
    auto pos = unprojectedPixes_[dm_.toIndex( pixelCoord )]; // should be OK not to validate if we got here
    auto rVec = pos - worldToolCenter;
    auto projection = dot( rVec, params_.direction );
    auto distSq = rVec.lengthSq() - sqr( projection );

    if ( distSq >= radiusSq_ )
        return -FLT_MAX;

    auto shift = std::sqrt( radiusSq_ - distSq );
    return shift - projection;
}
*/
/*
void RadiusCompensator::iteratePixelsInRadius_( const Vector2i& pixelCoord, const std::function<bool( const Vector2i& )>& callback )
{
    for ( int xi = pixelCoord.x - pixelsInRadius_.x; xi <= pixelCoord.x + pixelsInRadius_.x; ++xi )
    {
        if ( xi < 0 || xi >= dm_.resX() )
            continue;
        for ( int yi = pixelCoord.y - pixelsInRadius_.y; yi <= pixelCoord.y + pixelsInRadius_.y; ++yi )
        {
            if ( yi < 0 || yi >= dm_.resY() )
                continue;
            if ( !dm_.isValid( xi, yi ) )
                continue;
            if ( !callback( Vector2i( xi, yi ) ) )
                return;
        }
    }
}
*/

float RadiusCompensator::sumCompensationCost_( const Vector3f& toolCenter )
{
    float sumCost = 0.0f;
    auto planeToolCenter = toPlaneXf_( toolCenter );
    findPointsInBall( *planeTree_, { .center = to3dim( to2dim( planeToolCenter ) ),.radiusSq = sqr( params_.toolRadius ) },
        [&] ( VertId v, const Vector3f& )
    {
        sumCost += calcCompensationMovementInVertId_( v, planeToolCenter ).length();
    } );
    /*
    auto toolPixel = Vector2i( to2dim( toDmXf_( toolCenter ) ) );
    const float cTolerance = cDMTolearance * params_.toolRadius;
    iteratePixelsInRadius_( toolPixel, [&] ( const Vector2i& pixelCoord )->bool
    {
        auto value = dm_.getValue( pixelCoord.x, pixelCoord.y );
        auto height = value + calcCompensatedHeightAtPixel_( pixelCoord, toolCenter );
        if ( height + cTolerance >= value )
        {
            sumVolume += std::max( height - value, 0.0f );
            sumArea += 1.0;
        }
        return true;
    } );
    */
    if ( sumCost < 100.0f * std::numeric_limits<float>::epsilon() * params_.toolRadius )
        return -1.0f; // consider as empty
    return sumCost;
}

Expected<void> compensateRadius( Mesh& mesh, const CompensateRadiusParams& params )
{
    MR_TIMER;

    auto c = RadiusCompensator( mesh, params );

    auto res = c.init();
    if ( !res.has_value() )
        return res;

    res = c.calcCompensations();
    if ( !res.has_value() )
        return res;

    //res = c.compensateDistanceMap();
    //if ( !res.has_value() )
    //    return res;

    return res = c.applyCompensation();
    //if ( !res.has_value() )
    //    return res;
    //
    //return c.postprocessMesh();
}

}