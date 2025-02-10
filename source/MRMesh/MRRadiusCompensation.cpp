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
#include "MRMeshSave.h"

namespace MR
{

class RadiusCompensator
{
public:
    RadiusCompensator( Mesh& mesh, const CompensateRadiusParams& params ):
        mesh_{ mesh }, params_{ params }
    {
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

    // calculates weighted normal for pixel in distance map
    // outPixelWorldPos returns world position of the pixel within this argument
    Vector3f calcNormalAtPixel_( const Vector2i& pixelCoord, Vector3f* outPixelWorldPos );

    // find world tool center location applying it by its normal to distance map normal at given pixel
    // returns Vector3f::diagonal(FLT_MAX) if invalid
    Vector3f findToolCenterAtPixel_( const Vector2i& pixelCoord );

    // returns compensated value given pixel with given tool position
    // returns -FLT_MAX if invalid
    float calcCompensatedHeightAtPixel_( const Vector2i& pixelCoord, const Vector3f& worldToolCenter );

    // calls callback for each valid pixel in tool radius around pixelCoord
    // if callback returns false iterations stops
    void iteratePixelsInRadius_( const Vector2i& pixelCoord, const std::function<bool( const Vector2i& )>& callback );

    // calculates summary compensation cost for given tool location
    float sumCompensationCost_( const Vector3f& toolCenter );

    Mesh& mesh_;
    CompensateRadiusParams params_;
    const FaceBitSet* faceRegion_{ nullptr };
    VertBitSet vertRegion_;
    float radiusSq_{ 0.0f };

    // raw distance map with no compensation
    DistanceMap dm_;
    AffineXf3f toWorldXf_;
    AffineXf3f toDmXf_;
    Vector2i pixelsInRadius_;

    std::vector<Vector3f> toolCenters_; // per pixel
    // cost is (approx compensation Volume)/(approx compensation Projected Area) - less is better
    std::vector<std::pair<float,int>> costs_; // <cost, id> per pixel (this array will be sorted, thats why we need id as value and not only as key)

    DistanceMap compensatedDm_;

    // tolerance of distance map height for "touching" compensations
    static constexpr float cDMTolearance = 1e-6f;
};

Expected<void> RadiusCompensator::init()
{
    MR_TIMER;
    faceRegion_ = &mesh_.topology.getFaceIds( params_.region );
    assert( faceRegion_ );
    vertRegion_ = getIncidentVerts( mesh_.topology, *faceRegion_ );

    if ( MeshComponents::hasFullySelectedComponent( mesh_, vertRegion_ - mesh_.topology.findBoundaryVerts( &vertRegion_ ) ) )
        return unexpected( "MeshPart should not contain closed components" );

    MeshToDistanceMapParams dmParams;
    dmParams = MeshToDistanceMapParams( params_.direction, params_.distanceMapResolution, MeshPart( mesh_, params_.region ), true );

    dm_ = computeDistanceMap( MeshPart( mesh_, params_.region ), dmParams, subprogress( params_.callback, 0.0f, 0.05f ) );

    if ( dm_.size() == 0 )
        return unexpectedOperationCanceled();

    toWorldXf_ = dmParams.xf();
    toDmXf_ = toWorldXf_.inverse();

    Vector2f realPixelSize = div( Vector2f( dmParams.xRange.length(), dmParams.yRange.length() ), Vector2f( dmParams.resolution ) );
    pixelsInRadius_ = Vector2i( div( Vector2f::diagonal( params_.toolRadius ), realPixelSize ) ) + Vector2i::diagonal( 1 );

    return {};
}

Expected<void> RadiusCompensator::calcCompensations()
{
    MR_TIMER;
    toolCenters_.resize( dm_.size(), Vector3f::diagonal( FLT_MAX ) );
    costs_.resize( dm_.size(), std::make_pair( -1.0f, -1 ) );
    bool keepGoing = ParallelFor( size_t( 0 ), dm_.size(), [&] ( size_t i )
    {
        auto pixelCoord = dm_.toPos( i );
        auto& toolCenter = toolCenters_[i];
        toolCenter = findToolCenterAtPixel_(pixelCoord);
        if ( toolCenter.x != FLT_MAX )
            costs_[i] = std::make_pair( sumCompensationCost_( toolCenter ), int( i ) );
    }, subprogress( params_.callback, 0.05f, 0.15f ) );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    tbb::parallel_sort( costs_.begin(), costs_.end(), [] ( const auto& l, const auto& r )
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
    auto sb = subprogress( params_.callback, 0.2f, 0.5f );
    compensatedDm_ = DistanceMap( dm_.resX(), dm_.resY() );
    size_t i = 0;
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

        iteratePixelsInRadius_( pixelCoord, [&] ( const Vector2i& pixeli )->bool
        {
            auto realValue = dm_.getValue( pixeli.x, pixeli.y );
            auto& value = compensatedDm_.getValue( pixeli.x, pixeli.y );
            auto compValue = calcCompensatedHeightAtPixel_( pixeli, toolCenter );
            if ( compValue > value && compValue + std::abs( realValue * cDMTolearance ) >= realValue )
                value = compValue;
            return true;
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

    return {};
}

Expected<void> RadiusCompensator::applyCompensation()
{
    MR_TIMER;
    MR_WRITER( mesh_ );

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

    MeshSave::toAnySupportedFormat( mesh_, "C:\\WORK\\MODELS\\Radius_compensation\\RadiusCompensation\\RadiusCompensation\\#11 result\\debug0.mrmesh" );

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
        dParams.region = &flippedFaces;
        dParams.maxAngleChange = PI_F / 6;
        makeDeloneEdgeFlips( mesh_, dParams, etParams.iterations * 20 );
    }

    MeshSave::toAnySupportedFormat( mesh_, "C:\\WORK\\MODELS\\Radius_compensation\\RadiusCompensation\\RadiusCompensation\\#11 result\\debug1.mrmesh" );
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

    MeshSave::toAnySupportedFormat( mesh_, "C:\\WORK\\MODELS\\Radius_compensation\\RadiusCompensation\\RadiusCompensation\\#11 result\\debug2.mrmesh" );
    if ( vertRegion_.any() )
        positionVertsSmoothlySharpBd( mesh_, vertRegion_ );

    MeshSave::toAnySupportedFormat( mesh_, "C:\\WORK\\MODELS\\Radius_compensation\\RadiusCompensation\\RadiusCompensation\\#11 result\\debug3.mrmesh" );
    if ( !keepGoing )
        return unexpectedOperationCanceled();

    return {};
}

Expected<void> RadiusCompensator::postprocessMesh()
{
    MR_TIMER;
    auto edgeBounds = findRegionBoundaryUndirectedEdgesInsideMesh( mesh_.topology, *faceRegion_ );
    RemeshSettings rParams;
    rParams.finalRelaxIters = 2;
    rParams.targetEdgeLen = params_.remeshTargetEdgeLength <= 0.0f ? mesh_.averageEdgeLength() : params_.remeshTargetEdgeLength;
    rParams.region = params_.region;
    rParams.notFlippable = &edgeBounds;
    rParams.progressCallback = subprogress( params_.callback, 0.8f, 1.0f );

    if ( !remesh( mesh_, rParams ) )
        return unexpectedOperationCanceled();

    return {};
}

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
    auto pos0 = dm_.unproject( coord0.x, coord0.y, toWorldXf_ );
    if ( !pos0 )
        return {};
    if ( outPixelWorldPos )
        *outPixelWorldPos = *pos0;
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
        auto posi = dm_.unproject( coordi.x, coordi.y, toWorldXf_ );
        if ( !posi )
        {
            hasPrev = false;
            continue;
        }
        if ( !hasPrev )
        {
            prevPos = *posi;
            hasPrev = true;
            continue;
        }
        auto veci = *posi - *pos0;
        auto vecPrev = prevPos - *pos0;
        sumNorm -= ( MR::angle( veci, vecPrev ) * ( cross( veci, vecPrev ).normalized() ) );
        prevPos = *posi;
    }
    if ( sumNorm == Vector3f() )
        return {};
    sumNorm = sumNorm.normalized();
    return sumNorm;
}

Vector3f RadiusCompensator::findToolCenterAtPixel_( const Vector2i& coord0 )
{
    Vector3f pos0;
    auto normAtPixel = calcNormalAtPixel_( coord0, &pos0 );
    if ( normAtPixel == Vector3f() )
        return Vector3f::diagonal( FLT_MAX );

    // this block is not useful but lets keep it as comment for possible improvements
    /*
    auto normCos = dot( -normAtPixel, params_.direction );
    if ( normCos <= params_.criticalToolAngleCos )
    {
        normAtPixel += normCos * params_.direction; // plus here because we used -normAtPixel for dot product
        normAtPixel = normAtPixel.normalized();
    }
    */
    return pos0 + normAtPixel * params_.toolRadius;
}

float RadiusCompensator::calcCompensatedHeightAtPixel_( const Vector2i& pixelCoord, const Vector3f& worldToolCenter )
{
    auto val = dm_.getValue( pixelCoord.x, pixelCoord.y ); // should be OK if we got here
    auto pos = toWorldXf_( Vector3f( pixelCoord.x + 0.5f, pixelCoord.y + 0.5f, val ) );
    auto rVec = pos - worldToolCenter;
    auto projection = dot( rVec, params_.direction );
    auto distSq = rVec.lengthSq() - sqr( projection );

    if ( distSq >= radiusSq_ )
        return -FLT_MAX;

    auto shift = std::sqrt( radiusSq_ - distSq );
    return val - projection + shift;
}

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

float RadiusCompensator::sumCompensationCost_( const Vector3f& toolCenter )
{
    double sumVolume = 0.0;
    double sumArea = 0.0;
    auto toolPixel = Vector2i( to2dim( toDmXf_( toolCenter ) ) );
    iteratePixelsInRadius_( toolPixel, [&] ( const Vector2i& pixelCoord )->bool
    {
        auto value = dm_.getValue( pixelCoord.x, pixelCoord.y );
        auto height = calcCompensatedHeightAtPixel_( pixelCoord, toolCenter );
        if ( height + std::abs( value * cDMTolearance ) >= value )
        {
            sumVolume += ( height + std::abs( value * cDMTolearance ) - value );
            sumArea += 1.0;
        }
        return true;
    } );
    return sumArea == 0.0 ? -1.0f : float( sumVolume / sumArea );
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

    res = c.compensateDistanceMap();
    if ( !res.has_value() )
        return res;

    res = c.applyCompensation();
    if ( !res.has_value() )
        return res;

    return c.postprocessMesh();
}

}