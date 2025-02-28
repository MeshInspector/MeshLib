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
        params_.direction = params_.direction.normalized();
        radiusSq_ = sqr( params_.toolRadius );
    }

    // prepares raw distance map
    Expected<void> init();

    // finds tool locations for each pixel and summary compensation cost for each
    Expected<void> calcCompensations();

    // apply compensation
    Expected<void> applyCompensation();
private:

    // find world tool center location applying it by its normal at given vert
    // returns Vector3f::diagonal(FLT_MAX) if invalid
    Vector3f findToolCenterAtVertId_( VertId pixelCoord );

    // returns compensated shift for given vert out of given toolCenter
    // returns -FLT_MAX if invalid
    Vector3f calcCompensationMovementInVertId_( VertId v, const Vector3f& plneToolCenter );

    // calculates summary compensation cost for given tool location
    float sumCompensationCost_( const Vector3f& toolCenter );

    Mesh& mesh_;
    CompensateRadiusParams params_;
    VertBitSet vertRegion_;
    float radiusSq_{ 0.0f };

    // 2d tree to faster search of 
    std::unique_ptr<AABBTreePoints> planeTree_;
    VertCoords planeVerts_;
    AffineXf3f toWorldXf_;
    AffineXf3f toPlaneXf_;

    // speedup caches
    Vector<Vector3f, VertId> toolCenters_; // per vertex

    // cost is summary movement required for compensation - less is better
    Vector<std::pair<float, VertId>, VertId> costs_; // <cost, id> per vert (this array will be sorted, thats why we need id as value and not only as key)
};

Expected<void> RadiusCompensator::init()
{
    MR_TIMER;
    const auto& faceRegion = &mesh_.topology.getFaceIds( params_.region );
    assert( faceRegion );
    vertRegion_ = getInnerVerts( mesh_.topology, *faceRegion );

    if ( MeshComponents::hasFullySelectedComponent( mesh_, vertRegion_ - mesh_.topology.findBoundaryVerts( &vertRegion_ ) ) )
        return unexpected( "MeshPart should not contain closed components" );

    auto [xvec, yvec] = params_.direction.perpendicular();
    toWorldXf_ = AffineXf3f::linear( Matrix3f::fromColumns( xvec, yvec, params_.direction ) );
    toPlaneXf_ = toWorldXf_.inverse();

    planeVerts_.resize( vertRegion_.endId() );
    BitSetParallelFor( vertRegion_, [&] ( VertId v )
    {
        planeVerts_[v] = to3dim( to2dim( toPlaneXf_( mesh_.points[v] ) ) );
    } );

    planeTree_ = std::make_unique<AABBTreePoints>( planeVerts_, vertRegion_ );
    return {};
}

Expected<void> RadiusCompensator::calcCompensations()
{
    MR_TIMER;
    toolCenters_.resize( vertRegion_.endId(), Vector3f::diagonal( FLT_MAX ) );
    costs_.resize( vertRegion_.endId(), std::make_pair( -FLT_MAX, VertId() ) );
    BitSetParallelFor( vertRegion_, [&] ( VertId v )
    {
        auto tc = findToolCenterAtVertId_( v );
        toolCenters_[v] = tc;
        if ( tc.x != FLT_MAX )
            costs_[v] = std::make_pair( sumCompensationCost_( tc ), v );
    } );

    tbb::parallel_sort( begin( costs_ ), end( costs_ ), [] ( const auto& l, const auto& r )
    {
        return l.first < r.first;
    } );

    if ( !reportProgress( params_.callback, 0.2f ) )
        return unexpectedOperationCanceled();

    return {};
}

Expected<void> RadiusCompensator::applyCompensation()
{
    MR_TIMER;
    MR_WRITER( mesh_ );


    Mesh cpyMesh = mesh_; // for projecting
    VertBitSet updatedVerts( vertRegion_.size() );

    auto maxAllowedShiftSq = sqr( mesh_.computeBoundingBox( params_.region ).diagonal() * 1e-2f );
    for ( int i = 1; i <= 100; ++i )
    {
        float maxShiftSq = -FLT_MAX;
        updatedVerts.reset();
        for ( auto [cost, cId] : costs_ )
        {
            if ( cId < 0  || cost == -FLT_MAX )
                continue;
            const auto& toolCenter = toolCenters_[cId];
            if ( toolCenter.x == FLT_MAX )
                continue;
            auto planeToolCenter = toPlaneXf_( toolCenter );

            findPointsInBall( *planeTree_, { .center = to3dim( to2dim( planeToolCenter ) ),.radiusSq = radiusSq_ },
                [&] ( VertId v, const Vector3f& )
            {
                auto shift = calcCompensationMovementInVertId_( v, planeToolCenter );
                if ( shift == Vector3f() )
                    return;

                auto shiftLenSq = shift.lengthSq();
                if ( shiftLenSq > maxShiftSq )
                    maxShiftSq = shiftLenSq;
                if ( shiftLenSq < maxAllowedShiftSq )
                {
                    mesh_.points[v] += shift;
                    if ( shiftLenSq < maxAllowedShiftSq * 1e-4f )
                        return;
                }
                else
                {
                    mesh_.points[v] += shift.normalized() * maxAllowedShiftSq;
                }
                updatedVerts.set( v );
            } );
        }

        auto maxShift = std::sqrt( maxShiftSq );
        spdlog::info( "MaxShit: {}", maxShift );
        if ( updatedVerts.none() )
            break;

        expand( mesh_.topology, updatedVerts, 1 );
        updatedVerts &= vertRegion_;
        relax( mesh_, { {.iterations = 2, .region = &updatedVerts,.force = 0.2f} } );

        BitSetParallelFor( updatedVerts, [&] ( VertId v )
        {
            auto proj = findSignedDistance( mesh_.points[v], cpyMesh, 4 * maxShiftSq );
            if ( proj && proj->dist > 0 )
                mesh_.points[v] = proj->proj.point;

            planeVerts_[v] = to3dim( to2dim( toPlaneXf_( mesh_.points[v] ) ) );
        } );

        if ( i % 5 == 0 )
            planeTree_ = std::make_unique<AABBTreePoints>( planeVerts_, vertRegion_ );
        else
            planeTree_->refit( planeVerts_, updatedVerts );
    }

    return {};
}

Vector3f RadiusCompensator::findToolCenterAtVertId_( VertId v )
{
    auto norm = mesh_.normal( v );
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
        if ( vecLenSq > radiusSq_ || vecLenSq == 0 )
            return {}; // fast return for updated/non-determined points
        vec = vec / std::sqrt( vecLenSq );

        auto newPos = center2d + vec * params_.toolRadius;
        return toWorldXf_.A * to3dim( newPos - point2d );
    }
    auto vec = point - planeToolCenter;
    auto vecLenSq = vec.lengthSq();
    if ( vecLenSq > radiusSq_ || vecLenSq == 0 )
        return {}; // fast return for updated/non-determined points
    vec = vec / std::sqrt( vecLenSq );
    auto newPos = planeToolCenter + vec * params_.toolRadius;
    return toWorldXf_.A * ( newPos - point );
}

float RadiusCompensator::sumCompensationCost_( const Vector3f& toolCenter )
{
    float sumCost = 0.0f;
    auto planeToolCenter = toPlaneXf_( toolCenter );
    findPointsInBall( *planeTree_, { .center = to3dim( to2dim( planeToolCenter ) ),.radiusSq = radiusSq_ },
        [&] ( VertId v, const Vector3f& )
    {
        sumCost += calcCompensationMovementInVertId_( v, planeToolCenter ).length();
    } );
    if ( sumCost < 100.0f * std::numeric_limits<float>::epsilon() * params_.toolRadius )
        return -FLT_MAX; // consider as empty

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

    return res = c.applyCompensation();
}

}