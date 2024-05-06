#include "MRMultiwayICP.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPointToPointAligningTransform.h"
#include "MRPointToPlaneAligningTransform.h"
#include <algorithm>

namespace MR
{

MultiwayICP::MultiwayICP( const Vector<MeshOrPointsXf, MeshOrPointsId>& objects, float samplingVoxelSize ) :
    objs_{ objects }
{
    resamplePoints( samplingVoxelSize );
}

Vector<AffineXf3f, MeshOrPointsId> MultiwayICP::calculateTransformations()
{
    float minDist = std::numeric_limits<float>::max();
    int badIterCount = 0;
    resultType_ = ICPExitType::MaxIterations;

    for ( iter_ = 1; iter_ <= prop_.iterLimit; ++iter_ )
    {
        updatePointPairs();
        if ( iter_ == 1 && perIterationCb_ )
            perIterationCb_( 0 );

        const bool pt2pt = ( prop_.method == ICPMethod::Combined && iter_ < 3 )
            || prop_.method == ICPMethod::PointToPoint;
        
        bool res = !( pt2pt ? p2ptIter_() : p2plIter_() );

        if ( perIterationCb_ )
            perIterationCb_( iter_ );

        if ( res )
        {
            resultType_ = ICPExitType::NotFoundSolution;
            break;
        }

        const float curDist = pt2pt ? getMeanSqDistToPoint() : getMeanSqDistToPlane();
        if ( prop_.exitVal > curDist )
        {
            resultType_ = ICPExitType::StopMsdReached;
            break;
        }

        // exit if several(3) iterations didn't decrease minimization parameter
        if ( curDist < minDist )
        {
            minDist = curDist;
            badIterCount = 0;
        }
        else
        {
            if ( badIterCount >= prop_.badIterStopCount )
            {
                resultType_ = ICPExitType::MaxBadIterations;
                break;
            }
            badIterCount++;
        }
    }

    Vector<AffineXf3f, MeshOrPointsId> res;
    res.resize( objs_.size() );
    for ( int i = 0; i < objs_.size(); ++i )
        res[MeshOrPointsId( i )] = objs_[MeshOrPointsId( i )].xf;
    return res;
}

void MultiwayICP::resamplePoints( float samplingVoxelSize )
{
    MR_TIMER;
    pairsPerObj_.clear();
    pairsPerObj_.resize( objs_.size() );

    Vector<VertBitSet, MeshOrPointsId> samplesPerObj( objs_.size() );

    ParallelFor( objs_, [&] ( MeshOrPointsId ind )
    {
        const auto& obj = objs_[ind];
        samplesPerObj[ind] = *obj.obj.pointsGridSampling( samplingVoxelSize );
    } );

    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
    {
        auto& pairs = pairsPerObj_[i];
        pairs.resize( objs_.size() );
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
        {
            if ( i == j )
                continue;
            auto& thisPairs = pairs[j];
            thisPairs.vec.reserve( samplesPerObj[i].count());
            for ( auto v : samplesPerObj[i] )
                thisPairs.vec.emplace_back().srcVertId = v;
            thisPairs.active.reserve( thisPairs.vec.size() );
            thisPairs.active.clear();
        }
    }
}

float MultiwayICP::getMeanSqDistToPoint() const
{
    NumSum sum;
    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            if ( i != j )
                sum = sum + MR::getSumSqDistToPoint( pairsPerObj_[i][j] );
    return sum.rootMeanSqF();
}

float MultiwayICP::getMeanSqDistToPoint( MeshOrPointsId id ) const
{
    NumSum sum;
    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            if ( ( i == id || j == id ) && i != j )
                sum = sum + MR::getSumSqDistToPoint( pairsPerObj_[i][j] );
    return sum.rootMeanSqF();
}

float MultiwayICP::getMeanSqDistToPlane() const
{
    NumSum sum;
    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            if ( i != j )
                sum = sum + MR::getSumSqDistToPlane( pairsPerObj_[i][j] );
    return sum.rootMeanSqF();
}

float MultiwayICP::getMeanSqDistToPlane( MeshOrPointsId id ) const
{
    NumSum sum;
    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            if ( ( i == id || j == id ) && i != j )
                sum = sum + MR::getSumSqDistToPlane( pairsPerObj_[i][j] );
    return sum.rootMeanSqF();
}

size_t MultiwayICP::getNumActivePairs() const
{
    size_t num = 0;
    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            if ( i != j )
                num = num + MR::getNumActivePairs( pairsPerObj_[i][j] );
    return num;
}

size_t MultiwayICP::getNumActivePairs( MeshOrPointsId id ) const
{
    size_t num = 0;
    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            if ( ( i == id || j == id ) && i != j )
                num = num + MR::getNumActivePairs( pairsPerObj_[i][j] );
    return num;
}

std::string MultiwayICP::getStatusInfo() const
{
    return getICPStatusInfo( iter_, resultType_ );
}

void MultiwayICP::updatePointPairs()
{
    MR_TIMER;
    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            if ( i != j )
                MR::updatePointPairs( pairsPerObj_[i][j], objs_[i], objs_[j], prop_.cosTreshold, prop_.distThresholdSq, prop_.mutualClosest );
    deactivatefarDistPairs_();
}

void MultiwayICP::deactivatefarDistPairs_()
{
    MR_TIMER;

    for ( MeshOrPointsId i( 0 ); i < objs_.size(); ++i )
    {
        for ( int it = 0; it < 3; ++it )
        {
            size_t numDeactivated = 0;
            NumSum sum;
            for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            {
                if ( i == j )
                    continue;
                sum = sum + MR::getSumSqDistToPoint( pairsPerObj_[i][j] );
            }
            const auto avgDist = sum.rootMeanSqF();
            const auto maxDistSq = sqr( prop_.farDistFactor * avgDist );
            if ( maxDistSq >= prop_.distThresholdSq )
                break;
            for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
            {
                if ( i == j )
                    continue;
                numDeactivated += MR::deactivateFarPairs( pairsPerObj_[i][j], maxDistSq );
            }

            if ( numDeactivated == 0 )
                break; // nothing was deactivated
        }
    }
}

bool MultiwayICP::p2ptIter_()
{
    MR_TIMER;
    using FullSizeBool = uint8_t;
    Vector<FullSizeBool, MeshOrPointsId> valid( objs_.size() );
    ParallelFor( objs_, [&] ( MeshOrPointsId id )
    {
        PointToPointAligningTransform p2pt;
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairsPerObj_[id][j].active )
            {
                const auto& vp = pairsPerObj_[id][j].vec[idx];
                p2pt.add( vp.srcPoint, 0.5f * ( vp.srcPoint + vp.tgtPoint ), vp.weight );
            }
            for ( size_t idx : pairsPerObj_[j][id].active )
            {
                const auto& vp = pairsPerObj_[j][id].vec[idx];
                p2pt.add( vp.tgtPoint, 0.5f * ( vp.srcPoint + vp.tgtPoint ), vp.weight );
            }
        }

        AffineXf3f res;
        switch ( prop_.icpMode )
        {
        default:
            assert( false );
            [[fallthrough]];
        case ICPMode::RigidScale:
            res = AffineXf3f( p2pt.findBestRigidScaleXf() );
            break;
        case ICPMode::AnyRigidXf:
            res = AffineXf3f( p2pt.findBestRigidXf() );
            break;
        case ICPMode::OrthogonalAxis:
            res = AffineXf3f( p2pt.findBestRigidXfOrthogonalRotationAxis( Vector3d{ prop_.fixedRotationAxis } ) );
            break;
        case ICPMode::FixedAxis:
            res = AffineXf3f( p2pt.findBestRigidXfFixedRotationAxis( Vector3d{ prop_.fixedRotationAxis } ) );
            break;
        case ICPMode::TranslationOnly:
            res = AffineXf3f( Matrix3f(), Vector3f( p2pt.findBestTranslation() ) );
            break;
        }
        if ( std::isnan( res.b.x ) ) //nan check
        {
            valid[id] = FullSizeBool( false );
            return;
        }
        objs_[id].xf = res * objs_[id].xf;
        valid[id] = FullSizeBool( true );
    } );

    return std::all_of( valid.vec_.begin(), valid.vec_.end(), [] ( auto v ) { return bool( v ); } );
}

bool MultiwayICP::p2plIter_()
{
    MR_TIMER;
    using FullSizeBool = uint8_t;
    Vector<FullSizeBool, MeshOrPointsId> valid( objs_.size() );
    ParallelFor( objs_, [&] ( MeshOrPointsId id )
    {
        Vector3f centroidRef;
        int activeCount = 0;
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairsPerObj_[id][j].active )
            {
                const auto& vp = pairsPerObj_[id][j].vec[idx];
                centroidRef += ( vp.tgtPoint + vp.srcPoint ) * 0.5f;
                centroidRef += vp.srcPoint;
                ++activeCount;
            }
            for ( size_t idx : pairsPerObj_[j][id].active )
            {
                const auto& vp = pairsPerObj_[j][id].vec[idx];
                centroidRef += ( vp.tgtPoint + vp.srcPoint ) * 0.5f;
                centroidRef += vp.tgtPoint;
                ++activeCount;
            }
        }
        if ( activeCount <= 0 )
        {
            valid[id] = FullSizeBool( false );
            return;
        }
        centroidRef /= float( activeCount * 2 );
        AffineXf3f centroidRefXf = AffineXf3f( Matrix3f(), centroidRef );

        PointToPlaneAligningTransform p2pl;
        for ( MeshOrPointsId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairsPerObj_[id][j].active )
            {
                const auto& vp = pairsPerObj_[id][j].vec[idx];
                p2pl.add( vp.srcPoint - centroidRef, ( vp.tgtPoint + vp.srcPoint ) * 0.5f - centroidRef, vp.tgtNorm, vp.weight );
            }
            for ( size_t idx : pairsPerObj_[j][id].active )
            {
                const auto& vp = pairsPerObj_[j][id].vec[idx];
                p2pl.add( vp.tgtPoint - centroidRef, ( vp.tgtPoint + vp.srcPoint ) * 0.5f - centroidRef, vp.srcNorm, vp.weight );
            }
        }
        p2pl.prepare();

        AffineXf3f res = getAligningXf( p2pl, prop_.icpMode, prop_.p2plAngleLimit, prop_.p2plScaleLimit, prop_.fixedRotationAxis );
        if ( std::isnan( res.b.x ) ) //nan check
        {
            valid[id] = FullSizeBool( false );
            return;
        }
        objs_[id].xf = centroidRefXf * res * centroidRefXf.inverse() * objs_[id].xf;
        valid[id] = FullSizeBool( true );
    } );
    return std::all_of( valid.vec_.begin(), valid.vec_.end(), [] ( auto v ) { return bool( v ); } );
}

}
