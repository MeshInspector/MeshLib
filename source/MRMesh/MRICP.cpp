#include "MRICP.h"
#include "MRMesh.h"
#include "MRAligningTransform.h"
#include "MRMeshNormals.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MRQuaternion.h"
#include "MRBestFit.h"
#include "MRBitSetParallelFor.h"
#include <numeric>

namespace MR
{

void setupPairs( PointPairs & pairs, const VertBitSet& srcSamples )
{
    pairs.vec.clear();
    pairs.vec.reserve( srcSamples.count() );
    for ( auto id : srcSamples )
        pairs.vec.emplace_back().srcVertId = id;
    pairs.active.clear();
}

size_t deactivateFarPairs( PointPairs & pairs, float maxDistSq )
{
    size_t cnt0 = pairs.active.count();
    BitSetParallelFor( pairs.active, [&]( size_t i )
    {
        if ( pairs.vec[i].distSq > maxDistSq )
            pairs.active.reset( i );
    } );
    return cnt0 - pairs.active.count();
}


ICP::ICP( const MeshOrPoints& flt, const MeshOrPoints& ref, const AffineXf3f& fltXf, const AffineXf3f& refXf,
    const VertBitSet& fltSamples, const VertBitSet& refSamples )
    : flt_( flt )
    , ref_( ref )
{
    setXfs( fltXf, refXf );
    MR::setupPairs( flt2refPairs_, fltSamples );
    MR::setupPairs( ref2fltPairs_, refSamples );
}

ICP::ICP( const MeshOrPoints& flt, const MeshOrPoints& ref, const AffineXf3f& fltXf, const AffineXf3f& refXf,
    float samplingVoxelSize )
    : flt_( flt )
    , ref_( ref )
{
    setXfs( fltXf, refXf );
    samplePoints( samplingVoxelSize );
}

void ICP::setXfs( const AffineXf3f& fltXf, const AffineXf3f& refXf )
{
    refXf_ = refXf;
    setFloatXf( fltXf );
}

void ICP::setFloatXf( const AffineXf3f& fltXf )
{
    fltXf_ = fltXf;
}

AffineXf3f ICP::autoSelectFloatXf()
{
    MR_TIMER

    auto bestFltXf = fltXf_;
    float bestDist = getMeanSqDistToPoint();

    PointAccumulator refAcc;
    ref_.accumulate( refAcc );
    const auto refBasisXfs = refAcc.get4BasicXfs3f();

    PointAccumulator floatAcc;
    flt_.accumulate( floatAcc );
    const auto floatBasisXf = floatAcc.getBasicXf3f();

    // TODO: perform computations in parallel by calling free functions to measure the distance
    for ( const auto & refBasisXf : refBasisXfs )
    {
        auto fltXf = refXf_ * refBasisXf * floatBasisXf.inverse();
        setFloatXf( fltXf );
        updatePointPairs();
        const float dist = getMeanSqDistToPoint();
        if ( dist < bestDist )
        {
            bestDist = dist;
            bestFltXf = fltXf;
        }
    }
    setFloatXf( bestFltXf );
    return bestFltXf;
}

void ICP::sampleFltPoints( float samplingVoxelSize )
{
    setupPairs( flt2refPairs_, *flt_.pointsGridSampling( samplingVoxelSize ) );
}

void ICP::sampleRefPoints( float samplingVoxelSize )
{
    setupPairs( ref2fltPairs_, *ref_.pointsGridSampling( samplingVoxelSize ) );
}

void ICP::updatePointPairs()
{
    MR_TIMER
    MR::updatePointPairs( flt2refPairs_, flt_, fltXf_, ref_, refXf_, prop_.cosTreshold, prop_.distThresholdSq, prop_.mutualClosest );
    MR::updatePointPairs( ref2fltPairs_, ref_, refXf_, flt_, fltXf_, prop_.cosTreshold, prop_.distThresholdSq, prop_.mutualClosest );
    deactivatefarDistPairs_();
}

std::string getICPInfoLine( int iterations, ICPExitType exitType )
{
    std::string result = "Performed " + std::to_string( iterations ) + " iterations.\n";
    switch ( exitType )
    {
    case MR::ICPExitType::NotFoundSolution:
        result += "No solution found.";
        break;
    case MR::ICPExitType::MaxIterations:
        result += "Limit of iterations reached.";
        break;
    case MR::ICPExitType::MaxBadIterations:
        result += "No improvement iterations limit reached.";
        break;
    case MR::ICPExitType::StopMsdReached:
        result += "Required mean square deviation reached.";
        break;
    case MR::ICPExitType::NotStarted:
    default:
        result = "Not started yet.";
        break;
    }
    return result;
}

void updatePointPairs( PointPairs & pairs,
    const MeshOrPoints & src, const AffineXf3f & srcXf,
    const MeshOrPoints& tgt, const AffineXf3f& tgtXf,
    float cosTreshold, float distThresholdSq, bool mutualClosest )
{
    MR_TIMER
    const auto src2tgtXf = tgtXf.inverse() * srcXf;
    const auto tgt2srcXf = srcXf.inverse() * tgtXf;

    const VertCoords& srcPoints = src.points();
    const VertCoords& tgtPoints = tgt.points();

    const auto srcNormals = src.normals();
    const auto tgtNormals = tgt.normals();

    const auto srcWeights = src.weights();
    const auto srcLimProjector = src.limitedProjector();
    const auto tgtLimProjector = tgt.limitedProjector();

    pairs.active.clear();
    pairs.active.resize( pairs.vec.size(), true );

    // calculate pairs
    BitSetParallelForAll( pairs.active, [&] ( size_t idx )
    {
        auto & res = pairs.vec[idx];
        const auto p0 = srcPoints[res.srcVertId];
        const auto pt = src2tgtXf( p0 );

        MeshOrPoints::ProjectionResult prj;
        if ( res.tgtCloseVert )
        {
            // start with old closest point ...
            prj.point = tgtPoints[res.tgtCloseVert];
            if ( tgtNormals )
                prj.normal = tgtNormals( res.tgtCloseVert );
            prj.isBd = res.tgtOnBd;
            prj.distSq = ( pt - prj.point ).lengthSq();
            prj.closestVert = res.tgtCloseVert;
        }
        // ... and try to find only closer one
        tgtLimProjector( pt, prj );
        const auto p1 = prj.point;

        // save the result
        PointPair vp = res;
        vp.distSq = prj.distSq;
        vp.weight = srcWeights ? srcWeights( vp.srcVertId ) : 1.0f;
        vp.tgtCloseVert = prj.closestVert;
        vp.srcPoint = srcXf( p0 );
        vp.tgtPoint = tgtXf( p1 );
        vp.tgtNorm = prj.normal ? ( tgtXf.A * prj.normal.value() ).normalized() : Vector3f();
        vp.srcNorm = srcNormals ? ( srcXf.A * srcNormals( vp.srcVertId ) ).normalized() : Vector3f();
        vp.normalsAngleCos = ( prj.normal && srcNormals ) ? dot( vp.tgtNorm, vp.srcNorm ) : 1.0f;
        vp.tgtOnBd = prj.isBd;
        res = vp;
        if ( prj.isBd || vp.normalsAngleCos < cosTreshold || vp.distSq > distThresholdSq )
        {
            pairs.active.reset( idx );
            return;
        }
        if ( mutualClosest )
        {
            // keep prj.distSq
            prj.closestVert = res.srcVertId;
            srcLimProjector( tgt2srcXf( p1 ), prj );
            if ( prj.closestVert != res.srcVertId )
                pairs.active.reset( idx );
        }
    } );
}

void ICP::deactivatefarDistPairs_()
{
    MR_TIMER

    for ( int i = 0; i < 3; ++i )
    {
        const auto avgDist = getMeanSqDistToPoint();
        const auto maxDistSq = sqr( prop_.farDistFactor * avgDist );
        if ( maxDistSq >= prop_.distThresholdSq )
            break;

        if ( MR::deactivateFarPairs( flt2refPairs_, maxDistSq ) +
             MR::deactivateFarPairs( ref2fltPairs_, maxDistSq ) <= 0 )
            break; // nothing was deactivated
    }
}

bool ICP::p2ptIter_()
{
    MR_TIMER
    PointToPointAligningTransform p2pt;
    for ( size_t idx : flt2refPairs_.active )
    {
        const auto& vp = flt2refPairs_.vec[idx];
        p2pt.add( vp.srcPoint, vp.tgtPoint, vp.weight );
    }
    for ( size_t idx : ref2fltPairs_.active )
    {
        const auto& vp = ref2fltPairs_.vec[idx];
        p2pt.add( vp.tgtPoint, vp.srcPoint, vp.weight );
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

    if (std::isnan(res.b.x)) //nan check
        return false;
    setFloatXf( res * fltXf_ );
    return true;
}

bool ICP::p2plIter_()
{
    MR_TIMER
    Vector3f centroidRef;
    int activeCount = 0;
    for ( size_t idx : flt2refPairs_.active )
    {
        const auto& vp = flt2refPairs_.vec[idx];
        centroidRef += vp.tgtPoint;
        centroidRef += vp.srcPoint;
        ++activeCount;
    }
    for ( size_t idx : ref2fltPairs_.active )
    {
        const auto& vp = ref2fltPairs_.vec[idx];
        centroidRef += vp.tgtPoint;
        centroidRef += vp.srcPoint;
        ++activeCount;
    }
    if ( activeCount <= 0 )
        return false;
    centroidRef /= float(activeCount * 2);
    AffineXf3f centroidRefXf = AffineXf3f(Matrix3f(), centroidRef);

    PointToPlaneAligningTransform p2pl;
    for ( size_t idx : flt2refPairs_.active )
    {
        const auto& vp = flt2refPairs_.vec[idx];
        p2pl.add( vp.srcPoint - centroidRef, vp.tgtPoint - centroidRef, vp.tgtNorm, vp.weight );
    }
    for ( size_t idx : ref2fltPairs_.active )
    {
        const auto& vp = ref2fltPairs_.vec[idx];
        p2pl.add( vp.tgtPoint - centroidRef, vp.srcPoint - centroidRef, vp.srcNorm, vp.weight );
    }

    AffineXf3f res;
    if( prop_.icpMode == ICPMode::TranslationOnly )
    {
        res = AffineXf3f( Matrix3f(), Vector3f( p2pl.findBestTranslation() ) );
    }
    else
    {
        PointToPlaneAligningTransform::Amendment am;
        switch ( prop_.icpMode )
        {
        default:
            assert( false );
            [[fallthrough]];
        case ICPMode::RigidScale:
            am = p2pl.calculateAmendmentWithScale();
            break;
        case ICPMode::AnyRigidXf:
            am = p2pl.calculateAmendment();
            break;
        case ICPMode::OrthogonalAxis:
            am = p2pl.calculateOrthogonalAxisAmendment( Vector3d{ prop_.fixedRotationAxis } );
            break;
        case ICPMode::FixedAxis:
            am = p2pl.calculateFixedAxisAmendment( Vector3d{ prop_.fixedRotationAxis } );
            break;
        }

        const auto angle = am.rotAngles.length();
        assert( prop_.p2plAngleLimit > 0 );
        assert( prop_.p2plScaleLimit >= 1 );
        if ( angle > prop_.p2plAngleLimit || am.scale > prop_.p2plScaleLimit || prop_.p2plScaleLimit * am.scale < 1 )
        {
            // limit rotation angle and scale
            am.scale = std::clamp( am.scale, 1 / (double)prop_.p2plScaleLimit, (double)prop_.p2plScaleLimit );
            if ( angle > prop_.p2plAngleLimit )
                am.rotAngles *= prop_.p2plAngleLimit / angle;

            // recompute translation part
            am.shift = p2pl.findBestTranslation( am.rotAngles, am.scale );
        }
        res = AffineXf3f( am.rigidScaleXf() );
    }

    if (std::isnan(res.b.x)) //nan check
        return false;
    setFloatXf( centroidRefXf * res * centroidRefXf.inverse() * fltXf_ );
    return true;
}

AffineXf3f ICP::calculateTransformation()
{
    float minDist = std::numeric_limits<float>::max();
    int badIterCount = 0;
    resultType_ = ICPExitType::MaxIterations;
    for ( iter_ = 1; iter_ <= prop_.iterLimit; ++iter_ )
    {
        updatePointPairs();
        const bool pt2pt = ( prop_.method == ICPMethod::Combined && iter_ < 3 )
            || prop_.method == ICPMethod::PointToPoint;
        if ( !( pt2pt ? p2ptIter_() : p2plIter_() ) )
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
        if (curDist < minDist)
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
    return fltXf_;
}

size_t getNumActivePairs( const PointPairs & pairs )
{
    return pairs.active.count();
}

NumSum getSumSqDistToPoint( const PointPairs & pairs )
{
    NumSum res;
    for ( size_t idx : pairs.active )
    {
        const auto& vp = pairs.vec[idx];
        res.sum += vp.distSq;
        ++res.num;
    }
    return res;
}

NumSum getSumSqDistToPlane( const PointPairs & pairs )
{
    NumSum res;
    for ( size_t idx : pairs.active )
    {
        const auto& vp = pairs.vec[idx];
        auto v = dot( vp.tgtNorm, vp.tgtPoint - vp.srcPoint );
        res.sum += sqr( v );
        ++res.num;
    }
    return res;
}

void ICP::setCosineLimit(const float cos)
{
    prop_.cosTreshold = cos;
}

void ICP::setDistanceLimit(const float dist)
{
    prop_.distThresholdSq = dist * dist;
}

void ICP::setBadIterCount( const int iter )
{
    prop_.badIterStopCount = iter;
}

void ICP::setFarDistFactor(const float factor)
{
    prop_.farDistFactor = factor;
}

std::string ICP::getLastICPInfo() const
{
    return getICPInfoLine( iter_, resultType_ );
}

float ICP::getMeanSqDistToPoint() const
{
    return ( getSumSqDistToPoint( flt2refPairs_ ) + getSumSqDistToPoint( ref2fltPairs_ ) ).rootMeanSqF();
}

float ICP::getMeanSqDistToPlane() const
{
    return ( getSumSqDistToPlane( flt2refPairs_ ) + getSumSqDistToPlane( ref2fltPairs_ ) ).rootMeanSqF();
}

} //namespace MR
