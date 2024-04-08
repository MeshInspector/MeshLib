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

const int MAX_RESAMPLING_VOXEL_NUMBER = 500000;

namespace MR
{

namespace
{

void setupPairs( PointPairs & pairs, const VertBitSet& srcSamples )
{
    pairs.vec.clear();
    pairs.vec.reserve( srcSamples.count() );
    for ( auto id : srcSamples )
        pairs.vec.emplace_back().srcVertId = id;
    pairs.active.clear();
    pairs.active.resize( srcSamples.count(), true );
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

} // anonymous namespace

ICP::ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
    const VertBitSet& fltSamples)
    : flt_( floating )
    , ref_( reference )
{
    setXfs( fltXf, refXf );
    setupPairs( flt2refPairs_, fltSamples );
}

ICP::ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
    float floatSamplingVoxelSize )
    : flt_( floating )
    , ref_( reference )
{
    setXfs( fltXf, refXf );
    recomputeBitSet( floatSamplingVoxelSize );
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

void ICP::recomputeBitSet(const float floatSamplingVoxelSize)
{
    auto bboxDiag = flt_.computeBoundingBox().size() / floatSamplingVoxelSize;
    auto nSamples = bboxDiag[0] * bboxDiag[1] * bboxDiag[2];
    VertBitSet fltSamples;
    if (nSamples > MAX_RESAMPLING_VOXEL_NUMBER)
        fltSamples = *flt_.pointsGridSampling( floatSamplingVoxelSize * std::cbrt(float(nSamples) / float(MAX_RESAMPLING_VOXEL_NUMBER)) );
    else
        fltSamples = *flt_.pointsGridSampling( floatSamplingVoxelSize );
    setupPairs( flt2refPairs_, fltSamples );
}

void ICP::updatePointPairs()
{
    MR_TIMER
    updatePointPairs_( flt2refPairs_, flt_, fltXf_, ref_, refXf_ );
    deactivatefarDistPairs_();
}

void ICP::updatePointPairs_( PointPairs & pairs,
    const MeshOrPoints & src, const AffineXf3f & srcXf,
    const MeshOrPoints & tgt, const AffineXf3f & tgtXf )
{
    MR_TIMER
    const auto src2tgtXf = tgtXf.inverse() * srcXf;

    const VertCoords& srcPoints = src.points();
    const VertCoords& tgtPoints = tgt.points();

    const auto srcNormals = src.normals();
    const auto tgtNormals = tgt.normals();

    const auto srcWeights = src.weights();
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

        // save the result
        PointPair vp = res;
        vp.distSq = prj.distSq;
        vp.weight = srcWeights ? srcWeights( vp.srcVertId ) : 1.0f;
        vp.tgtCloseVert = prj.closestVert;
        vp.srcPoint = srcXf( p0 );
        vp.tgtPoint = tgtXf( prj.point );
        vp.tgtNorm = prj.normal ? ( tgtXf.A * prj.normal.value() ).normalized() : Vector3f();
        vp.srcNorm = srcNormals ? ( srcXf.A * srcNormals( vp.srcVertId ) ).normalized() : Vector3f();
        vp.normalsAngleCos = ( prj.normal && srcNormals ) ? dot( vp.tgtNorm, vp.srcNorm ) : 1.0f;
        vp.tgtOnBd = prj.isBd;
        res = vp;
        if ( prj.isBd || vp.normalsAngleCos < prop_.cosTreshold || vp.distSq > prop_.distThresholdSq )
            pairs.active.reset( idx );
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

        if ( deactivateFarPairs( flt2refPairs_, maxDistSq ) <= 0 )
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
    if ( activeCount <= 0 )
        return false;
    centroidRef /= float(activeCount * 2);
    AffineXf3f centroidRefXf = AffineXf3f(Matrix3f(), centroidRef);

    PointToPlaneAligningTransform p2pl;
    for ( size_t idx : flt2refPairs_.active )
    {
        const auto& vp = flt2refPairs_.vec[idx];
        p2pl.add( vp.srcPoint - centroidRef, vp.tgtPoint - centroidRef, vp.tgtNorm, vp.weight);
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
        assert( prop_.p2plScaleLimit >= 1 );
        if ( angle > prop_.p2plAngleLimit || am.scale > prop_.p2plScaleLimit || prop_.p2plScaleLimit * am.scale < 1 )
        {
            // limit rotation angle and scale
            Matrix3d mLimited = 
                std::clamp( am.scale, 1 / (double)prop_.p2plScaleLimit, (double)prop_.p2plScaleLimit ) *
                Matrix3d( Quaternion<double>(am.rotAngles, std::min( angle, (double)prop_.p2plAngleLimit ) ) );

            // recompute translation part
            PointToPlaneAligningTransform p2plTrans;
            for ( size_t idx : flt2refPairs_.active )
            {
                const auto& vp = flt2refPairs_.vec[idx];
                // below is incorrect, but some tests break when correcting it:
                // p2plTrans.add( mLimited * Vector3d( vp.srcPoint - centroidRef ), Vector3d( vp.tgtPoint - centroidRef ),
                //  Vector3d(vp.tgtNorm), vp.weight );
                p2plTrans.add( mLimited * Vector3d( vp.srcPoint - centroidRef ), mLimited * Vector3d( vp.tgtPoint - centroidRef ),
                    mLimited * Vector3d(vp.tgtNorm), vp.weight);
            }
            auto transOnly = p2plTrans.findBestTranslation();
            res = AffineXf3f(Matrix3f(mLimited), Vector3f(transOnly));
        }
        else
        {
            res = AffineXf3f( am.rigidScaleXf() );
        }
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
    resultType_ = ExitType::MaxIterations;
    for ( iter_ = 1; iter_ <= prop_.iterLimit; ++iter_ )
    {
        updatePointPairs();
        const bool pt2pt = ( prop_.method == ICPMethod::Combined && iter_ < 3 )
            || prop_.method == ICPMethod::PointToPoint;
        if ( !( pt2pt ? p2ptIter_() : p2plIter_() ) )
        {
            resultType_ = ExitType::NotFoundSolution;
            break;
        }

        const float curDist = pt2pt ? getMeanSqDistToPoint() : getMeanSqDistToPlane();
        if ( prop_.exitVal > curDist )
        {
            resultType_ = ExitType::StopMsdReached;
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
                resultType_ = ExitType::MaxBadIterations;
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

float getMeanSqDistToPoint( const PointPairs & pairs )
{
    int num = 0;
    double sum = 0;
    for ( size_t idx : pairs.active )
    {
        const auto& vp = pairs.vec[idx];
        sum += vp.distSq;
        ++num;
    }
    if ( num <= 0 )
        return FLT_MAX;
    return (float)std::sqrt( sum / num );
}

float getMeanSqDistToPlane( const PointPairs & pairs )
{
    int num = 0;
    double sum = 0;
    for ( size_t idx : pairs.active )
    {
        const auto& vp = pairs.vec[idx];
        auto v = dot( vp.tgtNorm, vp.tgtPoint - vp.srcPoint );
        sum += sqr( v );
        ++num;
    }
    if ( num <= 0 )
        return FLT_MAX;
    return (float)std::sqrt( sum / num );
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
    std::string result = "Performed " + std::to_string( iter_ ) + " iterations.\n";
    switch ( resultType_ )
    {
    case MR::ICP::ExitType::NotFoundSolution:
        result += "No solution found.";
        break;
    case MR::ICP::ExitType::MaxIterations:
        result += "Limit of iterations reached.";
        break;
    case MR::ICP::ExitType::MaxBadIterations:
        result += "No improvement iterations limit reached.";
        break;
    case MR::ICP::ExitType::StopMsdReached:
        result += "Required mean square deviation reached.";
        break;
    case MR::ICP::ExitType::NotStarted:
    default:
        result = "Not started yet.";
        break;
    }
    return result;
}

} //namespace MR
