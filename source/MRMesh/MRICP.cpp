#include "MRICP.h"
#include "MRMesh.h"
#include "MRAligningTransform.h"
#include "MRMeshNormals.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MRQuaternion.h"
#include "MRBestFit.h"
#include "MRParallelFor.h"
#include <numeric>

const int MAX_RESAMPLING_VOXEL_NUMBER = 500000;

namespace MR
{

namespace
{

void setupPairs( PointPairs & pairs, const VertBitSet& srcSamples )
{
    pairs.clear();
    pairs.reserve( srcSamples.count() );
    for ( auto id : srcSamples )
        pairs.emplace_back().srcVertId = id;
}

size_t deactivateFarPairs( PointPairs & pairs, float maxDistSq )
{
    return parallel_reduce( tbb::blocked_range( size_t(0), pairs.size() ), size_t(0),
    [&] ( const auto & range, size_t curr )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto & p = pairs[i];
            if ( p.active && p.distSq > maxDistSq )
            {
                p.active = false;
                ++curr;
            }
        }
        return curr;
    },
    [] ( auto a, auto b ) { return a + b; } );
}

} // anonymous namespace

ICP::ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
    const VertBitSet& fltSamples)
    : flt_( floating )
    , ref_( reference )
{
    setXfs( fltXf, refXf );
    setupPairs( flt2refPairs_, fltSamples );
    updatePointPairs();
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
    updatePointPairs();
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

    const auto srcNormals = src.normals();
    const auto srcWeights = src.weights();
    const auto tgtProjector = tgt.projector();

    // calculate pairs
    ParallelFor( pairs, [&] ( size_t idx )
    {
        const auto& p = srcPoints[pairs[idx].srcVertId];
        const auto prj = tgtProjector( src2tgtXf( p ) );

        // projection should be found and if point projects on the border it will be ignored
        if ( !prj.isBd )
        {
            PointPair vp = pairs[idx];
            vp.distSq = prj.distSq;
            vp.weight = srcWeights ? srcWeights( vp.srcVertId ) : 1.0f;
            vp.tgtPoint = tgtXf( prj.point );
            vp.tgtNorm = prj.normal ? ( tgtXf.A * prj.normal.value() ).normalized() : Vector3f();
            vp.srcNorm = srcNormals ? ( srcXf.A * srcNormals( vp.srcVertId ) ).normalized() : Vector3f();
            vp.normalsAngleCos = ( prj.normal && srcNormals ) ? dot( vp.tgtNorm, vp.srcNorm ) : 1.0f;
            vp.active = vp.normalsAngleCos >= prop_.cosTreshold && vp.distSq <= prop_.distThresholdSq;
            pairs[idx] = vp;
        }
        else
        {
            pairs[idx].active = false;
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

        if ( deactivateFarPairs( flt2refPairs_, maxDistSq ) <= 0 )
            break; // nothing was deactivated
    }
}

bool ICP::p2ptIter_()
{
    MR_TIMER;
    const VertCoords& points = flt_.points();
    PointToPointAligningTransform p2pt;
    for (const auto& vp : flt2refPairs_)
    {
        if ( !vp.active )
            continue;
        const auto v1 = fltXf_(points[vp.srcVertId]);
        const auto& v2 = vp.tgtPoint;
        p2pt.add(Vector3d(v1), Vector3d(v2), vp.weight);
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
    MR_TIMER;
    const VertCoords& points = flt_.points();
    Vector3f centroidRef;
    int activeCount = 0;
    for (auto& vp : flt2refPairs_)
    {
        if ( !vp.active )
            continue;
        centroidRef += vp.tgtPoint;
        centroidRef += fltXf_(points[vp.srcVertId]);
        ++activeCount;
    }
    if ( activeCount <= 0 )
        return false;
    centroidRef /= float(activeCount * 2);
    AffineXf3f centroidRefXf = AffineXf3f(Matrix3f(), centroidRef);

    PointToPlaneAligningTransform p2pl;
    for (const auto& vp : flt2refPairs_)
    {
        if ( !vp.active )
            continue;
        const auto v1 = fltXf_(points[vp.srcVertId]);
        const auto& v2 = vp.tgtPoint;
        p2pl.add(Vector3d(v1 - centroidRef), Vector3d(v2 - centroidRef), Vector3d(vp.tgtNorm), vp.weight);
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
            for (const auto& vp : flt2refPairs_)
            {
                if ( !vp.active )
                    continue;
                const auto v1 = fltXf_(points[vp.srcVertId]);
                const auto& v2 = vp.tgtPoint;
                p2plTrans.add(mLimited * Vector3d(v1 - centroidRef), mLimited * Vector3d(v2 - centroidRef),
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
    float curDist = 0;
    float minDist = std::numeric_limits<float>::max();
    int badIterCount = 0;
    resultType_ = ExitType::NotStarted;
    for (iter_ = 0; iter_ < prop_.iterLimit; iter_++)
    {
        if (prop_.method == ICPMethod::Combined)
        {
            if (iter_ < 2)
            {
                if ( !p2ptIter_() )
                {
                    resultType_ = ExitType::NotFoundSolution;
                    break;
                }
                updatePointPairs();
                curDist = getMeanSqDistToPoint();
            }
            else
            {
                if ( !p2plIter_() )
                {
                    resultType_ = ExitType::NotFoundSolution;
                    break;
                }
                updatePointPairs();
                curDist = getMeanSqDistToPlane();
                if ( prop_.exitVal > curDist )
                {
                    resultType_ = ExitType::StopMsdReached;
                    break;
                }
            }
        }

        if (prop_.method == ICPMethod::PointToPoint)
        {
            if ( !p2ptIter_() )
            {
                resultType_ = ExitType::NotFoundSolution;
                break;
            }
            updatePointPairs();
            curDist = getMeanSqDistToPoint();
            if ( prop_.exitVal > curDist )
            {
                resultType_ = ExitType::StopMsdReached;
                break;
            }
        }

        if (prop_.method == ICPMethod::PointToPlane)
        {
            if ( !p2plIter_() )
            {
                resultType_ = ExitType::NotFoundSolution;
                break;
            }
            updatePointPairs();
            curDist = getMeanSqDistToPlane();
            if ( prop_.exitVal > curDist )
            {
                resultType_ = ExitType::StopMsdReached;
                break;
            }
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
    if ( iter_ == prop_.iterLimit )
        resultType_ = ExitType::MaxIterations;
    else
        iter_++;
    return fltXf_;
}

float getMeanSqDistToPoint( const PointPairs & pairs )
{
    int num = 0;
    double sum = 0;
    for ( const auto& vp : pairs )
    {
        if ( !vp.active )
            continue;
        sum += vp.distSq;
        ++num;
    }
    if ( num <= 0 )
        return FLT_MAX;
    return (float)std::sqrt( sum / num );
}

float getMeanSqDistToPlane( const PointPairs & pairs, const MeshOrPoints & floating, const AffineXf3f & floatXf )
{
    const VertCoords& points = floating.points();
    int num = 0;
    double sum = 0;
    for ( const auto& vp : pairs )
    {
        if ( !vp.active )
            continue;
        auto v = dot( vp.tgtNorm, vp.tgtPoint - floatXf(points[vp.srcVertId]) );
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
