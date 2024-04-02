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

ICP::ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
    const VertBitSet& floatBitSet)
    : floating_( floating )
    , ref_( reference )
{
    setXfs( fltXf, refXf );
    floatVerts_ = floatBitSet;
    updatePointPairs();
}

ICP::ICP(const MeshOrPoints& floating, const MeshOrPoints& reference, const AffineXf3f& fltXf, const AffineXf3f& refXf,
    float floatSamplingVoxelSize )
    : floating_( floating )
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
    floatXf_ = fltXf;
    float2refXf_ = refXf_.inverse() * floatXf_;
}

AffineXf3f ICP::autoSelectFloatXf()
{
    MR_TIMER

    auto bestFltXf = floatXf_;
    float bestDist = getMeanSqDistToPoint();

    PointAccumulator refAcc;
    ref_.accumulate( refAcc );
    const auto refBasisXfs = refAcc.get4BasicXfs3f();

    PointAccumulator floatAcc;
    floating_.accumulate( floatAcc );
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
    auto bboxDiag = floating_.computeBoundingBox().size() / floatSamplingVoxelSize;
    auto nSamples = bboxDiag[0] * bboxDiag[1] * bboxDiag[2];
    if (nSamples > MAX_RESAMPLING_VOXEL_NUMBER)
        floatVerts_ = *floating_.pointsGridSampling( floatSamplingVoxelSize * std::cbrt(float(nSamples) / float(MAX_RESAMPLING_VOXEL_NUMBER)) );
    else
        floatVerts_ = *floating_.pointsGridSampling( floatSamplingVoxelSize );

    updatePointPairs();
}

void ICP::updatePointPairs()
{
    MR_TIMER
    const VertCoords& points = floating_.points();
    /// freeze pairs if there is at least one pair
    const bool freezePairs = prop_.freezePairs && !flt2refPairs_.empty();
    if ( !freezePairs )
    {
        flt2refPairs_.clear();
        flt2refPairs_.reserve( floatVerts_.count() );
        for ( auto id : floatVerts_ )
            flt2refPairs_.emplace_back().srcVertId = id;
    }

    const auto floatNormals = floating_.normals();
    const auto floatWeights = floating_.weights();
    const auto refProjector = ref_.projector();

    // calculate pairs
    ParallelFor( flt2refPairs_, [&] ( size_t idx )
    {
        PointPair& vp = flt2refPairs_[idx];
        auto& id = vp.srcVertId;
        const auto& p = points[id];
        const auto prj = refProjector( float2refXf_( p ) );

        // projection should be found and if point projects on the border it will be ignored
        // unless we are in freezePairs mode
        if ( freezePairs || !prj.isBd )
        {
            vp.distSq = prj.distSq;
            vp.weight = floatWeights ? floatWeights( id ) : 1.0f;
            vp.tgtPoint = refXf_( prj.point );
            vp.tgtNorm = prj.normal ? ( refXf_.A * prj.normal.value() ).normalized() : Vector3f();
            vp.srcNorm = floatNormals ? ( floatXf_.A * floatNormals( id ) ).normalized() : Vector3f();
            vp.normalsAngleCos = ( prj.normal && floatNormals ) ? dot( vp.tgtNorm, vp.srcNorm ) : 1.0f;
        }
        else
        {
            vp.srcVertId = VertId(); //invalid
        }
    } );

    if ( !freezePairs )
        filterPairs_();
}

size_t removeInvalidPointPairs( PointPairs & pairs )
{
    return std::erase_if( pairs, []( const PointPair & vp )
    {
        return !vp.srcVertId.valid();
    } );
}

void ICP::filterPairs_()
{
    MR_TIMER
    removeInvalidPointPairs( flt2refPairs_ );

    for ( int i = 0; i < 3; ++i )
    {
        const auto avgDist = getMeanSqDistToPoint();
        const auto distThresholdSq = std::min( prop_.distThresholdSq,
            sqr( prop_.farDistFactor * avgDist ) );

        ParallelFor( flt2refPairs_, [&]( size_t idx )
        {
            PointPair& vp = flt2refPairs_[idx];
            if ( !vp.srcVertId )
                return;
            if ( vp.normalsAngleCos < prop_.cosTreshold || //cos filter
                 vp.distSq > distThresholdSq ) //dist filter
            {
                vp.srcVertId = VertId(); //invalidate
            }
        } );

        if ( removeInvalidPointPairs( flt2refPairs_ ) == 0 )
            break; //nothing was filter on this iteration
    }
}

std::pair<float,float> ICP::getDistLimitsSq() const
{
    float minPairsDist2_ = std::numeric_limits < float>::max();
    float maxPairsDist2_ = 0.f;
    for (const auto& vp : flt2refPairs_)
    {
        maxPairsDist2_ = std::max(vp.distSq, maxPairsDist2_);
        minPairsDist2_ = std::min(vp.distSq, minPairsDist2_);
    }
    return std::make_pair(minPairsDist2_, maxPairsDist2_);
}

bool ICP::p2ptIter_()
{
    MR_TIMER;
    const VertCoords& points = floating_.points();
    PointToPointAligningTransform p2pt;
    for (const auto& vp : flt2refPairs_)
    {
        const auto& id = vp.srcVertId;
        const auto v1 = floatXf_(points[id]);
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
    setFloatXf( res * floatXf_ );
    return true;
}

bool ICP::p2plIter_()
{
    MR_TIMER;
    if ( flt2refPairs_.empty() )
        return false;
    const VertCoords& points = floating_.points();
    Vector3f centroidRef;
    for (auto& vp : flt2refPairs_)
    {
        centroidRef += vp.tgtPoint;
        centroidRef += floatXf_(points[vp.srcVertId]);
    }
    centroidRef /= float(flt2refPairs_.size() * 2);
    AffineXf3f centroidRefXf = AffineXf3f(Matrix3f(), centroidRef);

    PointToPlaneAligningTransform p2pl;
    for (const auto& vp : flt2refPairs_)
    {
        const auto v1 = floatXf_(points[vp.srcVertId]);
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
                const auto v1 = floatXf_(points[vp.srcVertId]);
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
    setFloatXf( centroidRefXf * res * centroidRefXf.inverse() * floatXf_ );
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
    return floatXf_;
}

float getMeanSqDistToPoint( const PointPairs & pairs )
{
    int num = 0;
    double sum = 0;
    for ( const auto& vp : pairs )
    {
        if ( !vp.srcVertId )
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
        if ( !vp.srcVertId )
            continue;
        auto v = dot( vp.tgtNorm, vp.tgtPoint - floatXf(points[vp.srcVertId]) );
        sum += sqr( v );
        ++num;
    }
    if ( num <= 0 )
        return FLT_MAX;
    return (float)std::sqrt( sum / num );
}

Vector3f ICP::getShiftVector() const
{
    const VertCoords& points = floating_.points();
    Vector3f vecAcc{ 0.f,0.f,0.f };
    for (const auto& vp : flt2refPairs_)
    {
        auto vec = (vp.tgtPoint) - floatXf_(points[vp.srcVertId]);
        vecAcc += vec;
    }
    return flt2refPairs_.size() == 0 ? vecAcc : vecAcc / float(flt2refPairs_.size());
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

void ICP::setPairsWeight(const std::vector<float> & w)
{
    assert(flt2refPairs_.size() == w.size());
    for (int i = 0; i < w.size(); i++)
    {
        flt2refPairs_[i].weight = w[i];
    }
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
