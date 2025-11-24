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

size_t deactivateFarPairs( IPointPairs& pairs, float maxDistSq )
{
    size_t cnt0 = pairs.active.count();
    BitSetParallelFor( pairs.active, [&]( size_t i )
    {
        if ( pairs[i].distSq > maxDistSq )
            pairs.active.reset( i );
    } );
    return cnt0 - pairs.active.count();
}


ICP::ICP( const MeshOrPointsXf& flt, const MeshOrPointsXf& ref, const VertBitSet& fltSamples, const VertBitSet& refSamples )
    : flt_( flt )
    , ref_( ref )
{
    MR::setupPairs( flt2refPairs_, fltSamples );
    MR::setupPairs( ref2fltPairs_, refSamples );
}

ICP::ICP( const MeshOrPointsXf& flt, const MeshOrPointsXf& ref, float samplingVoxelSize )
    : flt_( flt )
    , ref_( ref )
{
    samplePoints( samplingVoxelSize );
}

void ICP::setXfs( const AffineXf3f& fltXf, const AffineXf3f& refXf )
{
    ref_.xf = refXf;
    setFloatXf( fltXf );
}

void ICP::setFloatXf( const AffineXf3f& fltXf )
{
    flt_.xf = fltXf;
}

AffineXf3f ICP::autoSelectFloatXf()
{
    MR_TIMER;

    auto bestFltXf = flt_.xf;
    float bestDist = getMeanSqDistToPoint();

    PointAccumulator refAcc;
    ref_.obj.accumulate( refAcc );
    const auto refBasisXfs = refAcc.get4BasicXfs();

    PointAccumulator floatAcc;
    flt_.obj.accumulate( floatAcc );
    const auto floatBasisXf = floatAcc.getBasicXf();

    // TODO: perform computations in parallel by calling free functions to measure the distance
    for ( const auto & refBasisXf : refBasisXfs )
    {
        AffineXf3f fltXf( AffineXf3d( ref_.xf ) * refBasisXf * floatBasisXf.inverse() );
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

void ICP::setFltSamples( const VertBitSet& fltSamples )
{
    setupPairs( flt2refPairs_, fltSamples );
}

void ICP::sampleFltPoints( float samplingVoxelSize )
{
    setFltSamples( *flt_.obj.pointsGridSampling( samplingVoxelSize ) );
}

void ICP::setRefSamples( const VertBitSet& refSamples )
{
    setupPairs( ref2fltPairs_, refSamples );
}

void ICP::sampleRefPoints( float samplingVoxelSize )
{
    setRefSamples( *ref_.obj.pointsGridSampling( samplingVoxelSize ) );
}

void ICP::updatePointPairs()
{
    MR_TIMER;
    MR::updatePointPairs( flt2refPairs_, flt_, ref_, prop_.cosThreshold, prop_.distThresholdSq, prop_.mutualClosest );
    MR::updatePointPairs( ref2fltPairs_, ref_, flt_, prop_.cosThreshold, prop_.distThresholdSq, prop_.mutualClosest );
    deactivatefarDistPairs_();
}

std::string getICPStatusInfo( int iterations, ICPExitType exitType )
{
    std::string result = "Performed " + std::to_string( iterations - 1 ) + " iterations.\n";
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
    const MeshOrPointsXf& src, const MeshOrPointsXf& tgt,
    float cosThreshold, float distThresholdSq, bool mutualClosest )
{
    MR_TIMER;
    const AffineXf3f src2tgtXf( AffineXf3d( tgt.xf ).inverse() * AffineXf3d( src.xf ) );
    const AffineXf3f tgt2srcXf( AffineXf3d( src.xf ).inverse() * AffineXf3d( tgt.xf ) );

    const VertCoords& srcPoints = src.obj.points();
    const VertCoords& tgtPoints = tgt.obj.points();

    const auto srcNormals = src.obj.normals();
    const auto tgtNormals = tgt.obj.normals();

    const auto srcWeights = src.obj.weights();
    const auto srcLimProjector = src.obj.limitedProjector();
    const auto tgtLimProjector = tgt.obj.limitedProjector();

    pairs.active.clear();
    pairs.active.resize( pairs.vec.size(), true );

    // calculate pairs
    BitSetParallelForAll( pairs.active, [&] ( size_t idx )
    {
        auto & res = pairs.vec[idx];
        const auto p0 = srcPoints[res.srcVertId];
        const auto pt = src2tgtXf( p0 );

        MeshOrPoints::ProjectionResult prj;
        // do not search for target point further than distance threshold
        prj.distSq = distThresholdSq;
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
        if ( !prj.closestVert )
        {
            // no target point found within distance threshold
            pairs.active.reset( idx );
            return;
        }
        const auto p1 = prj.point;

        // save the result
        PointPair vp = res;
        vp.distSq = prj.distSq;
        vp.weight = srcWeights ? srcWeights( vp.srcVertId ) : 1.0f;
        vp.tgtCloseVert = prj.closestVert;
        vp.srcPoint = src.xf( p0 );
        vp.tgtPoint = tgt.xf( p1 );
        vp.tgtNorm = prj.normal ? ( tgt.xf.A * prj.normal.value() ).normalized() : Vector3f();
        vp.srcNorm = srcNormals ? ( src.xf.A * srcNormals( vp.srcVertId ) ).normalized() : Vector3f();
        vp.normalsAngleCos = ( prj.normal && srcNormals ) ? dot( vp.tgtNorm, vp.srcNorm ) : 1.0f;
        vp.tgtOnBd = prj.isBd;
        res = vp;
        if ( prj.isBd || vp.normalsAngleCos < cosThreshold || vp.distSq > distThresholdSq )
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
    MR_TIMER;

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
    MR_TIMER;
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
    setFloatXf( res * flt_.xf );
    return true;
}

AffineXf3d getAligningXf( const PointToPlaneAligningTransform & p2pl,
    ICPMode mode, float angleLimit, float scaleLimit, const Vector3f & fixedRotationAxis )
{
    AffineXf3d res;
    if( mode == ICPMode::TranslationOnly )
    {
        res = AffineXf3d( Matrix3d(), p2pl.findBestTranslation() );
    }
    else
    {
        RigidScaleXf3d am;
        switch ( mode )
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
            am = p2pl.calculateOrthogonalAxisAmendment( Vector3d{ fixedRotationAxis } );
            break;
        case ICPMode::FixedAxis:
            am = p2pl.calculateFixedAxisAmendment( Vector3d{ fixedRotationAxis } );
            break;
        }

        const auto angle = am.a.length();
        assert( angleLimit > 0 );
        assert( scaleLimit >= 1 );
        if ( angle > angleLimit || am.s > scaleLimit || scaleLimit * am.s < 1 )
        {
            // limit rotation angle and scale
            am.s = std::clamp( am.s, 1 / (double)scaleLimit, (double)scaleLimit );
            if ( angle > angleLimit )
                am.a *= angleLimit / angle;

            // recompute translation part
            am.b = p2pl.findBestTranslation( am.a, am.s );
        }
        res = am.rigidScaleXf();
    }
    return res;
}

bool ICP::p2plIter_()
{
    MR_TIMER;
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
    p2pl.prepare();

    const AffineXf3d res = getAligningXf( p2pl, prop_.icpMode, prop_.p2plAngleLimit, prop_.p2plScaleLimit, prop_.fixedRotationAxis );
    if (std::isnan(res.b.x)) //nan check
        return false;

    const AffineXf3d subCentroid{ Matrix3d(), Vector3d( -centroidRef ) };
    const AffineXf3d addCentroid{ Matrix3d(), Vector3d(  centroidRef ) };
    setFloatXf( AffineXf3f( addCentroid * res * subCentroid * AffineXf3d( flt_.xf ) ) );
    return true;
}

void ICP::calcGen_( float (ICP::*dist)() const, bool (ICP::*iter)() )
{
    updatePointPairs();
    float minDist = (this->*dist)();

    int badIterCount = 0;
    resultType_ = ICPExitType::MaxIterations;
    AffineXf3f resXf = flt_.xf;
    for ( iter_ = 1; iter_ <= prop_.iterLimit; ++iter_ )
    {
        if ( !(this->*iter)() )
        {
            resultType_ = ICPExitType::NotFoundSolution;
            break;
        }
        // without this call, getMeanSqDistToPoint()/getMeanSqDistToPlane() will ignore xf changed in p2ptIter_()/p2plIter_()
        updatePointPairs();

        const float curDist = (this->*dist)();

        // exit if several(3) iterations didn't decrease minimization parameter
        if ( curDist < minDist )
        {
            resXf = flt_.xf;
            minDist = curDist;
            badIterCount = 0;

            if ( prop_.exitVal > curDist )
            {
                resultType_ = ICPExitType::StopMsdReached;
                break;
            }
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
    flt_.xf = resXf;
}

void ICP::calcP2Pt_()
{
    MR_TIMER;
    calcGen_( &ICP::getMeanSqDistToPoint, &ICP::p2ptIter_ );
}

void ICP::calcP2Pl_()
{
    MR_TIMER;
    calcGen_( &ICP::getMeanSqDistToPlane, &ICP::p2plIter_ );
}

void ICP::calcCombined_()
{
    MR_TIMER;
    assert( prop_.iterLimit > 2 );
    const auto safeLimit = prop_.iterLimit;

    prop_.iterLimit = 2;
    calcP2Pt_();

    prop_.iterLimit = safeLimit - 2;
    calcP2Pl_();

    prop_.iterLimit = safeLimit;
}

AffineXf3f ICP::calculateTransformation()
{
    switch ( prop_.method )
    {
    case ICPMethod::Combined:
        calcCombined_();
        break;
    case ICPMethod::PointToPoint:
        calcP2Pt_();
        break;
    case ICPMethod::PointToPlane:
        calcP2Pl_();
        break;
    default:
        assert( false );
        break;
    }
    return flt_.xf;
}

size_t getNumActivePairs( const IPointPairs& pairs )
{
    return pairs.active.count();
}

NumSum getSumSqDistToPoint( const IPointPairs& pairs, std::optional<double> inaccuracy )
{
    NumSum res;
    for ( size_t idx : pairs.active )
    {
        const auto& vp = pairs[idx];
        if ( inaccuracy )
            res.sum += sqr( std::sqrt( vp.distSq ) - *inaccuracy );
        else
            res.sum += vp.distSq;
        ++res.num;
    }
    return res;
}

NumSum getSumSqDistToPlane( const IPointPairs& pairs, std::optional<double> inaccuracy )
{
    NumSum res;
    for ( size_t idx : pairs.active )
    {
        const auto& vp = pairs[idx];
        auto v = dot( vp.tgtNorm, vp.tgtPoint - vp.srcPoint );
        if ( inaccuracy )
            res.sum += sqr( std::abs( v ) - *inaccuracy );
        else
            res.sum += sqr( v );
        ++res.num;
    }
    return res;
}

void ICP::setCosineLimit(const float cos)
{
    prop_.cosThreshold = cos;
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

std::string ICP::getStatusInfo() const
{
    return getICPStatusInfo( iter_, resultType_ );
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
