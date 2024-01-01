#include "MRICP.h"
#include "MRMesh.h"
#include "MRAligningTransform.h"
#include "MRMeshNormals.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MRQuaternion.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"
#include <numeric>
#include "MRBestFit.h"

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
    updateVertPairs();
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
        updateVertPairs();
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

    updateVertPairs();
}

void ICP::updateVertPairs()
{
    MR_TIMER;
    const auto& actualBitSet = floatVerts_;

    const VertCoords& points = floating_.points();
    if (!prop_.freezePairs)
    {
        vertPairs_.clear();
        vertPairs_.resize(actualBitSet.count());
        {
            size_t i = 0;
            for (auto id : actualBitSet)
            {
                VertPair& vp = vertPairs_[i];
                vp.vertId = id;
                i++;
            }
        }
    }

    const auto floatNormals = floating_.normals();
    const auto floatWeights = floating_.weights();
    const auto refProjector = ref_.projector();

    // calculate pairs
    tbb::parallel_for(tbb::blocked_range<size_t>(0, vertPairs_.size()),
        [&](const tbb::blocked_range<size_t>& range)
        {
            for (size_t idx = range.begin(); idx < range.end(); ++idx)
            {
                VertPair& vp = vertPairs_[idx];
                auto& id = vp.vertId;
                const auto& p = points[id];
                const auto prj = refProjector( float2refXf_(p) );

                // projection should be found and if point projects on the border it will be ignored
                if ( !prj.isBd )
                {
                    vp.vertDist2 = prj.distSq;
                    vp.weight = floatWeights ? floatWeights(id) : 1.0f;
                    vp.refPoint = refXf_( prj.point );
                    vp.normRef = prj.normal ? ( refXf_.A * prj.normal.value() ).normalized() : Vector3f();
                    vp.norm = floatNormals ? ( floatXf_.A * floatNormals( id ) ).normalized() : Vector3f();
                    vp.normalsAngleCos = ( prj.normal && floatNormals ) ? dot( vp.normRef, vp.norm ) : 1.0f;
                }
                else
                {
                    vp.vertId = VertId(); //invalid
                }
            }
        });

    removeInvalidVertPairs_();

    if (!prop_.freezePairs)
    {
        updateVertFilters_();
    }
}

void ICP::removeInvalidVertPairs_()
{
    // remove border and unprojected cases pairs
    auto newEndIter = std::remove_if(vertPairs_.begin(), vertPairs_.end(), [](const VertPair & vp) {
            return !vp.vertId.valid();
        });
    vertPairs_.erase(newEndIter, vertPairs_.end());
}

void ICP::updateVertFilters_()
{
    // finding mean value
    float meanVal = 0.f;
    for (const auto& vp : vertPairs_)
    {
        meanVal += std::sqrt(vp.vertDist2);
    }
    meanVal /= float(vertPairs_.size());

    //finding standart deviation
    float stDev = 0.f;
    for (const auto& vp : vertPairs_)
    {
        float dif = (meanVal - std::sqrt(vp.vertDist2));
        stDev += dif * dif;
    }
    stDev /= float(vertPairs_.size());
    stDev = std::sqrt(stDev);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, vertPairs_.size()),
        [&](const tbb::blocked_range<size_t>& range)
        {
            for (size_t idx = range.begin(); idx < range.end(); ++idx)
            {
                VertPair& vp = vertPairs_[idx];
                if ((vp.normalsAngleCos < prop_.cosTreshold) || //cos filter
                    (vp.vertDist2 > prop_.distTresholdSq) || //dist filter
                    (std::abs(std::sqrt(vp.vertDist2) - meanVal) > prop_.distStatisticSigmaFactor * stDev)) //sigma filter
                {
                    vp.vertId = VertId(); //invalidate
                }
            }
        });

    removeInvalidVertPairs_();
}

std::pair<float,float> ICP::getDistLimitsSq() const
{
    float minPairsDist2_ = std::numeric_limits < float>::max();
    float maxPairsDist2_ = 0.f;
    for (const auto& vp : vertPairs_)
    {
        maxPairsDist2_ = std::max(vp.vertDist2, maxPairsDist2_);
        minPairsDist2_ = std::min(vp.vertDist2, minPairsDist2_);
    }
    return std::make_pair(minPairsDist2_, maxPairsDist2_);
}

bool ICP::p2ptIter_()
{
    MR_TIMER;
    const VertCoords& points = floating_.points();
    PointToPointAligningTransform p2pt;
    for (const auto& vp : vertPairs_)
    {
        const auto& id = vp.vertId;
        const auto v1 = floatXf_(points[id]);
        const auto& v2 = vp.refPoint;
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
    if ( vertPairs_.empty() )
        return false;
    const VertCoords& points = floating_.points();
    Vector3f centroidRef;
    for (auto& vp : vertPairs_)
    {
        centroidRef += vp.refPoint;
        centroidRef += floatXf_(points[vp.vertId]);
    }
    centroidRef /= float(vertPairs_.size() * 2);
    AffineXf3f centroidRefXf = AffineXf3f(Matrix3f(), centroidRef);

    PointToPlaneAligningTransform p2pl;
    for (const auto& vp : vertPairs_)
    {
        const auto v1 = floatXf_(points[vp.vertId]);
        const auto& v2 = vp.refPoint;
        p2pl.add(Vector3d(v1 - centroidRef), Vector3d(v2 - centroidRef), Vector3d(vp.normRef), vp.weight);
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

        auto angle = am.rotAngles.length();
        if (angle > prop_.p2plAngleLimit)
        {
            Matrix3d mLimited = am.scale * Matrix3d( Quaternion<double>(am.rotAngles, prop_.p2plAngleLimit) );

            // recompute translation part
            PointToPlaneAligningTransform p2plTrans;
            for (const auto& vp : vertPairs_)
            {
                const auto v1 = floatXf_(points[vp.vertId]);
                const auto& v2 = vp.refPoint;
                p2plTrans.add(mLimited * Vector3d(v1 - centroidRef), mLimited * Vector3d(v2 - centroidRef),
                    mLimited * Vector3d(vp.normRef), vp.weight);
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
                updateVertPairs();
                curDist = getMeanSqDistToPoint();
            }
            else
            {
                if ( !p2plIter_() )
                {
                    resultType_ = ExitType::NotFoundSolution;
                    break;
                }
                updateVertPairs();
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
            updateVertPairs();
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
            updateVertPairs();
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

float getMeanSqDistToPoint( const VertPairs & pairs )
{
    if ( pairs.empty() )
        return FLT_MAX;
    double sum = 0;
    for ( const auto& vp : pairs )
    {
        sum += vp.vertDist2;
    }
    return (float)std::sqrt( sum / pairs.size() );
}

float getMeanSqDistToPlane( const VertPairs & pairs, const MeshOrPoints & floating, const AffineXf3f & floatXf )
{
    if ( pairs.empty() )
        return FLT_MAX;
    const VertCoords& points = floating.points();
    double sum = 0;
    for ( const auto& vp : pairs )
    {
        auto v = dot( vp.normRef, vp.refPoint - floatXf(points[vp.vertId]) );
        sum += sqr( v );
    }
    return (float)std::sqrt( sum / pairs.size() );
}

Vector3f ICP::getShiftVector() const
{
    const VertCoords& points = floating_.points();
    Vector3f vecAcc{ 0.f,0.f,0.f };
    for (const auto& vp : vertPairs_)
    {
        auto vec = (vp.refPoint) - floatXf_(points[vp.vertId]);
        vecAcc += vec;
    }
    return vertPairs_.size() == 0 ? vecAcc : vecAcc / float(vertPairs_.size());
}

void ICP::setCosineLimit(const float cos)
{
    prop_.cosTreshold = cos;
}

void ICP::setDistanceLimit(const float dist)
{
    prop_.distTresholdSq = dist * dist;
}

void ICP::setBadIterCount( const int iter )
{
    prop_.badIterStopCount = iter;
}

void ICP::setDistanceFilterSigmaFactor(const float factor)
{
    prop_.distStatisticSigmaFactor = factor;
}

void ICP::setPairsWeight(const std::vector<float> w)
{
    assert(vertPairs_.size() == w.size());
    for (int i = 0; i < w.size(); i++)
    {
        vertPairs_[i].weight = w[i];
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

TEST(MRMesh, RegistrationOneIterPointToPlane)
{
    const auto err = 1e-5;
    // set points
    std::vector<Vector3d> pointsFloat = {
        {   1.0,   1.0, -5.0 },
        {  14.0,   1.0,  1.0 },
        {   1.0,  14.0,  2.0 },
        { -11.0,   2.0,  3.0 },
        {   1.0, -11.0,  4.0 },
        {   1.0,   2.0,  8.0 },
        {   2.0,   1.0, -5.0 },
        {  15.0,   1.5,  1.0 },
        {   1.5,  15.0,  2.0 },
        { -11.0,   2.5,  3.1 },
    };

    // large absolute value testing
    AffineXf3d largeShiftXf = AffineXf3d(Matrix3d(), Vector3d(/*10000, - 10000, 0*/));
    for (auto& pn : pointsFloat) pn = largeShiftXf(pn);

    std::vector<Vector3d> pointsNorm = {
        {  0.0,  0.0, -1.0 },
        {  1.0,  0.1,  1.0 },
        {  0.1,  1.0,  1.2 },
        { -1.0,  0.1,  1.0 },
        {  0.1, -1.1,  1.1 },
        {  0.1,  0.1,  1.0 },
        {  0.1,  0.0, -1.0 },
        {  1.1,  0.1,  1.0 },
        {  0.1,  1.0,  1.2 },
        { -1.1,  0.1,  1.1 }
    };
    for (auto& pn : pointsNorm) pn = pn.normalized();

    // init translation
    std::vector<AffineXf3d> initP2plXfSet = {
        // zero xf
        AffineXf3d(
            Matrix3d(
                Vector3d(1, 0, 0),
                Vector3d(0, 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // Rz
        AffineXf3d(
            Matrix3d(
                Vector3d(1, sin(0.5), 0),
                Vector3d(-sin(0.5), 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // Rz + transl
        AffineXf3d(
            Matrix3d(
                Vector3d(1, sin(0.5), 0),
                Vector3d(-sin(0.5), 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(2,-2,0)),

        // complex xf
        AffineXf3d(
            Matrix3d(
                Vector3d(1, sin(0.15), -sin(0.1)),
                Vector3d(-sin(0.15), 1, sin(0.2)),
                Vector3d(sin(0.1), -sin(0.2), 1)
            ),
            Vector3d(2,-20,8)),
    };

    //std::random_device rd;
    //std::mt19937 gen(rd());
    //const double max_rnd = 0.01;
    //std::uniform_real_distribution<> dis(-max_rnd, max_rnd);
    for (const auto& initXf : initP2plXfSet)
    {
        std::vector<Vector3d> pointsRef = pointsFloat;
        std::vector<Vector3d> pointsNormT = pointsNorm;
        // add some noise
        for (int i = 0; i < pointsRef.size(); i++)
        {
            auto& pRef = pointsRef[i];
            auto& pRefNorm = pointsNormT[i];
            pRef = initXf(pRef);// +Vector3d(dis(gen), dis(gen), dis(gen));
            pRefNorm = initXf.A * pRefNorm;
        }

        PointToPlaneAligningTransform p2pl;
        for (int i = 0; i < pointsRef.size(); i++)
        {
            const auto& pRef = pointsRef[i];
            const auto& pFl = pointsFloat[i];
            const auto& pRefNorm = pointsNormT[i];

            p2pl.add(pFl, pRef, pRefNorm);
        }

        auto am = p2pl.calculateAmendment();
        AffineXf3d xfResP2pl = AffineXf3d(
            Matrix3d(
                Vector3d(1.0, -am.rotAngles[2], am.rotAngles[1]),
                Vector3d(am.rotAngles[2], 1.0, -am.rotAngles[0]),
                Vector3d(-am.rotAngles[1], am.rotAngles[0], 1.0)),
            am.shift);

        auto diffX = xfResP2pl.A.x - initXf.A.x;
        EXPECT_NEAR(diffX.length(), 0., err);

        auto diffY = xfResP2pl.A.y - initXf.A.y;
        EXPECT_NEAR(diffY.length(), 0., err);

        auto diffZ = xfResP2pl.A.z - initXf.A.z;
        EXPECT_NEAR(diffZ.length(), 0., err);

        // otrhogonality check
        // auto diffOrtho = xfResP2pl.A * xfResP2pl.A.transposed() - Matrix3d();
        // EXPECT_NEAR(diffOrtho[0].length() + diffOrtho[1].length() + diffOrtho[2].length(), 0., err);

        EXPECT_NEAR((xfResP2pl.b - initXf.b).length(), 0., err);
    }
}

}
