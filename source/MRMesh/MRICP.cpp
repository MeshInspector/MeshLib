#include "MRICP.h"
#include "MRMesh.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRGridSampling.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MRQuaternion.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"
#include <numeric>

const int MAX_RESAMPLING_VOXEL_NUMBER = 500000;
namespace MR
{

MeshICP::MeshICP(const MeshPart& floatingMesh, const MeshPart& referenceMesh, const AffineXf3f& fltMeshXf, const AffineXf3f& refMeshXf,
    const VertBitSet& floatingMeshBitSet)
    : meshPart_( floatingMesh )
    , refPart_( referenceMesh )
{
    setXfs( fltMeshXf, refMeshXf );
    bitSet_ = floatingMeshBitSet;
    updateVertPairs();
}

MeshICP::MeshICP(const MeshPart& floatingMesh, const MeshPart& referenceMesh, const AffineXf3f& fltMeshXf, const AffineXf3f& refMeshXf,
    float floatSamplingVoxelSize )
    : meshPart_( floatingMesh )
    , refPart_( referenceMesh )
{
    setXfs( fltMeshXf, refMeshXf );
    recomputeBitSet(floatSamplingVoxelSize);
}

void MeshICP::setXfs( const AffineXf3f& fltMeshXf, const AffineXf3f& refMeshXf )
{
    refXf_ = refMeshXf;
    refXfInv_ = refMeshXf.inverse();
    xf_ = fltMeshXf;
}

void MeshICP::recomputeBitSet(const float floatSamplingVoxelSize)
{
    auto bboxDiag = meshPart_.mesh.computeBoundingBox( meshPart_.region ).size() / floatSamplingVoxelSize;
    auto nSamples = bboxDiag[0] * bboxDiag[1] * bboxDiag[2];
    if (nSamples > MAX_RESAMPLING_VOXEL_NUMBER)
        bitSet_ = verticesGridSampling( meshPart_, floatSamplingVoxelSize * std::cbrt(float(nSamples) / float(MAX_RESAMPLING_VOXEL_NUMBER)) );
    else
        bitSet_ = verticesGridSampling( meshPart_, floatSamplingVoxelSize );

    updateVertPairs();
}

void MeshICP::updateVertPairs()
{
    MR_TIMER;
    const auto& actualBitSet = bitSet_;

    const VertCoords& points = meshPart_.mesh.points;
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

    // calculate pairs
    tbb::parallel_for(tbb::blocked_range<size_t>(0, vertPairs_.size()),
        [&](const tbb::blocked_range<size_t>& range)
        {
            for (size_t idx = range.begin(); idx < range.end(); ++idx)
            {
                VertPair& vp = vertPairs_[idx];
                auto& id = vp.vertId;
                const auto& p = points[id];
                MeshProjectionResult mp = findProjection( (refXfInv_ * xf_)(p), refPart_ );

                // projection should be found and if point projects on the border it will be ignored
                if ( !mp.mtp.isBd( refPart_.mesh.topology ) )
                {
                    vp.vertDist2 = mp.distSq;
                    vp.weight = meshPart_.mesh.dblArea(id);
                    vp.refPoint = refXf_(mp.proj.point);
                    vp.norm = xf_.A * meshPart_.mesh.normal(id);
                    vp.normRef = refXf_.A * refPart_.mesh.normal(mp.proj.face);
                    vp.normalsAngleCos = dot((vp.normRef), (vp.norm));
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

void MeshICP::removeInvalidVertPairs_()
{
    // remove border and unprojected cases pairs
    auto newEndIter = std::remove_if(vertPairs_.begin(), vertPairs_.end(), [](const VertPair & vp) {
            return !vp.vertId.valid();
        });
    vertPairs_.erase(newEndIter, vertPairs_.end());
}

void MeshICP::updateVertFilters_()
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

std::pair<float,float> MeshICP::getDistLimitsSq() const
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

bool MeshICP::p2ptIter_()
{
    MR_TIMER;
    const VertCoords& points = meshPart_.mesh.points;
    for (const auto& vp : vertPairs_)
    {
        const auto& id = vp.vertId;
        const auto v1 = xf_(points[id]);
        const auto& v2 = vp.refPoint;
        p2pt_->add(Vector3d(v1), Vector3d(v2), vp.weight);
    }

    AffineXf3f res;
    if ( prop_.icpMode == ICPMode::TranslationOnly )
        res = AffineXf3f(Matrix3f(), Vector3f(p2pt_->calculateTranslation()));
    if( prop_.icpMode == ICPMode::AnyRigidXf )
        res = AffineXf3f( p2pt_->calculateTransformationMatrix() );
    if( prop_.icpMode == ICPMode::FixedAxis )
        res = AffineXf3f( p2pt_->calculateFixedAxisRotation( Vector3d{ prop_.fixedRotationAxis } ) );
    if( prop_.icpMode == ICPMode::OrthogonalAxis )
        res = AffineXf3f( p2pt_->calculateOrthogonalAxisRotation( Vector3d{ prop_.fixedRotationAxis } ) );

    if (std::isnan(res.b.x)) //nan check
        return false;
    xf_ = res * xf_;
    p2pt_->clear();
    return true;
}

bool MeshICP::p2plIter_()
{
    MR_TIMER;
    const VertCoords& points = meshPart_.mesh.points;
    Vector3f centroidRef;
    for (auto& vp : vertPairs_)
    {
        centroidRef += vp.refPoint;
        centroidRef += xf_(points[vp.vertId]);
    }
    centroidRef /= float(vertPairs_.size() * 2);
    AffineXf3f centroidRefXf = AffineXf3f(Matrix3f(), centroidRef);

    for (const auto& vp : vertPairs_)
    {
        const auto v1 = xf_(points[vp.vertId]);
        const auto& v2 = vp.refPoint;
        p2pl_->add(Vector3d(v1 - centroidRef), Vector3d(v2 - centroidRef), Vector3d(vp.normRef), vp.weight);
    }

    AffineXf3f res;
    PointToPlaneAligningTransform::Amendment am;
    if( prop_.icpMode == ICPMode::TranslationOnly )
    {
        res = AffineXf3f( Matrix3f(), Vector3f( p2pl_->calculateTranslation() ) );
    }
    else
    {
        if( prop_.icpMode == ICPMode::FixedAxis )
        {
            am = p2pl_->calculateFixedAxisAmendment( Vector3d{ prop_.fixedRotationAxis } );
        }
        if( prop_.icpMode == ICPMode::OrthogonalAxis )
        {
            am = p2pl_->calculateOrthogonalAxisAmendment( Vector3d{ prop_.fixedRotationAxis } );
        }
        if( prop_.icpMode == ICPMode::AnyRigidXf )
        {
            am = p2pl_->calculateAmendment();
        }

        auto angle = am.rotAngles.length();
        if (angle > prop_.p2plAngleLimit)
        {
            Matrix3d mLimited = Quaternion<double>(am.rotAngles, prop_.p2plAngleLimit);

            // recompute translation part
            PointToPlaneAligningTransform p2plTrans;
            for (const auto& vp : vertPairs_)
            {
                const auto v1 = xf_(points[vp.vertId]);
                const auto& v2 = vp.refPoint;
                p2plTrans.add(mLimited * Vector3d(v1 - centroidRef), mLimited * Vector3d(v2 - centroidRef),
                    mLimited * Vector3d(vp.normRef), vp.weight);
            }
            auto transOnly = p2plTrans.calculateTranslation();
            res = AffineXf3f(Matrix3f(mLimited), Vector3f(transOnly));
        }
        else
        {
            auto xfResD = AffineXf3d(Quaternion<double>(am.rotAngles, am.rotAngles.length()), am.shift);
            res = AffineXf3f(xfResD);
        }
    }

    if (std::isnan(res.b.x)) //nan check
        return false;
    xf_ = centroidRefXf * res * centroidRefXf.inverse() * xf_;
    p2pl_->clear();
    return true;
}

AffineXf3f MeshICP::calculateTransformation()
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
    return xf_;
}

float MeshICP::getMeanSqDistToPoint() const
{
    if ( vertPairs_.empty() )
        return 0;
    double sum = 0;
    for ( const auto& vp : vertPairs_ )
    {
        sum += vp.vertDist2;
    }
    return (float)std::sqrt( sum / vertPairs_.size() );
}

float MeshICP::getMeanSqDistToPlane() const
{
    if ( vertPairs_.empty() )
        return 0;
    const VertCoords& points = meshPart_.mesh.points;
    double sum = 0;
    for (const auto& vp : vertPairs_)
    {
        auto v = dot( vp.normRef, vp.refPoint - xf_(points[vp.vertId]) );
        sum += sqr( v );
    }
    return (float)std::sqrt( sum / vertPairs_.size() );
}

Vector3f MeshICP::getShiftVector() const
{
    const VertCoords& points = meshPart_.mesh.points;
    Vector3f vecAcc{ 0.f,0.f,0.f };
    for (const auto& vp : vertPairs_)
    {
        auto vec = (vp.refPoint) - xf_(points[vp.vertId]);
        vecAcc += vec;
    }
    return vertPairs_.size() == 0 ? vecAcc : vecAcc / float(vertPairs_.size());
}

void MeshICP::setCosineLimit(const float cos)
{
    prop_.cosTreshold = cos;
}

void MeshICP::setDistanceLimit(const float dist)
{
    prop_.distTresholdSq = dist * dist;
}

void MeshICP::setBadIterCount( const int iter )
{
    prop_.badIterStopCount = iter;
}

void MeshICP::setDistanceFilterSigmaFactor(const float factor)
{
    prop_.distStatisticSigmaFactor = factor;
}

void MeshICP::setPairsWeight(const std::vector<float> w)
{
    assert(vertPairs_.size() == w.size());
    for (int i = 0; i < w.size(); i++)
    {
        vertPairs_[i].weight = w[i];
    }
}

std::string MeshICP::getLastICPInfo() const
{
    std::string result = "ICP done " + std::to_string( iter_ + 1 ) + " iterations.\n";
    switch ( resultType_ )
    {
    case MR::MeshICP::ExitType::NotFoundSolution:
        result += "No solution found.";
        break;
    case MR::MeshICP::ExitType::MaxIterations:
        result += "Limit of iterations.";
        break;
    case MR::MeshICP::ExitType::MaxBadIterations:
        result += "Limit of bad iterations.";
        break;
    case MR::MeshICP::ExitType::StopMsdReached:
        result += "Stop mean square deviation reached.";
        break;
    case MR::MeshICP::ExitType::NotStarted:
    default:
        result = "ICP hasn't started yet.";
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

TEST(MRMesh, RegistrationOneIterPointToPoint)
{
    const auto err = 1e-10;
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
    // Point to Point part
    std::vector<AffineXf3d> initP2ptXfSet = {
        // zero xf
        AffineXf3d(
            Matrix3d(
                Vector3d(1, 0, 0),
                Vector3d(0, 1, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // small Rz
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0.6, 0),
                Vector3d(-0.6, 0.8, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(0,0,0)),

        // small transl
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0.6, 0),
                Vector3d(-0.6, 0.8, 0),
                Vector3d(0, 0, 1)
            ),
            Vector3d(2,-2,0)),

        // complex xf
        AffineXf3d(
            Matrix3d(
                Vector3d(0.8, 0, -0.6),
                Vector3d(0, 1, 0),
                Vector3d(0.6, 0, 0.8)
            ),
            Vector3d(200,-200,0)),
    };
    for (const auto& initXf : initP2ptXfSet)
    {
        std::vector<Vector3d> pointsRef = pointsFloat;
        for (int i = 0; i < pointsRef.size(); i++)
        {
            auto& pRef = pointsRef[i];
            pRef = initXf(pRef);
        }

        PointToPointAligningTransform p2pt;
        for (int i = 0; i < pointsRef.size(); i++)
        {
            const auto& pRef = pointsRef[i];
            const auto& pFl = pointsFloat[i];
            p2pt.add(pFl, pRef);
        }

        auto xfResP2pt = p2pt.calculateTransformationMatrix();
        EXPECT_NEAR((xfResP2pt.A.x - initXf.A.x).length(), 0., err);
        EXPECT_NEAR((xfResP2pt.A.y - initXf.A.y).length(), 0., err);
        EXPECT_NEAR((xfResP2pt.A.z - initXf.A.z).length(), 0., err);
        EXPECT_NEAR((xfResP2pt.b - initXf.b).length(), 0., err);
    }
}

}
