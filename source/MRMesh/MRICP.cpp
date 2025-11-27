#include "MRICP.h"
#include "MRMesh.h"
#include "MRAligningTransform.h"
#include "MRMeshNormals.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MRQuaternion.h"
#include "MRBestFit.h"
#include "MRBitSetParallelFor.h"
#include "MRStreamOperators.h"
#include "MRPch/MRSpdlog.h"
#include <numeric>
#include "MRSystem.h"
#include <fenv.h>
#include "MRRingIterator.h"
#include <bit>
#ifdef __linux__
#include <dlfcn.h>
#endif

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
    const auto refBasisXfs = refAcc.get4BasicXfs3f();

    PointAccumulator floatAcc;
    flt_.obj.accumulate( floatAcc );
    const auto floatBasisXf = floatAcc.getBasicXf3f();

    // TODO: perform computations in parallel by calling free functions to measure the distance
    for ( const auto & refBasisXf : refBasisXfs )
    {
        auto fltXf = ref_.xf * refBasisXf * floatBasisXf.inverse();
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
    spdlog::info( "updatePointPairs flt->ref" );
    MR::updatePointPairs( flt2refPairs_, flt_, ref_, prop_.cosThreshold, prop_.distThresholdSq, prop_.mutualClosest );
    spdlog::info( "updatePointPairs ref->flt" );
    MR::updatePointPairs( ref2fltPairs_, ref_, flt_, prop_.cosThreshold, prop_.distThresholdSq, prop_.mutualClosest );
    deactivatefarDistPairs_();
    spdlog::info( "flt2refPairs_.active = {}", MR::getNumActivePairs( flt2refPairs_ ) );
    spdlog::info( "ref2fltPairs_.active = {}", MR::getNumActivePairs( ref2fltPairs_ ) );
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

void logPointPairs( const PointPairs& pairs )
{
    for ( int i = 0; i < pairs.vec.size(); ++i )
    {
        const auto& pair = pairs.vec[i];
        spdlog::info( "#{}: active={}, srcVertId={}, tgtCloseVert={}, srcPoint=({} {} {}), srcNorm=({} {} {}), tgtPoint=({} {} {}), tgtNorm=({} {} {}), distSq={}, weight={}, normalsAngleCos={}, tgtOnBd={}",
            i, pairs.active.test( i ), (int)pair.srcVertId, (int)pair.tgtCloseVert,
            pair.srcPoint.x, pair.srcPoint.y, pair.srcPoint.z,
            pair.srcNorm.x, pair.srcNorm.y, pair.srcNorm.z,
            pair.tgtPoint.x, pair.tgtPoint.y, pair.tgtPoint.z,
            pair.tgtNorm.x, pair.tgtNorm.y, pair.tgtNorm.z,
            pair.distSq, pair.weight, pair.normalsAngleCos, pair.tgtOnBd );
    }
}

static void logXf( const char* var, const AffineXf3f& xf )
{
    spdlog::info( "{} x: {} {} {} {}", var, xf.A.x.x, xf.A.x.y, xf.A.x.z, xf.b.x );
    spdlog::info( "{} y: {} {} {} {}", var, xf.A.y.x, xf.A.y.y, xf.A.y.z, xf.b.y );
    spdlog::info( "{} z: {} {} {} {}", var, xf.A.z.x, xf.A.z.y, xf.A.z.z, xf.b.z );
}

static void logXfHex( const char* var, const AffineXf3f& xf )
{
    auto d = [] ( float f )
    {
        return std::bit_cast< std::uint32_t >( f );
    };
    spdlog::info( "{} x: {:08x} {:08x} {:08x} {:08x}", var, d( xf.A.x.x ), d( xf.A.x.y ), d( xf.A.x.z ), d( xf.b.x ) );
    spdlog::info( "{} y: {:08x} {:08x} {:08x} {:08x}", var, d( xf.A.y.x ), d( xf.A.y.y ), d( xf.A.y.z ), d( xf.b.y ) );
    spdlog::info( "{} z: {:08x} {:08x} {:08x} {:08x}", var, d( xf.A.z.x ), d( xf.A.z.y ), d( xf.A.z.z ), d( xf.b.z ) );
}

void updatePointPairs( PointPairs & pairs,
    const MeshOrPointsXf& src, const MeshOrPointsXf& tgt,
    float cosThreshold, float distThresholdSq, bool mutualClosest )
{
    MR_TIMER;
    const auto src2tgtXf = tgt.xf.inverse() * src.xf;
    const auto tgt2srcXf = src.xf.inverse() * tgt.xf;
    logXf( "src2tgtXf", src2tgtXf );
    logXfHex( "src2tgtXf", src2tgtXf );
    logXf( "tgt2srcXf", tgt2srcXf );
    logXfHex( "tgt2srcXf", tgt2srcXf );

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
    //BitSetParallelForAll( pairs.active, [&] ( size_t idx )
    for( size_t idx : pairs.active )
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
            continue;
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
            continue;
        }
        if ( mutualClosest )
        {
            // keep prj.distSq
            prj.closestVert = res.srcVertId;
            srcLimProjector( tgt2srcXf( p1 ), prj );
            if ( prj.closestVert != res.srcVertId )
                pairs.active.reset( idx );
        }
    }// );
}

void ICP::deactivatefarDistPairs_()
{
    MR_TIMER;

    for ( int i = 0; i < 3; ++i )
    {
        const auto avgDist = getMeanSqDistToPoint();
        const auto maxDistSq = sqr( prop_.farDistFactor * avgDist );
        spdlog::info( "avgDist = {}, maxDistSq = {}", avgDist, maxDistSq );
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

AffineXf3f getAligningXf( const PointToPlaneAligningTransform & p2pl,
    ICPMode mode, float angleLimit, float scaleLimit, const Vector3f & fixedRotationAxis )
{
    AffineXf3f res;
    if( mode == ICPMode::TranslationOnly )
    {
        res = AffineXf3f( Matrix3f(), Vector3f( p2pl.findBestTranslation() ) );
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
        res = AffineXf3f( am.rigidScaleXf() );
    }
    return res;
}

struct __attribute__((visibility("default"))) A {
    void foo() {}
    void bar();
};

auto pfoo = &A::foo;
auto pbar = &A::bar;

void A::bar() {}

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
    p2pl.prepare();

    AffineXf3f res = getAligningXf( p2pl, prop_.icpMode, prop_.p2plAngleLimit, prop_.p2plScaleLimit, prop_.fixedRotationAxis );
    if (std::isnan(res.b.x)) //nan check
        return false;
    setFloatXf( centroidRefXf * res * centroidRefXf.inverse() * flt_.xf );
    return true;
}

inline float myLength( const Vector3f& a )
{
    return std::sqrt( a.x * a.x + a.y * a.y + a.z * a.z );
}

inline float angleLog( const Vector3f& a, const Vector3f& b )
{
    const auto c = cross( a, b );
    spdlog::info( "a=({} {} {}), b=({} {} {}), c=({} {} {})",
        a.x, a.y, a.z,
        b.x, b.y, b.z,
        c.x, c.y, c.z );
    auto x = myLength( c );
    auto y = dot( a, b );
    auto r = std::atan2( x, y );
    spdlog::info( "std::atan2( {}, {} ) = {}", x, y, r );
    return r;
}

static Vector3f pseudonormalLog( const MeshTopology& topology, const VertCoords& points, VertId v, const FaceBitSet* region = nullptr )
{
    Vector3f sum;
    for ( EdgeId e : orgRing( topology, v ) )
    {
        const auto l = topology.left( e );
        if ( l && ( !region || region->test( l ) ) )
        {
            auto d0 = edgeVector( topology, points, e );
            auto d1 = edgeVector( topology, points, topology.next( e ) );
            auto angle = MR::angleLog( d0, d1 );
            auto n = cross( d0, d1 );
            sum += angle * n.normalized();
        }
    }

    return sum.normalized();
}

AffineXf3f ICP::calculateTransformation()
{
    MR::setupLoggerByDefault();
    spdlog::info( "fegetround() = {}", fegetround() );
#ifndef __EMSCRIPTEN__
    spdlog::info( "stacktrace:\n{}", getCurrentStacktrace() );
#endif
    auto* refMeshPart = ref_.obj.asMeshPart();
    auto* fltMeshPart = flt_.obj.asMeshPart();
    spdlog::info( "ref mesh = {}, flt mesh = {}", refMeshPart != nullptr, fltMeshPart != nullptr );
    if ( refMeshPart && fltMeshPart )
    {
        spdlog::info( "ref region = {}, flt region = {}", (void*)refMeshPart->region, (void*)fltMeshPart->region );
        spdlog::info( "same meshes = {}", refMeshPart->mesh == fltMeshPart->mesh );

#ifdef __linux__
        using A = std::array<void*,2>;
        void* symbol = std::bit_cast<A>(&Vector3f::lengthSq)[0];
        Dl_info info;
        if ( !dladdr( symbol, &info ) )
            spdlog::error( "Failed to get library path" );
        else
            spdlog::info( "Vector3f::lengthSq: fname={}, sname={}", info.dli_fname, info.dli_sname );
#endif

        Vector3f c( -0.02146666f, 0.0014069901f, 0.0014069926f );
        spdlog::info( "({} {} {}).lengthSq() = {}, length = {}", c.x, c.y, c.z, c.lengthSq(), c.length() );

        const auto& refPoints = refMeshPart->mesh.points;
        const auto& refTopology = refMeshPart->mesh.topology;
        //const auto& fltTopology = fltMeshPart->mesh.topology;
/*        for ( auto v = 0_v; v < refTopology.vertSize(); ++v )
        {
            auto refN = refMeshPart->mesh.pseudonormal( v );
            auto fltN = fltMeshPart->mesh.pseudonormal( v );
            auto fltNN = refN.normalized();
            spdlog::info( "{}_v: refN=({} {} {}) fltN=({} {} {}) fltNN=({} {} {})", (int)v,
                refN.x, refN.y, refN.z,
                fltN.x, fltN.y, fltN.z,
                fltNN.x, fltNN.y, fltNN.z );
        }*/
        auto n0 = pseudonormalLog( refTopology, refPoints, 0_v );
        spdlog::info( "pseudonormal 0: {} {} {}", n0.x, n0.y, n0.z );
        spdlog::info( "point 0: {} {} {}", refPoints[0_v].x, refPoints[0_v].y, refPoints[0_v].z );
        {
            std::ostringstream s;
            s << "0_v nei verts: ";
            for ( auto e : orgRing( refTopology, 0_v ) )
            {
                auto d = refTopology.dest( e );
                spdlog::info( "point {}: {} {} {}", (int)d, refPoints[d].x, refPoints[d].y, refPoints[d].z );
                s << (int)d << ' ';
            }
            spdlog::info( "{}", s.str() );
        }
        {
            std::ostringstream s;
            s << "0_v nei faces: ";
            for ( auto e : orgRing( refTopology, 0_v ) )
            {
                s << ( int )refTopology.left( e ) << ' ';
            }
            spdlog::info( "{}", s.str() );
        }
    }
    logXf( "ref_.xf", ref_.xf );
    logXfHex( "ref_.xf", ref_.xf );
    logXf( "flt_.xf", flt_.xf );
    logXfHex( "flt_.xf", flt_.xf );

    spdlog::info( "method = {}", ( int )prop_.method );
    spdlog::info( "p2plAngleLimit = {}", prop_.p2plAngleLimit );
    spdlog::info( "p2plScaleLimit = {}", prop_.p2plScaleLimit );
    spdlog::info( "cosThreshold = {}", prop_.cosThreshold );
    spdlog::info( "distThresholdSq = {}", prop_.distThresholdSq );
    spdlog::info( "farDistFactor = {}", prop_.farDistFactor );
    spdlog::info( "icpMode = {}", (int)prop_.icpMode );
    spdlog::info( "fixedRotationAxis = {} {} {}", prop_.fixedRotationAxis.x, prop_.fixedRotationAxis.y, prop_.fixedRotationAxis.z );
    spdlog::info( "iterLimit = {}", prop_.iterLimit );
    spdlog::info( "badIterStopCount = {}", prop_.badIterStopCount );
    spdlog::info( "exitVal = {}", prop_.exitVal );
    spdlog::info( "mutualClosest = {}", prop_.mutualClosest );

    bool pt2pt = prop_.method == ICPMethod::Combined || prop_.method == ICPMethod::PointToPoint;
    updatePointPairs();
    //spdlog::info( "flt2refPairs" );
    //logPointPairs( flt2refPairs_ );
    //spdlog::info( "ref2fltPairs" );
    //logPointPairs( ref2fltPairs_ );

    spdlog::info( "getMeanSqDistToPoint = {}", getMeanSqDistToPoint() );
    spdlog::info( "getMeanSqDistToPlane = {}", getMeanSqDistToPlane() );
    float minDist = pt2pt ? getMeanSqDistToPoint() : getMeanSqDistToPlane();
    spdlog::info( "minDist0 = {}", minDist );

    int badIterCount = 0;
    resultType_ = ICPExitType::MaxIterations;
    AffineXf3f resXf = flt_.xf;

    for ( iter_ = 1; iter_ <= prop_.iterLimit; ++iter_ )
    {
        spdlog::info( "Iter #{}", iter_ );
        logXf( "ref_.xf", ref_.xf );
        logXf( "flt_.xf", flt_.xf );
        pt2pt = ( prop_.method == ICPMethod::Combined && iter_ < 3 )
            || prop_.method == ICPMethod::PointToPoint;
        spdlog::info( "pt2pt = {}", pt2pt );

        if ( !( pt2pt ? p2ptIter_() : p2plIter_() ) )
        {
            resultType_ = ICPExitType::NotFoundSolution;
            break;
        }
        updatePointPairs();

        const float curDist = pt2pt ? getMeanSqDistToPoint() : getMeanSqDistToPlane();
        spdlog::info( "curDist = {}", curDist );

        // exit if several(3) iterations didn't decrease minimization parameter
        if (curDist < minDist)
        {
            resXf = flt_.xf;
            minDist = curDist;
            spdlog::info( "minDist = {}", minDist );
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
    spdlog::info( "minDist1 = {}", minDist );
    logXf( "resXf", resXf );
    flt_.xf = resXf;
    return resXf;
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
