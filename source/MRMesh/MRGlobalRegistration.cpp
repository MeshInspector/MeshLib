#include "MRGlobalRegistration.h"
#include "MRPointCloudFPFH.h"
#include "MRRigidXfRansac.h"
#include "MRICP.h"
#include "MRPointCloud.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRBox.h"
#include "MRUnionFind.h"
#include "MRTimer.h"
#include <algorithm>
#include <atomic>
#include <optional>
#include <thread>

namespace MR
{

namespace
{

// grid samples of the object's surface with normals, in world space
Expected<PointCloud> sampleSurface( const MeshOrPointsXf & obj, float voxelSize )
{
    std::shared_ptr<PointCloud> dense;
    const auto ids = obj.obj.surfaceGridSampling( voxelSize, dense );
    const auto src = dense ? MeshOrPoints( *dense ) : obj.obj;
    const auto normals = src.normals();
    if ( !normals )
        return unexpected( "Point cloud without normals" );
    const auto & points = src.points();
    const auto normalA = obj.xf.A.inverse().transposed();
    PointCloud res;
    for ( auto v : ids )
        res.addPoint( obj.xf( points[v] ), ( normalA * normals( v ) ).normalized() );
    return res;
}

// the fraction of given samples that (motion) moves within (tol) from (obj) with similar normals
float overlapFraction( const PointCloud & samples, const AffineXf3f & motion, const MeshOrPointsXf & obj, float tol )
{
    const auto proj = obj.limitedProjector();
    VertBitSet on( samples.points.size() );
    BitSetParallelFor( samples.validPoints, [&]( VertId v )
    {
        MeshOrPoints::ProjectionResult res;
        res.distSq = sqr( tol );
        if ( !proj( motion( samples.points[v] ), res ) || !res.normal )
            return;
        if ( dot( ( motion.A * samples.normals[v] ).normalized(), res.normal->normalized() ) > 0.9f )
            on.set( v );
    } );
    return float( on.count() ) / std::max( size_t( 1 ), samples.validPoints.count() );
}

struct PairXf
{
    ObjId a, b;
    float overlap = 0;
    AffineXf3f bInA; // the transformation of b to the local space of a
};

} // anonymous namespace

Expected<GlobalRegistrationResult> globalRegistration( const Vector<MeshOrPointsXf, ObjId> & objects,
    const GlobalRegistrationParams & params, const ProgressCallback & progress )
{
    MR_TIMER;
    const auto n = objects.size();
    float diag = 0;
    for ( const auto & o : objects )
        if ( const auto box = o.obj.computeBoundingBox( &o.xf ); box.valid() )
            diag = std::max( diag, box.diagonal() );
    if ( !( diag > 0 ) )
        return unexpected( "No object has points" );
    const float spacing = params.samplingVoxelSize > 0 ? params.samplingVoxelSize : 0.005f * diag;

    Vector<PointCloud, ObjId> samples;
    Vector<FPFHDescriptors, ObjId> descriptors;
    samples.reserve( n );
    descriptors.reserve( n );
    for ( const auto & o : objects )
    {
        auto s = sampleSurface( o, spacing );
        if ( !s )
            return unexpected( std::move( s.error() ) );
        // the neighborhood of 5 sampling steps is a usual choice for FPFH
        auto d = computeFPFH( *s, 5 * spacing, subprogress( progress, 0.1f * samples.size() / n, 0.1f * ( samples.size() + 1 ) / n ) );
        if ( !d )
            return unexpected( std::move( d.error() ) );
        samples.push_back( std::move( *s ) );
        descriptors.push_back( std::move( *d ) );
    }

    ICPProperties icpParams;
    // after RANSAC the pose is within a few samples; a larger threshold lets the non-overlapping parts of point clouds drag it away
    icpParams.distThresholdSq = sqr( 5 * spacing );
    icpParams.iterLimit = 100;
    icpParams.badIterStopCount = 20;

    std::vector<PairXf> pairs;
    pairs.reserve( n * ( n - 1 ) / 2 );
    for ( ObjId a( 0 ); a < n; ++a )
        for ( ObjId b( a + 1 ); b < n; ++b )
            pairs.push_back( { a, b } );
    for ( const auto & s : samples )
        s.getAABBTree();

    // the pairs are processed in parallel, since the work inside one pair is too small to occupy all threads;
    // the steps done in all pairs are reported as progress, only from the calling thread
    const int attempts = std::max( 1, params.pairAttempts );
    const size_t stepsPerPair = 1 + 2 * attempts; // matching, then RANSAC and ICP in each attempt
    const float numSteps = float( std::max( size_t( 1 ), pairs.size() * stepsPerPair ) );
    std::atomic<size_t> stepsDone{ 0 };
    std::atomic<bool> canceled{ false };
    const auto callingThread = std::this_thread::get_id();
    const ProgressCallback insidePair = progress ? ProgressCallback( [&]( float )
    {
        if ( canceled.load( std::memory_order_relaxed ) )
            return false;
        if ( std::this_thread::get_id() == callingThread && !progress( 0.1f + 0.9f * stepsDone.load( std::memory_order_relaxed ) / numSteps ) )
            canceled = true;
        return !canceled.load( std::memory_order_relaxed );
    } ) : ProgressCallback{};
    ParallelFor( size_t( 0 ), pairs.size(), [&]( size_t pairIndex )
    {
        if ( !reportProgress( insidePair, 0.0f ) )
            return;
        auto & best = pairs[pairIndex];
        const auto a = best.a;
        const auto b = best.b;
        const auto & sa = samples[a];
        const auto & sb = samples[b];
        auto matches = findFPFHMatches( descriptors[b], sb.validPoints, descriptors[a], sa.validPoints, true, insidePair );
        if ( !matches || matches->size() < 3 )
        {
            stepsDone += stepsPerPair;
            return;
        }
        ++stepsDone;
        const bool bSmaller = sb.validPoints.count() <= sa.validPoints.count();
        std::optional<ICP> icp; // sampled once per pair
        for ( int i = 0; i < attempts; ++i )
        {
            RigidXfRansacParams rp;
            rp.inlierDistance = 1.5f * spacing;
            rp.seed = params.seed + std::uint32_t( i );
            if ( bSmaller )
            {
                rp.overlapSrcPoints = &sb.points;
                rp.overlapTgtCloud = &sa;
            }
            else
            {
                rp.overlapTgtPoints = &sa.points;
                rp.overlapSrcCloud = &sb;
            }
            const auto bMotion = findRigidXfRansac( sb.points, sa.points, *matches, rp, insidePair );
            if ( canceled )
                return;
            ++stepsDone;
            if ( !bMotion )
            {
                ++stepsDone;
                continue;
            }

            if ( icp )
                icp->setFloatXf( *bMotion * objects[b].xf );
            else
            {
                icp.emplace( MeshOrPointsXf{ objects[b].obj, *bMotion * objects[b].xf }, objects[a], 3 * spacing );
                icp->setParams( icpParams );
            }
            const auto bXf = icp->calculateTransformation();
            ++stepsDone;
            const auto refined = bXf * objects[b].xf.inverse();
            const float overlap = bSmaller
                ? overlapFraction( sb, refined, objects[a], 0.5f * spacing )
                : overlapFraction( sa, refined.inverse(), objects[b], 0.5f * spacing );
            if ( overlap > best.overlap )
            {
                best.overlap = overlap;
                best.bInA = objects[a].xf.inverse() * bXf;
            }
        }
    } );
    if ( canceled )
        return unexpectedOperationCanceled();
    std::erase_if( pairs, [&]( const PairXf & p ) { return p.overlap < params.minOverlap; } );

    // maximum spanning forest of the pairs by overlap
    std::stable_sort( pairs.begin(), pairs.end(), []( const PairXf & x, const PairXf & y ) { return x.overlap > y.overlap; } );
    UnionFind<ObjId> uf( n );
    Vector<std::vector<std::pair<ObjId, AffineXf3f>>, ObjId> neighbours( n ); // neighbour and its transformation to local space
    for ( const auto & p : pairs )
    {
        if ( !uf.unite( p.a, p.b ).second )
            continue;
        neighbours[p.a].emplace_back( p.b, p.bInA );
        neighbours[p.b].emplace_back( p.a, p.bInA.inverse() );
    }

    GlobalRegistrationResult res;
    res.xfs.resize( n );
    res.groupRoots.resize( n );
    std::vector<ObjId> stack;
    for ( ObjId r( 0 ); r < n; ++r )
    {
        if ( res.groupRoots[r] )
            continue;
        res.groupRoots[r] = r;
        res.xfs[r] = objects[r].xf;
        stack.push_back( r );
        while ( !stack.empty() )
        {
            const auto a = stack.back();
            stack.pop_back();
            for ( const auto & [b, bInA] : neighbours[a] )
            {
                if ( res.groupRoots[b] )
                    continue;
                res.groupRoots[b] = r;
                res.xfs[b] = res.xfs[a] * bInA;
                stack.push_back( b );
            }
        }
    }
    reportProgress( progress, 1.0f );
    return res;
}

} //namespace MR
