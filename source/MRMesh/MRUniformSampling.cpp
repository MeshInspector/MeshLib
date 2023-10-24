#include "MRUniformSampling.h"
#include "MRPointCloud.h"
#include "MRBitSetParallelFor.h"
#include "MRVector.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"
#include "MRBox.h"
#include "MRPointCloudMakeNormals.h"
#include "MRPointCloudRadius.h"
#include <cfloat>

namespace MR
{

std::optional<VertBitSet> pointUniformSampling( const PointCloud& pointCloud, float distance, const ProgressCallback & cb )
{
    MR_TIMER

    const auto sz = pointCloud.points.size();
    const int reportStep = std::min( int( sz / 64 ), 1024 );
    VertId reportNext = 0_v;

    VertBitSet res = pointCloud.validPoints;
    for ( auto v : res )
    {
        if ( cb && v >= reportNext )
        {
            if ( !cb( float( v ) / sz ) )
                return {};
            reportNext = v + reportStep;
        }

        findPointsInBall( pointCloud, pointCloud.points[v], distance, [&]( VertId cv, const Vector3f& )
        {
            if ( cv > v )
                res.reset( cv );
        } );
    }
    return res;
}

std::optional<VertBitSet> pointRegularUniformSampling( const PointCloud& pointCloud, float distance, 
    const ProgressCallback& cb /*= {} */ )
{
    MR_TIMER

    std::vector<VertId> searchQueue = pointCloud.getLexicographicalOrder();

    if ( cb && !cb( 0.3f ) )
        return {};

    int progressCounter = 0;
    auto sp = subprogress( cb, 0.3f, 1.0f );

    VertBitSet visited( pointCloud.validPoints.size() );
    VertBitSet sampled( pointCloud.validPoints.size() );
    for ( auto v : searchQueue )
    {
        if ( sp && !( ( ++progressCounter ) & 0x3ff ) &&
            !sp( float( progressCounter ) / float( searchQueue.size() ) ) )
            return {};
        if ( visited.test( v ) )
            continue;
        sampled.set( v );
        findPointsInBall( pointCloud, pointCloud.points[v], distance, [&] ( VertId cv, const Vector3f& )
        {
            visited.set( cv );
        } );
    }
    return sampled;
}

std::optional<VertBitSet> pointNormalBasedSampling( const PointCloud& pointCloud, float distance, const ProgressCallback& cb /*= {} */ )
{
    MR_TIMER

    std::vector<VertId> searchQueue = pointCloud.getLexicographicalOrder();

    if ( cb && !cb( 0.3f ) )
        return {};

    std::optional<VertNormals> optNormals;
    if ( !pointCloud.hasNormals() )
    {
        optNormals = makeUnorientedNormals( pointCloud, findAvgPointsRadius( pointCloud, 48 ), subprogress( cb, 0.3f, 0.6f ) );
        if ( !optNormals )
            return {};
    }

    const auto& normals = optNormals ? *optNormals : pointCloud.normals;
    int progressCounter = 0;
    auto sp = subprogress( cb, optNormals ? 0.6 : 0.3f, 1.0f );
    VertBitSet visited( pointCloud.validPoints.size() );
    VertBitSet sampled( pointCloud.validPoints.size() );
    const auto critDistSq = sqr( distance );
    for ( auto v : searchQueue )
    {
        if ( sp && !( ( ++progressCounter ) & 0x3ff ) &&
            !sp( float( progressCounter ) / float( searchQueue.size() ) ) )
            return {};
        if ( visited.test( v ) )
            continue;
        sampled.set( v );
        const auto c = pointCloud.points[v];
        const auto n = normals[v];
        findPointsInBall( pointCloud, c, distance, [&] ( VertId u, const Vector3f& )
        {
            const auto dd = 2 * sqr( dot( n, normals[u] ) ) - 1;
            if ( dd <= 0 )
                return;
            if ( ( c - pointCloud.points[u] ).lengthSq() > dd * critDistSq )
                return;
            visited.set( u );
        } );
    }
    return sampled;
}

std::optional<PointCloud> makeUniformSampledCloud( const PointCloud& pointCloud, float distance, 
    const VertNormals * extNormals, const ProgressCallback & cb )
{
    MR_TIMER

    std::optional<PointCloud> res;
    auto optVerts = pointUniformSampling( pointCloud, distance, subprogress( cb, 0.0f, 0.9f ) );
    if ( !optVerts )
        return res;

    res.emplace();
    res->addPartByMask( pointCloud, *optVerts, {}, extNormals );

    if ( !reportProgress( cb, 1.0f ) )
        res.reset();
    return res;
}

} //namespace MR
