#include "MRUniformSampling.h"
#include "MRPointCloud.h"
#include "MRBitSetParallelFor.h"
#include "MRVector.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"
#include "MRBox.h"
#include <cfloat>

namespace MR
{

std::optional<VertBitSet> pointUniformSampling( const PointCloud& pointCloud, const UniformSamplingSettings & settings )
{
    MR_TIMER

    auto cb = settings.progress;

    const VertNormals * pNormals = settings.pNormals;
    if ( !pNormals && pointCloud.hasNormals() )
        pNormals = &pointCloud.normals;

    VertBitSet visited( pointCloud.validPoints.size() );
    VertBitSet sampled( pointCloud.validPoints.size() );

    struct NearVert
    {
        VertId v;
        float distSq = 0;
    };
    std::vector<NearVert> nearVerts;

    auto processOne = [&]( VertId v )
    {
        if ( visited.test( v ) )
            return;
        sampled.set( v );
        const auto c = pointCloud.points[v];
        float localMaxDistSq = sqr( settings.distance );
        findPointsInBall( pointCloud, c, settings.distance, [&] ( VertId u, const Vector3f& pu )
        {
            const auto distSq = ( c - pu ).lengthSq();
            if ( pNormals && std::abs( dot( (*pNormals)[v], (*pNormals)[u] ) ) < settings.minNormalDot )
            {
                localMaxDistSq = std::min( localMaxDistSq, distSq );
                return;
            }
            nearVerts.push_back( { u, distSq } );
        } );
        for ( const auto & [ u, distSq ] : nearVerts )
        {
            if ( distSq >= localMaxDistSq )
                continue;
            visited.set( u );
        }
        nearVerts.clear();
    };

    size_t progressCount = 0;
    if ( settings.lexicographicalOrder )
    {
        std::vector<VertId> searchQueue = pointCloud.getLexicographicalOrder();
        if ( !reportProgress( cb, 0.3f ) )
            return {};
        cb = subprogress( cb, 0.3f, 1.0f );
        size_t totalCount = searchQueue.size();
        for ( auto v : searchQueue )
        {
            if ( cb && !( ( ++progressCount ) & 0x3ff ) && !cb( float( progressCount ) / float( totalCount ) ) )
                return {};
            processOne( v );
        }
    }
    else
    {
        size_t totalCount = pointCloud.validPoints.count();
        for ( auto v : pointCloud.validPoints )
        {
            if ( cb && !( ( ++progressCount ) & 0x3ff ) && !cb( float( progressCount ) / float( totalCount ) ) )
                return {};
            processOne( v );
        }
    }

    return sampled;
}

std::optional<PointCloud> makeUniformSampledCloud( const PointCloud& pointCloud, const UniformSamplingSettings & settings )
{
    MR_TIMER

    std::optional<PointCloud> res;
    auto s = settings;
    s.progress = subprogress( s.progress, 0.0f, 0.9f );
    auto optVerts = pointUniformSampling( pointCloud, s );
    if ( !optVerts )
        return res;

    res.emplace();
    res->addPartByMask( pointCloud, *optVerts );

    if ( !reportProgress( settings.progress, 1.0f ) )
        res.reset();
    return res;
}

} //namespace MR
