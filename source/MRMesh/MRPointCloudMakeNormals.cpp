#include "MRPointCloudMakeNormals.h"
#include "MRPointCloud.h"
#include "MRVector3.h"
#include "MRId.h"
#include "MRBox.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MRPointsInBall.h"
#include "MRTimer.h"
#include "MRPlane3.h"
#include "MRPointCloudRadius.h"
#include <cfloat>
#include <queue>

namespace MR
{

struct NormalCandidate
{
    NormalCandidate() = default;
    NormalCandidate(VertId _id, VertId _baseId,float w ):
        id{_id}, baseId{_baseId}, weight{w}{}
    VertId id;
    VertId baseId;
    float weight{FLT_MAX};
};

bool operator < ( const NormalCandidate& l, const NormalCandidate& r )
{
    return l.weight > r.weight;
}

VertCoords makeNormals( const PointCloud& pointCloud, int avgNeighborhoodSize )
{
    MR_TIMER;

    VertCoords normals( pointCloud.points.size() );
    const auto& tree = pointCloud.getAABBTree();
    AABBTreePoints::NodeId nodeId = tree.rootNodeId();
    while ( !tree[nodeId].leaf() )
        nodeId = tree[nodeId].leftOrFirst;

    auto firstLeafRadius = findAvgPointsRadius( pointCloud, avgNeighborhoodSize );

    BitSetParallelFor( pointCloud.validPoints, [&]( VertId vid )
    {
        PointAccumulator accum;
        findPointsInBall( pointCloud, pointCloud.points[vid], firstLeafRadius, 
                          [&]( VertId, const Vector3f& coord )
        {
            accum.addPoint( Vector3d( coord ) );
        } );
        normals[vid] = Vector3f( accum.getBestPlane().n ).normalized();
    } );

    Vector<float, VertId> minWeights( normals.size(), FLT_MAX );
    std::priority_queue<NormalCandidate> queue;

    auto enweight = [&]( VertId base, VertId candidate )
    {
        Vector3f cb = pointCloud.points[base] - pointCloud.points[candidate];
        return 0.01f * cb.lengthSq() + sqr( dot( cb, normals[base] ) ) + sqr( dot( cb, normals[candidate] ) );
    };

    auto enqueueNeighbors = [&]( VertId base )
    {
        findPointsInBall( pointCloud, pointCloud.points[base], firstLeafRadius,
                          [&]( VertId v, const Vector3f& )
        {
            if ( v == base )
                return;
            float weight = enweight( base, v );
            if ( weight < minWeights[v] )
            {
                queue.emplace( v, base, weight );
                minWeights[v] = weight;
            }
        } );
    };

    auto findFirst = [&]()->VertId
    {
        for ( auto v : pointCloud.validPoints )
            if ( minWeights[v] == FLT_MAX )
            {
                minWeights[v] = 0.0f;
                return v;
            }
        return VertId{};
    };

    NormalCandidate current;
    VertId first = findFirst();
    while ( first.valid() )
    {
        enqueueNeighbors( first );
        while ( !queue.empty() )
        {
            current = queue.top(); // cannot use std::move unfortunately since top() returns const reference
            queue.pop();
            if ( current.weight > minWeights[current.id] )
                continue;
            if ( dot( normals[current.baseId], normals[current.id] ) < 0.0f )
                normals[current.id] = -normals[current.id];
            enqueueNeighbors( current.id );
        }
        first = findFirst();
    }

    return normals;
}

}
