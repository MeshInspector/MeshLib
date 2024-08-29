#include "MRPointsInBall.h"
#include "MRPointCloud.h"
#include "MRMesh.h"
#include "MRAABBTreePoints.h"

namespace MR
{

void findPointsInBall( const PointCloud& pointCloud, const Vector3f& center, float radius,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    findPointsInBall( pointCloud.getAABBTree(), center, radius, foundCallback, xf );
}

void findPointsInBall( const Mesh& mesh, const Vector3f& center, float radius,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    findPointsInBall( mesh.getAABBTreePoints(), center, radius, foundCallback, xf );
}

void findPointsInBall( const AABBTreePoints& tree, const Vector3f& center, float radius, 
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    if ( !foundCallback )
    {
        assert( false );
        return;
    }

    if ( tree.nodes().empty() )
        return;

    const auto& orderedPoints = tree.orderedPoints();
    const float radiusSq = sqr( radius );

    constexpr int MaxStackSize = 32; // to avoid allocations
    NodeId subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( center ) : box.getDistanceSq( center );
        if ( distSq <= radiusSq )
            subtasks[stackSize++] = n;
    };

    addSubTask( tree.rootNodeId() );

    while ( stackSize > 0 )
    {
        const auto n = subtasks[--stackSize];
        const auto& node = tree[n];

        if ( node.leaf() )
        {
            auto [first, last] = node.getLeafPointRange();
            for ( int i = first; i < last; ++i )
            {
                auto coord = xf ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;
                if ( ( coord - center ).lengthSq() <= radiusSq )
                    foundCallback( orderedPoints[i].id, coord );
            }
            continue;
        }

        addSubTask( node.rightOrLast ); // look at right node later
        addSubTask( node.leftOrFirst ); // look at left node first
    }
}

} //namespace MR
