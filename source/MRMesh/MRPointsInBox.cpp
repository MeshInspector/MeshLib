#include "MRPointsInBox.h"
#include "MRPointCloud.h"
#include "MRMesh.h"
#include "MRAABBTreePoints.h"

namespace MR
{

void findPointsInBox( const PointCloud& pointCloud, const Box3f& box,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    findPointsInBox( pointCloud.getAABBTree(), box, foundCallback, xf );
}

void findPointsInBox( const Mesh& mesh, const Box3f& box,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    findPointsInBox( mesh.getAABBTreePoints(), box, foundCallback, xf );
}

void findPointsInBox( const AABBTreePoints& tree, const Box3f& box,
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

    constexpr int MaxStackSize = 32; // to avoid allocations
    NodeId subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( NodeId n )
    {
        const auto & nodeBox = tree.nodes()[n].box;
        if ( xf ? transformed( nodeBox, *xf ).intersects( box ) : nodeBox.intersects( box ) )
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
                if ( box.contains( coord ) )
                    foundCallback( orderedPoints[i].id, coord );
            }
            continue;
        }

        addSubTask( node.r ); // look at right node later
        addSubTask( node.l ); // look at left node first
    }
}

} //namespace MR
