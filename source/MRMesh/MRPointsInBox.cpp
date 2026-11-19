#include "MRPointsInBox.h"
#include "MRPointCloud.h"
#include "MRMesh.h"
#include "MRAABBTreePointsTraversal.hpp"
#include "MRAABBTreeTraversal.hpp"
#include "MRInplaceStack.h"

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

    if ( tree.empty() )
        return;

    using namespace Simd4;
    const auto& orderedPoints = tree.orderedPoints();
    const auto& simdNodes = tree.simdNodes();

    // every level adds at most MaxChildren - 1 tasks
    InplaceStack<int, 3 * AABBTreePoints::MaxDepth + 1> subtasks;
    {
        const auto rootBox = tree.getBoundingBox();
        if ( xf ? transformed( rootBox, *xf ).intersects( box ) : rootBox.intersects( box ) )
            subtasks.push( tree.rootRef() );
    }

    while ( !subtasks.empty() )
    {
        const int ref = subtasks.top();
        subtasks.pop();

        if ( ref < 0 )
        {
            auto [first, last] = tree.leafPointRange( ~ref );
            if ( !xf )
            {
                const F lo[3] = { set1( box.min.x ), set1( box.min.y ), set1( box.min.z ) };
                const F hi[3] = { set1( box.max.x ), set1( box.max.y ), set1( box.max.z ) };
                for ( ; first + 4 <= last; first += 4 )
                {
                    F r[4] = { loadu( &orderedPoints[first].coord.x ), loadu( &orderedPoints[first + 1].coord.x ),
                        loadu( &orderedPoints[first + 2].coord.x ), loadu( &orderedPoints[first + 3].coord.x ) };
                    transpose( r );
                    M in = andm( le( lo[0], r[0] ), le( r[0], hi[0] ) );
                    in = andm( in, andm( le( lo[1], r[1] ), le( r[1], hi[1] ) ) );
                    in = andm( in, andm( le( lo[2], r[2] ), le( r[2], hi[2] ) ) );
                    for ( int m = bits( in ); m; m &= m - 1 )
                    {
                        const auto & p = orderedPoints[first + lowestBit( m )];
                        foundCallback( p.id, p.coord );
                    }
                }
            }
            for ( int i = first; i < last; ++i )
            {
                auto coord = xf ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;
                if ( box.contains( coord ) )
                    foundCallback( orderedPoints[i].id, coord );
            }
            continue;
        }

        const auto & node = simdNodes[ref];
        int m = 0;
        if ( xf )
        {
            for ( int i = 0; i < AABBTreePoints::MaxChildren && node.child[i]; ++i )
                if ( transformed( node.get( i ), *xf ).intersects( box ) )
                    m |= 1 << i;
        }
        else
            m = simdBoxesIntersect( node, box );
        // the first child is looked at first
        for ( int i = AABBTreePoints::MaxChildren - 1; i >= 0; --i )
            if ( m & ( 1 << i ) )
                subtasks.push( node.child[i] );
    }
}

} //namespace MR
