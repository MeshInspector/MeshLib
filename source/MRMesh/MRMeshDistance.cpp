#include "MRMeshDistance.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRTriDist.h"
#include "MRTimer.h"

namespace MR
{

void processCloseTriangles( const MeshPart& mp, const Triangle3f & t, float rangeSq, const TriangleCallback & call )
{
    assert( call );
    if ( !call )
        return;

    const AABBTree & tree = mp.mesh.getAABBTree();
    if ( tree.nodes().empty() )
        return;

    Box3f tbox;
    for ( const auto & p : t )
        tbox.include( p );

    struct SubTask
    {
        NodeId n;
        float distSq;
        SubTask() : n( noInit ) {}
        SubTask( NodeId n, float dd ) : n( n ), distSq( dd ) {}
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( const SubTask & s )
    {
        if ( s.distSq < rangeSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&]( NodeId n )
    {
        float distSq = tree.nodes()[n].box.getDistanceSq( tbox );
        return SubTask( n, distSq );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto & node = tree[s.n];

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            if ( mp.region && !mp.region->test( face ) )
                continue;
            const auto leafTriangle = mp.mesh.getTriPoints( face );
            Vector3f p, q;
            const float distSq = TriDist( p, q, t.data(), leafTriangle.data() );
            if ( distSq > rangeSq )
                continue;
            if ( call( p, face, q, distSq ) == ProcessOneResult::ContinueProcessing )
                continue;
            break;
        }
        
        addSubTask( getSubTask( node.r ) ); // right to look later
        addSubTask( getSubTask( node.l ) ); // left to look first
    }
}

} //namespace MR
