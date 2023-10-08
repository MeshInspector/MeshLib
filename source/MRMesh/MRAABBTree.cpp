#include "MRAABBTree.h"
#include "MRAABBTreeMaker.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRUVSphere.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"
#include "MRRegionBoundary.h"

namespace MR
{

using BoxedFace = BoxedLeaf<FaceTreeTraits3>;

bool AABBTree::containsSameNumberOfTris( const Mesh & mesh ) const
{
    if ( mesh.topology.numValidFaces() <= 0 )
        return nodes_.size() == 0;
    return int(nodes_.size()) == getNumNodes( mesh.topology.numValidFaces() );
}

inline Box3f computeFaceBox( const Mesh & mesh, FaceId f )
{
    Box3f box;
    Vector3f a, b, c;
    mesh.getTriPoints( f, a, b, c );
    box.include( a );
    box.include( b );
    box.include( c );

    // micro expand boxes to have better precision in AABB algorithms
    box = box.insignificantlyExpanded();
    return box;
}

AABBTree::AABBTree( const MeshPart & mp )
{
    MR_TIMER

    const auto numFaces = mp.region ? (int)mp.region->count() : mp.mesh.topology.numValidFaces();
    if ( numFaces <= 0 )
        return;

    Buffer<BoxedFace> boxedFaces( numFaces );
    const bool packed = numFaces == mp.mesh.topology.faceSize();
    if ( !packed )
    {
        int n = 0;
        for ( auto f : mp.mesh.topology.getFaceIds( mp.region ) )
            boxedFaces[n++].leafId = f;
    }

    // compute aabb's of each face
    tbb::parallel_for( tbb::blocked_range<int>( 0, (int)numFaces ),
        [&]( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            FaceId f;
            if ( packed )
                boxedFaces[i].leafId = f = FaceId( i );
            else
                f = boxedFaces[i].leafId;
            boxedFaces[i].box = computeFaceBox( mp.mesh, f );
        }
    } );

    nodes_ = makeAABBTreeNodeVec( std::move( boxedFaces ) );
}

FaceBitSet AABBTree::getSubtreeFaces( NodeId subtreeRoot ) const
{
    MR_TIMER;
    FaceBitSet res;

    constexpr int MaxStackSize = 32; // to avoid allocations
    NodeId subtasks[MaxStackSize];
    int stackSize = 0;
    auto addSubTask = [&]( NodeId n )
    {
        if ( nodes_[n].leaf() )
            res.autoResizeSet( nodes_[n].leafId() );
        else
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = n;
        }
    };
    addSubTask( subtreeRoot );

    while( stackSize > 0 )
    {
        NodeId n = subtasks[--stackSize];
        addSubTask( nodes_[n].r );
        addSubTask( nodes_[n].l );
    }

    return res;
}

auto AABBTree::getSubtrees( int minNum ) const -> std::vector<NodeId>
{
    MR_TIMER;
    assert( minNum > 0 );
    std::vector<NodeId> res;
    if ( nodes_.empty() )
        return res;
    res.push_back( rootNodeId() );
    std::vector<NodeId> tmp;
    while ( res.size() < minNum )
    {
        for ( NodeId n : res )
        {
            if ( nodes_[n].leaf() )
                tmp.push_back( n );
            else
            {
                tmp.push_back( nodes_[n].l );
                tmp.push_back( nodes_[n].r );
            }
        }
        res.swap( tmp );
        if ( res.size() == tmp.size() )
            break;
        tmp.clear();
    }
    return res;
}

void AABBTree::getLeafOrder( FaceBMap & faceMap ) const
{
    MR_TIMER;
    FaceId f = 0_f;
    for ( auto & n : nodes_ )
    {
        if ( !n.leaf() )
            continue;
        faceMap.b[n.leafId()] = f++;
    }
    faceMap.tsize = int( f );
}

void AABBTree::getLeafOrderAndReset( FaceBMap & faceMap )
{
    MR_TIMER;
    FaceId f = 0_f;
    for ( auto & n : nodes_ )
    {
        if ( !n.leaf() )
            continue;
        faceMap.b[n.leafId()] = f;
        n.setLeafId( f++ );
    }
    faceMap.tsize = int( f );
}

auto AABBTree::getNodesFromFaces( const FaceBitSet & faces ) const -> NodeBitSet
{
    MR_TIMER;
    NodeBitSet res( nodes_.size() );

    // mark leaves
    BitSetParallelForAll( res, [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        res[nid] = node.leaf() && faces.test( node.leafId() );
    } );

    // mark inner nodes marching from leaves to root
    for ( NodeId nid{ nodes_.size() - 1 }; nid; --nid )
    {
        auto & node = nodes_[nid];
        if ( node.leaf() )
            continue;
        res[nid] = res.test( node.l ) || res.test( node.r );
    }

    return res;
}

void AABBTree::refit( const Mesh & mesh, const VertBitSet & changedVerts )
{
    MR_TIMER

    const auto changedFaces = getIncidentFaces( mesh.topology, changedVerts );

    // update leaf nodes
    NodeBitSet changedNodes( nodes_.size() );
    BitSetParallelForAll( changedNodes, [&]( NodeId nid )
    {
        auto & node = nodes_[nid];
        if ( !node.leaf() )
            return;
        const auto f = node.leafId();
        if ( !changedFaces.test( f ) )
            return;
        changedNodes.set( nid );
        node.box = computeFaceBox( mesh, f );
    } );

    //update not-leaf nodes
    for ( auto nid = nodes_.backId(); nid; --nid )
    {
        auto & node = nodes_[nid];
        if ( node.leaf() )
            continue;
        if ( !changedNodes.test( node.l ) && !changedNodes.test( node.r ) )
            continue;
        changedNodes.set( nid );
        node.box = nodes_[node.l].box;
        node.box.include( nodes_[node.r].box );
    }
}

TEST(MRMesh, AABBTree) 
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    AABBTree tree( sphere );
    EXPECT_EQ( tree.nodes().size(), getNumNodes( sphere.topology.numValidFaces() ) );
    EXPECT_EQ( tree[AABBTree::rootNodeId()].box, sphere.computeBoundingBox().insignificantlyExpanded() );
    EXPECT_TRUE( tree[AABBTree::rootNodeId()].l.valid() );
    EXPECT_TRUE( tree[AABBTree::rootNodeId()].r.valid() );

    assert( !tree.nodes().empty() );
    auto m = std::move( tree );
    assert( tree.nodes().empty() );

    FaceBitSet fs;
    fs.autoResizeSet( 1_f );
    AABBTree smallerTree( { sphere, &fs } );
    EXPECT_EQ( smallerTree.nodes().size(), 1 );
}

} //namespace MR
