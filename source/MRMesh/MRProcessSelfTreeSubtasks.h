#pragma once
#include "MRMeshFwd.h"
#include "MRAABBTree.h"
#include "MREnums.h"
#include <cassert>
#include <vector>

namespace MR
{


/// <summary>
/// This function process all subtasks in one tree\n
/// left/right == right/left in this case, so on same non-leaf nodes it only adds 3 next subtasks,\n
/// same leafs are skipped and different leafs area processed with `processLeaf` callback
/// </summary>
/// <param name="tree">the AABB tree</param>
/// <param name="subtasks">initial tasks to process: a node with itself, or two nodes already accepted by `processNodes`</param>
/// <param name="nextSubtasks">subtasks to process next, could be same as `subtasks`</param>
/// <param name="processLeaf">function that is called for two different leafs accepted by `processNodes`</param>
/// <param name="processNodes">function that determines how to process two different nodes (e.g. whether their boxes intersect);
/// it is called for a pair before the pair is added to `nextSubtasks`, and an accepted pair of leaves goes to `processLeaf` without being added</param>
/// the callbacks are template parameters to be inlined in this hot loop
template <typename ProcessLeaf, typename ProcessNodes>
MR_BIND_IGNORE void processSelfSubtasks( const AABBTree& tree,
    std::vector<NodeNode>& subtasks,
    std::vector<NodeNode>& nextSubtasks, ///< may be same as subtasks
    ProcessLeaf && processLeaf,
    ProcessNodes && processNodes )
{
    // the pair of children is added to nextSubtasks if processNodes accepts it, and the accepted pairs of leaves
    // are given to processLeaf from one call site: more call sites or a lambda were slower with MSVC and on Apple M1
    NodeNode leafPairs[2];
    int numLeafPairs = 0;
#define MR_PROCESS_CHILDREN_PAIR( A, B ) \
    { \
        const NodeNode c{ A, B }; \
        if ( processNodes( c ) == Processing::Continue ) \
        { \
            if ( !tree[c.aNode].leaf() || !tree[c.bNode].leaf() ) \
                nextSubtasks.push_back( c ); \
            else \
                leafPairs[numLeafPairs++] = c; \
        } \
    }
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();
        numLeafPairs = 0;
        const auto& aNode = tree[s.aNode];
        if ( s.aNode == s.bNode )
        {
            if ( aNode.leaf() )
                continue;
            nextSubtasks.push_back( { aNode.l, aNode.l } );
            nextSubtasks.push_back( { aNode.r, aNode.r } );
            MR_PROCESS_CHILDREN_PAIR( aNode.l, aNode.r )
        }
        else
        {
            const auto& bNode = tree[s.bNode];
            if ( aNode.leaf() && bNode.leaf() )
            {
                leafPairs[numLeafPairs++] = s;
            }
            else if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
            {
                // split aNode
                MR_PROCESS_CHILDREN_PAIR( aNode.l, s.bNode )
                MR_PROCESS_CHILDREN_PAIR( aNode.r, s.bNode )
            }
            else
            {
                assert( !bNode.leaf() );
                // split bNode
                MR_PROCESS_CHILDREN_PAIR( s.aNode, bNode.l )
                MR_PROCESS_CHILDREN_PAIR( s.aNode, bNode.r )
            }
        }
        for ( int i = 0; i < numLeafPairs; ++i )
            if ( processLeaf( leafPairs[i] ) == Processing::Stop )
                return;
    }
#undef MR_PROCESS_CHILDREN_PAIR
}

}
