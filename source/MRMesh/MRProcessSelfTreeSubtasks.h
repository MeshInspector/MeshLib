#pragma once
#include "MRMeshFwd.h"
#include "MRAABBTree.h"
#include "MREnums.h"
#include "MRMacros.h"
#include <cassert>
#include <vector>

namespace MR
{


/// tests a pair of children by processNodes: an accepted pair is added to nextSubtasks, or to leafPairs if both are leaves
template <typename ProcessNodes>
MR_FORCE_INLINE void processChildrenPair( const AABBTree& tree, const NodeNode& c, ProcessNodes& processNodes,
    std::vector<NodeNode>& nextSubtasks, NodeNode* leafPairs, int& numLeafPairs )
{
    if ( processNodes( c ) != Processing::Continue )
        return;
    if ( !tree.isLeaf( c.aNode ) || !tree.isLeaf( c.bNode ) )
        nextSubtasks.push_back( c );
    else
        leafPairs[numLeafPairs++] = c;
}

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
    // are given to processLeaf from one call site: more call sites were slower with MSVC and on Apple M1
    NodeNode leafPairs[2];
    int numLeafPairs = 0;
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();
        numLeafPairs = 0;
        const bool aLeaf = tree.isLeaf( s.aNode );
        if ( s.aNode == s.bNode )
        {
            if ( aLeaf )
                continue;
            const auto& aNode = tree[s.aNode];
            nextSubtasks.push_back( { aNode.l, aNode.l } );
            nextSubtasks.push_back( { aNode.r, aNode.r } );
            processChildrenPair( tree, { aNode.l, aNode.r }, processNodes, nextSubtasks, leafPairs, numLeafPairs );
        }
        else
        {
            const bool bLeaf = tree.isLeaf( s.bNode );
            if ( aLeaf && bLeaf )
            {
                leafPairs[numLeafPairs++] = s;
            }
            else if ( !aLeaf && ( bLeaf || tree[s.aNode].box.volume() >= tree[s.bNode].box.volume() ) )
            {
                // split aNode
                const auto& aNode = tree[s.aNode];
                processChildrenPair( tree, { aNode.l, s.bNode }, processNodes, nextSubtasks, leafPairs, numLeafPairs );
                processChildrenPair( tree, { aNode.r, s.bNode }, processNodes, nextSubtasks, leafPairs, numLeafPairs );
            }
            else
            {
                assert( !bLeaf );
                // split bNode
                const auto& bNode = tree[s.bNode];
                processChildrenPair( tree, { s.aNode, bNode.l }, processNodes, nextSubtasks, leafPairs, numLeafPairs );
                processChildrenPair( tree, { s.aNode, bNode.r }, processNodes, nextSubtasks, leafPairs, numLeafPairs );
            }
        }
        for ( int i = 0; i < numLeafPairs; ++i )
            if ( processLeaf( leafPairs[i] ) == Processing::Stop )
                return;
    }
}

}
