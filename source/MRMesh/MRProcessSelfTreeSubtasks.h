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
/// it is called for a pair before the pair is added to `nextSubtasks`, and a pair of leaves is given to `processLeaf` at once</param>
/// the callbacks are template parameters to be inlined in this hot loop
template <typename ProcessLeaf, typename ProcessNodes>
MR_BIND_IGNORE void processSelfSubtasks( const AABBTree& tree,
    std::vector<NodeNode>& subtasks,
    std::vector<NodeNode>& nextSubtasks, ///< may be same as subtasks
    ProcessLeaf && processLeaf,
    ProcessNodes && processNodes )
{
    // the pair of children is added to nextSubtasks or given to processLeaf only if processNodes accepts it;
    // a macro, since the same code in a loop over children or in a lambda was 5..10% slower with MSVC
#define MR_PROCESS_CHILDREN_PAIR( A, B ) \
    { \
        const NodeNode c{ A, B }; \
        if ( processNodes( c ) == Processing::Continue ) \
        { \
            if ( !tree.isLeaf( c.aNode ) || !tree.isLeaf( c.bNode ) ) \
                nextSubtasks.push_back( c ); \
            else if ( processLeaf( c ) == Processing::Stop ) \
                return; \
        } \
    }
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();
        const bool aLeaf = tree.isLeaf( s.aNode );
        if ( s.aNode == s.bNode )
        {
            if ( !aLeaf )
            {
                const auto& aNode = tree[s.aNode];
                nextSubtasks.push_back( { aNode.l, aNode.l } );
                nextSubtasks.push_back( { aNode.r, aNode.r } );
                MR_PROCESS_CHILDREN_PAIR( aNode.l, aNode.r )
            }
            continue;
        }
        const bool bLeaf = tree.isLeaf( s.bNode );
        if ( aLeaf && bLeaf )
        {
            if ( processLeaf( s ) == Processing::Stop )
                return;
            continue;
        }
        if ( !aLeaf && ( bLeaf || tree[s.aNode].box.volume() >= tree[s.bNode].box.volume() ) )
        {
            // split aNode
            const auto& aNode = tree[s.aNode];
            MR_PROCESS_CHILDREN_PAIR( aNode.l, s.bNode )
            MR_PROCESS_CHILDREN_PAIR( aNode.r, s.bNode )
        }
        else
        {
            assert( !bLeaf );
            // split bNode
            const auto& bNode = tree[s.bNode];
            MR_PROCESS_CHILDREN_PAIR( s.aNode, bNode.l )
            MR_PROCESS_CHILDREN_PAIR( s.aNode, bNode.r )
        }
    }
#undef MR_PROCESS_CHILDREN_PAIR
}

}