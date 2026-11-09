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
/// <param name="subtasks">initial tasks to process</param>
/// <param name="nextSubtasks">subtasks to process next, could be same as `subtasks`</param>
/// <param name="processLeaf">function that is called for two different leafs</param>
/// <param name="processNodes">function that determines how to process two different nodes (e.g. whether their boxes intersect)</param>
/// the callbacks are template parameters to be inlined in this hot loop
template <typename ProcessLeaf, typename ProcessNodes>
MR_BIND_IGNORE void processSelfSubtasks( const AABBTree& tree,
    std::vector<NodeNode>& subtasks,
    std::vector<NodeNode>& nextSubtasks, ///< may be same as subtasks
    ProcessLeaf && processLeaf,
    ProcessNodes && processNodes )
{
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();
        const auto& aNode = tree[s.aNode];
        const auto& bNode = tree[s.bNode];

        if ( s.aNode == s.bNode )
        {
            if ( !aNode.leaf() )
            {
                nextSubtasks.push_back( { aNode.l, aNode.l } );
                nextSubtasks.push_back( { aNode.r, aNode.r } );
                nextSubtasks.push_back( { aNode.l, aNode.r } );
            }
            continue;
        }

        const auto processNode = processNodes( s );
        if ( processNode == Processing::Stop )
            continue;

        if ( aNode.leaf() && bNode.leaf() )
        {
            if ( processLeaf( s ) == Processing::Stop )
                return;
            continue;
        }

        if ( !aNode.leaf() && ( bNode.leaf() || aNode.box.volume() >= bNode.box.volume() ) )
        {
            // split aNode
            nextSubtasks.push_back( { aNode.l, s.bNode } );
            nextSubtasks.push_back( { aNode.r, s.bNode } );
        }
        else
        {
            assert( !bNode.leaf() );
            // split bNode
            nextSubtasks.push_back( { s.aNode, bNode.l } );
            nextSubtasks.push_back( { s.aNode, bNode.r } );
        }
    }
}

}