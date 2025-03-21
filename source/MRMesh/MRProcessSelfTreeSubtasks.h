#pragma once
#include "MRMeshFwd.h"
#include "MRAABBTreeNode.h"
#include "MREnums.h"
#include <functional>

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
/// <param name="processNodes">function that determines how to process boxes</param>
void processSelfSubtasks( const AABBTree& tree,
    std::vector<NodeNode>& subtasks,
    std::vector<NodeNode>& nextSubtasks, ///< may be same as subtasks
    std::function<Processing( const NodeNode& )> processLeaf,
    std::function<Processing( const Box3f& lBox, const Box3f& rBox )> processNodes );

}