#include "MRProcessSelfTreeSubtasks.h"
#include "MRAABBTree.h"

namespace MR
{

void processSelfSubtasks( 
    const AABBTree& tree, 
    std::vector<NodeNode>& subtasks, 
    std::vector<NodeNode>& nextSubtasks, /*/< may be same as subtasks */ 
    std::function<Processing( const NodeNode& )> processLeaf, 
    std::function<Processing( const NodeNode& )> processNodes )
{
    while ( !subtasks.empty() )
    {
        const auto s = subtasks.back();
        subtasks.pop_back();
        const bool aLeaf = tree.isLeaf( s.aNode );
        const bool bLeaf = tree.isLeaf( s.bNode );
        if ( s.aNode == s.bNode )
        {
            if ( !aLeaf )
            {
                const auto& aNode = tree[s.aNode];
                nextSubtasks.push_back( { aNode.l, aNode.l } );
                nextSubtasks.push_back( { aNode.r, aNode.r } );
                nextSubtasks.push_back( { aNode.l, aNode.r } );
            }
            continue;
        }
        const auto processNode = processNodes( s );
        if ( processNode == Processing::Stop )
            continue;
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
            nextSubtasks.push_back( { aNode.l, s.bNode } );
            nextSubtasks.push_back( { aNode.r, s.bNode } );
        }
        else
        {
            assert( !bLeaf );
            // split bNode
            const auto& bNode = tree[s.bNode];
            nextSubtasks.push_back( { s.aNode, bNode.l } );
            nextSubtasks.push_back( { s.aNode, bNode.r } );
        }
    }
}

}