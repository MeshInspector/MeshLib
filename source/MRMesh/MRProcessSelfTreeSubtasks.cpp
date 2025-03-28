#include "MRProcessSelfTreeSubtasks.h"
#include "MRAABBTree.h"

namespace MR
{

void processSelfSubtasks( 
    const AABBTree& tree, 
    std::vector<NodeNode>& subtasks, 
    std::vector<NodeNode>& nextSubtasks, /*/< may be same as subtasks */ 
    std::function<Processing( const NodeNode& )> processLeaf, 
    std::function<Processing( const Box3f& lBox, const Box3f& rBox )> processNodes )
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

        const auto processNode = processNodes( aNode.box, bNode.box );
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