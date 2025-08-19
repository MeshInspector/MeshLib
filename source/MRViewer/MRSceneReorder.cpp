#include "MRSceneReorder.h"
#include "MRAppendHistory.h"
#include <MRMesh/MRObject.h>
#include <MRMesh/MRChangeSceneAction.h>
#include <MRMesh/MRChangeXfAction.h>

namespace MR
{

bool sceneReorderWithUndo( const SceneReorder & task )
{
    const bool filledReorderCommand = !task.who.empty() && task.to;
    const bool sourceNotTarget = std::all_of( task.who.begin(), task.who.end(), [target = task.to] ( auto it )
    {
        return it != target;
    } );
    const bool trueTarget = !task.before || task.to->parent();
    const bool trueSource = std::all_of( task.who.begin(), task.who.end(), [] ( auto it )
    {
        return bool( it->parent() );
    } );
    if ( !( filledReorderCommand && sourceNotTarget && trueSource && trueTarget ) )
        return false;

    bool dragOrDropFailed = false;
    std::shared_ptr<Object> childTo = nullptr;
    if ( task.before )
    {
        for ( auto childToItem : task.to->parent()->children() )
            if ( childToItem.get() == task.to )
            {
                childTo = childToItem;
                break;
            }
        assert( childTo );
    }

    struct MoveAction
    {
        std::shared_ptr<ChangeSceneAction> detachAction;
        std::shared_ptr<ChangeSceneAction> attachAction;
        std::shared_ptr<ChangeXfAction> xfAction;
    };
    std::vector<MoveAction> actionList;
    for ( const auto& source : task.who )
    {
        Object * fromParent = source->parent();
        assert( fromParent );
        std::shared_ptr<Object> sourcePtr = source->getSharedPtr();
        assert( sourcePtr );
        const auto worldXf = source->worldXf();

        auto detachAction = std::make_shared<ChangeSceneAction>( "Detach object", sourcePtr, ChangeSceneAction::Type::RemoveObject );
        bool detachSuccess = sourcePtr->detachFromParent();
        if ( !detachSuccess )
        {
            dragOrDropFailed = true;
            break;
        }

        auto attachAction = std::make_shared<ChangeSceneAction>( "Attach object", sourcePtr, ChangeSceneAction::Type::AddObject );
        bool attachSucess{ false };
        Object * toParent;
        if ( !task.before )
        {
            toParent = task.to;
            attachSucess = toParent->addChild( sourcePtr );
        }
        else
        {
            toParent = task.to->parent();
            attachSucess = toParent->addChildBefore( sourcePtr, childTo );
        }
        if ( !attachSucess )
        {
            detachAction->action( HistoryAction::Type::Undo );
            dragOrDropFailed = true;
            break;
        }

        // change xf to preserve world location of the object
        std::shared_ptr<ChangeXfAction> xfAction;
        const auto fromParentXf = fromParent->worldXf();
        const auto toParentXf = toParent->worldXf();
        if ( fromParentXf != toParentXf )
        {
            xfAction = std::make_shared<ChangeXfAction>( "Xf", sourcePtr );
            source->setWorldXf( worldXf );
        }

        actionList.push_back( { detachAction, attachAction, xfAction } );
    }

    if ( dragOrDropFailed )
    {
        for ( int i = int( actionList.size() ) - 1; i >= 0; --i )
        {
            actionList[i].attachAction->action( HistoryAction::Type::Undo );
            actionList[i].detachAction->action( HistoryAction::Type::Undo );
            if ( actionList[i].xfAction )
                actionList[i].xfAction->action( HistoryAction::Type::Undo );
        }
    }
    else
    {
        SCOPED_HISTORY( task.historyName );
        for ( const auto& moveAction : actionList )
        {
            AppendHistory( moveAction.detachAction );
            AppendHistory( moveAction.attachAction );
            if ( moveAction.xfAction )
                AppendHistory( moveAction.xfAction );
        }
    }
    return true;
}

bool moveAllChildrenWithUndo( Object& oldParent, Object& newParent, const std::string& historyName )
{
    if ( &oldParent == &newParent )
        return false;

    SceneReorder task
    {
        .to = &newParent,
        .historyName = historyName
    };
    for ( const auto& child : oldParent.children() )
    {
        if ( child->isAncillary() )
            continue;
        task.who.push_back( child.get() );
    }
    return sceneReorderWithUndo( task );
}

} //namespace MR
