#include "MRChangeSceneAction.h"
#include "MRHeapBytes.h"

namespace MR
{

ChangeSceneAction::ChangeSceneAction( const std::string& name, const std::shared_ptr<Object>& obj, Type type ):
    obj_{obj},
    name_{name},
    type_{type}
{
    updateParent_();
}

void ChangeSceneAction::action( HistoryAction::Type actionType )
{
    if ( !obj_ )
        return;
    if ( ( type_ == Type::AddObject && actionType == HistoryAction::Type::Undo ) ||
        ( type_ == Type::RemoveObject && actionType == HistoryAction::Type::Redo ) )
    {
        updateParent_();
        obj_->detachFromParent();
    }
    else
    {
        if ( !parent_ )
            return;
        if ( nextObj_ )
            parent_->addChildBefore( obj_, nextObj_ );
        else
            parent_->addChild( obj_ );
    }
}

size_t ChangeSceneAction::heapBytes() const
{
    return name_.capacity() + MR::heapBytes( obj_ );
}

void ChangeSceneAction::updateParent_()
{
    if ( parent_ )
        return;
    parent_ = obj_->parent();
    if ( !parent_ )
        return;
    bool foundNext = false;
    for ( const auto& child : parent_->children() )
    {
        if ( foundNext )
        {
            nextObj_ = child;
            break;
        }
        if ( child == obj_ )
            foundNext = true;
    }
}

}