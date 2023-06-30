#include "MRChangeSceneAction.h"
#include "MRHeapBytes.h"
#include "MRPch/MRSpdlog.h"

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
        {
            if ( !parent_->addChildBefore( obj_, nextObj_ ) )
            {
                spdlog::warn( "ChangeSceneAction: could not find next object \"{}\" in scene to add \"{}\"", 
                    nextObj_->name(), obj_->name() );
                parent_->addChild( obj_ );
            }
        }
        else
            parent_->addChild( obj_ );
    }
}

size_t ChangeSceneAction::heapBytes() const
{
    // if type_ == Type::AddObject: obj_ has at least two owners - scene and this action
    // so this action does not really hold it, so we do not count it as part of action
    // otherwise - action is (in common case) the only owner of obj_, so we count it
    //
    // note: calling undo makes scene own object again, but it does not really matter 
    // because we only interested in `heapBytes` for undo part of stack
    return name_.capacity() + ( type_ == Type::RemoveObject ? MR::heapBytes( obj_ ) : 0 );
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
        if ( child->isAncillary() )
            continue;
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