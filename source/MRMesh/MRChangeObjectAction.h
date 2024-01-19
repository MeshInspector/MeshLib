#pragma once
#include "MRHistoryAction.h"
#include "MRObject.h"
#include "MRVisualObject.h"
#include "MRHeapBytes.h"
#include <memory>

namespace MR
{

/// History action for object change
/// \ingroup HistoryGroup
class ChangeObjectAction : public HistoryAction
{
public:
    /// Constructed from original Object
    ChangeObjectAction( const std::string& name, const std::shared_ptr<Object>& obj ):
        obj_{ obj },
        name_{name}
    {
        if ( obj )
        {
            cloneObj_ = obj->clone();

            // do not save ClippedByPlane property in Undo, TODO move the logic in SectionPlugin
            if ( auto visObj = cloneObj_->asType<VisualObject>() )
                visObj->setVisualizeProperty( false, VisualizeMaskType::ClippedByPlane, ViewportMask::all() );
        }
    }

    virtual std::string name() const override { return name_; }

    virtual void action( HistoryAction::Type ) override
    {
        if ( obj_.expired() || !cloneObj_ )
            return;

        auto obj = obj_.lock();
        auto children = obj->children();
        for ( auto& child : children )
        {
            child->detachFromParent();
            cloneObj_->addChild( child );
        }

        // do not save ClippedByPlane property in Undo, TODO move the logic in SectionPlugin
        if ( auto visObj = obj->asType<VisualObject>() )
            visObj->setVisualizeProperty( false, VisualizeMaskType::ClippedByPlane, ViewportMask::all() );

        obj->swap( *cloneObj_ );
        if ( auto visObj = obj->asType<VisualObject>() )
            visObj->setDirtyFlags( DIRTY_ALL );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + MR::heapBytes( cloneObj_ );
    }

private:
    std::weak_ptr<Object> obj_;
    std::shared_ptr<Object> cloneObj_;
    std::string name_;
};

}