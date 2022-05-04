#pragma once

#include "MRHistoryAction.h"
#include "MRObject.h"
#include "MRHeapBytes.h"

namespace MR
{

/// History action for scene objects order change
/// \ingroup HistoryGroup
class ChangeSceneObjectsOrder : public HistoryAction
{
public:
    /// Constructed before change order
    ChangeSceneObjectsOrder( const std::string& name, const std::shared_ptr<Object>& obj ) :
        obj_( obj ),
        name_( name )
    {
        if ( obj_ )
            childrenOrder_ = obj_->children();
    }

    virtual std::string name() const override { return name_; }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;

        std::vector<std::shared_ptr<Object>> oldOrder = obj_->children();
        for ( auto& child : oldOrder )
        {
            child->detachFromParent();
        }
        for ( const auto& child : childrenOrder_ )
        {
            if ( child->parent() )
                child->detachFromParent();
            obj_->addChild( child );
        }
        childrenOrder_ = std::move( oldOrder );
    };

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return MR::heapBytes( childrenOrder_ )
            + name_.capacity();
    }

private:
    std::vector<std::shared_ptr<Object>> childrenOrder_;
    std::shared_ptr<Object> obj_;
    std::string name_;
};

}