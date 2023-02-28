#pragma once
#include "MRHistoryAction.h"
#include "MRVisualObject.h"
#include <memory>

namespace MR
{

/// History action for xf change
/// \ingroup HistoryGroup
class ChangeVertsColorMapAction : public HistoryAction
{
public:
    using Obj = VisualObject;
    /// Constructed from original obj
    ChangeVertsColorMapAction( const std::string& name, const std::shared_ptr<VisualObject>& obj ) :
        obj_{ obj },        
        name_{ name }
    {
        if ( obj )
            vertsColorMap_ = obj->getVertsColorMap();
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        obj_->updateVertsColorMap( vertsColorMap_ );
    }

    static void setObjectDirty( const std::shared_ptr<VisualObject>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_VERTS_COLORMAP );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity() + vertsColorMap_.capacity();
    }

private:
    std::shared_ptr<VisualObject> obj_;
    Vector<Color, VertId> vertsColorMap_;
    std::string name_;
};

}
