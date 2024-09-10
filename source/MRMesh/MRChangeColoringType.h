#pragma once
#include "MRHistoryAction.h"
#include "MRVisualObject.h"
#include <memory>

namespace MR
{

/// History action for ColoringType change
/// \ingroup HistoryGroup
class ChangeColoringType : public HistoryAction
{
public:
    using Obj = VisualObject;

    /// use this constructor to remember object's coloring type before making any changes in it
    ChangeColoringType( const std::string& name, const std::shared_ptr<VisualObject>& obj ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj )
            coloringType_ = obj->getColoringType();
    }

    /// use this constructor to remember object's coloring type and immediate set new value
    ChangeColoringType( const std::string& name, const std::shared_ptr<VisualObject>& obj, ColoringType newType ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj_ )
        {
            coloringType_ =  obj_->getColoringType();
            obj_->setColoringType( newType );
        }
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        auto c = obj_->getColoringType();
        obj_->setColoringType( coloringType_ );
        coloringType_ = c;
    }

    static void setObjectDirty( const std::shared_ptr<VisualObject>& obj )
    {
        if ( obj )
            obj->setDirtyFlags( DIRTY_VERTS_COLORMAP );
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<VisualObject> obj_;
    ColoringType coloringType_ = ColoringType::SolidColor;
    std::string name_;
};

}
