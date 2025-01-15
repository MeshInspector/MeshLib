#pragma once
#include "MRHistoryAction.h"
#include "MRObject.h"
#include "MRVisualObject.h"
#include "MRHeapBytes.h"
#include <memory>

namespace MR
{

/// History action for scale object change
/// \ingroup HistoryGroup
class ChangeScaleAction : public HistoryAction
{
public:
    /// Constructor that performs object scaling, and remembers inverted scale value for undoing
    ChangeScaleAction( const std::string& name, const std::shared_ptr<Object>& obj, float scale ) :
        obj_{ obj },
        name_{ name }
    {
        if ( obj )
        {
            obj->applyScale(scale);
            scale_ = 1.0f / scale;
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

        obj_->applyScale( scale_ );
        scale_ = 1.0f / scale_;
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

private:
    std::shared_ptr<Object> obj_;
    float scale_ = 1.0f;
    std::string name_;
};

}