#pragma once
#include "MRHistoryAction.h"
#include "MRObject.h"
#include "MRAffineXf3.h"
#include <memory>

namespace MR
{

/// History action for xf change
/// \ingroup HistoryGroup
class ChangeXfAction : public HistoryAction
{
public:
    /// use this constructor to remember object's transformation before making any changes in it
    ChangeXfAction( const std::string& name, const std::shared_ptr<Object>& obj ) :
        obj_{ obj },
        xf_{ obj->xf() },
        name_{ name }
    {
    }

    /// use this constructor to remember object's transformation and immediately set new mesh
    ChangeXfAction( const std::string& name, const std::shared_ptr<Object>& obj, const AffineXf3f& newXf ) :
        obj_{ obj },
        xf_{ obj->xf() },
        name_{ name }
    {
        if ( obj_ )
            obj_->setXf( newXf );
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;
        auto tmpXf = obj_->xf();
        obj_->setXf( xf_ );
        xf_ = tmpXf;
    }

    [[nodiscard]] virtual size_t heapBytes() const override
    {
        return name_.capacity();
    }

    const std::shared_ptr<Object> & obj() const
    {
        return obj_;
    }

private:
    std::shared_ptr<Object> obj_;
    AffineXf3f xf_;
    std::string name_;
};

} //namespace MR
