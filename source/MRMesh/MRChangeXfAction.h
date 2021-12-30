#pragma once
#include "MRHistoryAction.h"
#include "MRObject.h"
#include "MRAffineXf3.h"
#include <memory>

namespace MR
{

class ChangeXfAction : public HistoryAction
{
public:
    // Constructed from original obj
    ChangeXfAction( const std::string& name, const std::shared_ptr<Object>& obj ) :
        obj_{ obj },
        xf_{ obj->xf() },
        name_{ name }
    {
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

private:
    std::shared_ptr<Object> obj_;
    AffineXf3f xf_;
    std::string name_;
};
}