#pragma once

#include "MRHistoryAction.h"
#include "MRObject.h"
#include <memory>
#include <string>

namespace MR
{

// This action to undo/redo the change of object name
class ChangeNameAction : public HistoryAction
{
public:
    // construct before giving new name to the object
    ChangeNameAction( const std::string& actionName, std::shared_ptr<Object> obj ) :
        obj_{ std::move( obj ) },
        actionName_{ actionName }
    {
        objName_ = obj_->name();
    }

    virtual std::string name() const override
    {
        return actionName_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !obj_ )
            return;

        auto n = obj_->name();
        obj_->setName( std::move( objName_ ) );
        objName_ = std::move( n );
    }

private:
    std::shared_ptr<Object> obj_;
    std::string objName_;

    std::string actionName_;
};

} //namespace MR
