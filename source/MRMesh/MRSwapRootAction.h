#pragma once
#include "MRHistoryAction.h"
#include "MRObject.h"
#include "MRSceneRoot.h"
#include <memory>

namespace MR
{

class SwapRootAction : public HistoryAction
{
public:
    // Constructed from original root
    SwapRootAction( const std::string& name, const std::shared_ptr<Object>& root ) :
        root_{ root },
        name_{ name }
    {
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !root_ )
            return;
        std::swap( root_, SceneRoot::getSharedPtr() );
    }

private:
    std::shared_ptr<Object> root_;
    std::string name_;
};
}