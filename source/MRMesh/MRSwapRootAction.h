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
    SwapRootAction( const std::string& name ) :
        root_{ SceneRoot::getSharedPtr() },
        scenePath_{ SceneRoot::getScenePathSharedPtr() },
        name_{ name }
    {
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !root_ || !scenePath_ )
            return;
        std::swap( root_, SceneRoot::getSharedPtr() );
        std::swap( scenePath_, SceneRoot::getScenePathSharedPtr() );
    }

private:
    std::shared_ptr<Object> root_;
    std::shared_ptr<std::filesystem::path> scenePath_;
    std::string name_;
};
}