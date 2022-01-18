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
        scenePath_{ SceneRoot::getScenePath() },
        name_{ name }
    {
    }

    virtual std::string name() const override
    {
        return name_;
    }

    virtual void action( HistoryAction::Type ) override
    {
        if ( !root_  )
            return;
        std::swap( root_, SceneRoot::getSharedPtr() );

        auto scenePathClone = SceneRoot::getScenePath();
        SceneRoot::setScenePath( scenePath_ );
        scenePath_ = scenePathClone;
    }

private:
    std::shared_ptr<Object> root_;
    std::filesystem::path scenePath_;
    std::string name_;
};
}