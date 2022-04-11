#include "MRSceneRoot.h"

namespace MR
{

#ifndef MR_SCENEROOT_CONST

Object& SceneRoot::get()
{
    return *instace_().root_;
}

std::shared_ptr<Object>& SceneRoot::getSharedPtr()
{
    return instace_().root_;
}

void SceneRoot::setScenePath( const std::filesystem::path& scenePath )
{
    instace_().scenePath_ = scenePath;
}

#endif

SceneRoot& SceneRoot::instace_()
{
    static SceneRoot scene;
    return scene;
}

SceneRoot::SceneRoot()
{
    root_ = std::make_shared<Object>();
    root_->setName( "Root" );
    root_->setAncillary( true );
}

const Object& SceneRoot::constGet()
{
    return *instace_().root_;
}

std::shared_ptr<const Object> SceneRoot::constGetSharedPtr()
{
    return std::const_pointer_cast<const Object>( instace_().root_ ); 
}

const std::filesystem::path& SceneRoot::getScenePath()
{
    return instace_().scenePath_;
}

}