#include "MRSceneRoot.h"

namespace MR
{

Object& SceneRoot::get()
{
    return *instace_().root_;
}

std::shared_ptr<Object>& SceneRoot::getSharedPtr()
{
    return instace_().root_;
}

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


bool SceneRoot::isScenePathEmpty()
{
    return instace_().scenePath_->empty();
}

void SceneRoot::clearScenePath()
{
    instace_().scenePath_->clear();
}

void SceneRoot::setScenePath( const std::filesystem::path& scenePath )
{
    instace_().scenePath_ = std::make_shared<std::filesystem::path>( scenePath );
}

std::filesystem::path SceneRoot::getScenePath()
{
    return *( instace_().scenePath_ );
}

std::shared_ptr<std::filesystem::path>& SceneRoot::getScenePathSharedPtr()
{
    return instace_().scenePath_;
}

std::string SceneRoot::getSceneFileName()
{
    return getScenePath().filename().string();
}

}