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

}