#include "MRSceneCache.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"

namespace MR
{

void SceneCache::invalidateAll()
{
    instance_().allObjects_.reset();
    instance_().allObjectsDepth_.reset();
    instance_().selectedObjects_.reset();
}

const std::vector<std::shared_ptr<MR::Object>>& SceneCache::getAllObjects()
{
    if ( !instance_().allObjects_ )
        instance_().updateAllObjectsWithDepth_();

    return *instance_().allObjects_;
}

const std::vector<int>& SceneCache::getAllObjectsDepth()
{
    if ( !instance_().allObjectsDepth_ )
        instance_().updateAllObjectsWithDepth_();

    return *instance_().allObjectsDepth_;
}

const std::vector<std::shared_ptr<MR::Object>>& SceneCache::getSelectedObjects()
{
    if ( !instance_().selectedObjects_ )
        instance_().selectedObjects_ = getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );

    return *instance_().selectedObjects_;
}

MR::SceneCache& SceneCache::instance_()
{
    static SceneCache sceneCahce;
    return sceneCahce;
}

void SceneCache::updateAllObjectsWithDepth_()
{
    std::vector<int> vecDepth;
    std::vector<std::shared_ptr<Object>> vecObjs;
    std::function<void( std::shared_ptr<Object>, int )> checkFn;
    checkFn = [&] ( std::shared_ptr<Object> obj, int depth )
    {
        if ( !obj || obj->isAncillary() )
            return;
        vecDepth.push_back( depth );
        vecObjs.push_back( obj );
        for ( const auto& child : obj->children() )
            checkFn( child, depth + 1 );
    };
    for ( const auto& child : SceneRoot::get().children() )
        checkFn( child, 0 );

    instance_().allObjects_ = vecObjs;
    instance_().allObjectsDepth_ = vecDepth;
}

}
