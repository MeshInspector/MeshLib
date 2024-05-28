#include "MRSceneCache.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRTimer.h"

#include <iostream>

namespace MR
{

void SceneCache::invalidateAll()
{
    instance_().allObjectDepths_.reset();
    for ( auto& data : instance_().cachedData_ )
        data.reset();
}

const std::vector<int>& SceneCache::getAllObjectsDepth()
{
    const int templateParamsUniqueId = TypeMap::getId<Object, ObjectSelectivityType::Selectable>();
    if ( templateParamsUniqueId + 1 > instance_().cachedData_.size() )
        instance_().cachedData_.push_back( CachedStoredType() );
    if ( !instance_().cachedData_[templateParamsUniqueId] )
    {
        auto data = updateAllObjectsWithDepth_();
        instance_().cachedData_[templateParamsUniqueId] = std::make_shared<StoredType>( std::move( data.first ) );
        instance_().allObjectDepths_ = std::make_shared<std::vector<int>>( std::move( data.second ) );
    }
    return *instance_().allObjectDepths_;
}

MR::SceneCache& SceneCache::instance_()
{
    static SceneCache sceneCahce;
    return sceneCahce;
}

std::pair<SceneCache::StoredType, std::vector<int>> SceneCache::updateAllObjectsWithDepth_()
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

    return { std::move( vecObjs ), std::move( vecDepth ) };
}

}
