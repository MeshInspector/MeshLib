#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "exports.h"
#include <unordered_map>

namespace MR
{

// class to cached scene objects data
class MRVIEWER_CLASS SceneCache
{
public:
    // invalidate all cached data
    // call it in the beginning each frame
    MRVIEWER_API static void invalidateAll();

    template <typename ObjectType>
    using ObjectList = std::vector<std::shared_ptr<ObjectType>>;
    // analog getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType ) but use cached data
    // reference copy is valid until invalidateAll() is called
    template <typename ObjectType, ObjectSelectivityType SelectivityType>
    static const ObjectList<ObjectType>& getAllObjects();

private:
    MRVIEWER_API static SceneCache& instance_();
    SceneCache() = default;

    struct BasicVectorHolder
    {
        BasicVectorHolder() = default;
        BasicVectorHolder( const BasicVectorHolder& ) = default;
        BasicVectorHolder( BasicVectorHolder&& ) = default;
        virtual ~BasicVectorHolder() = default;
    };
    template <typename ObjectType>
    struct VectorHolder : BasicVectorHolder
    {
        ObjectList<ObjectType> value;
    };
    std::unordered_map<std::string, std::shared_ptr<BasicVectorHolder>> cachedData_;
};

template <typename ObjectType, ObjectSelectivityType SelectivityType>
const SceneCache::ObjectList<ObjectType>& SceneCache::getAllObjects()
{
    using ResultType = VectorHolder<ObjectType>;
    const auto typeName = std::to_string( std::is_const_v<ObjectType> ) + ObjectType::TypeName() + std::to_string( int( SelectivityType ) );
    auto& cachedData = instance_().cachedData_;
    auto& cachedVec = cachedData[typeName];
    if ( !cachedVec )
    {
        auto dataList = std::make_shared<ResultType>();
        dataList->value = getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType );
        cachedVec = dataList;
    }
    assert( cachedVec );
    auto resPtr = dynamic_pointer_cast< ResultType >( cachedVec );
    assert( resPtr );
    return resPtr->value;
}

}
