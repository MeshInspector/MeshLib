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
    template <typename ObjectType, ObjectSelectivityType SelectivityType>
    struct VectorHolder : BasicVectorHolder
    {
        ObjectList<ObjectType> value;
    };
    std::unordered_map<std::type_index, std::shared_ptr<BasicVectorHolder>> cachedData_;
};

template <typename ObjectType, ObjectSelectivityType SelectivityType>
const SceneCache::ObjectList<ObjectType>& SceneCache::getAllObjects()
{
    using ResultType = VectorHolder<ObjectType, SelectivityType>;
    const auto typeIndex = std::type_index( typeid( ResultType ) );
    auto& cachedData = instance_().cachedData_;
    if ( !cachedData.contains( typeIndex ) || !cachedData[typeIndex] )
    {
        ResultType newData;
        newData.value = getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType );
        std::shared_ptr<ResultType> newDataPtr = std::make_shared<ResultType>( std::move( newData ) );
        cachedData[typeIndex] = std::dynamic_pointer_cast<BasicVectorHolder>( newDataPtr );
    }
    return std::dynamic_pointer_cast< ResultType >( cachedData[typeIndex] )->value;
}

}
