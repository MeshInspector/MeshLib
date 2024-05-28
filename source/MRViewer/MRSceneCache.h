#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "exports.h"
#include <optional>

namespace MR
{

// class to cached scene objects data
class MRVIEWER_CLASS SceneCache
{
public:
    // invalidate all cached data
    // call it in the beginning each frame
    MRVIEWER_API static void invalidateAll();

    // same as getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType ) but use cached data
    template <typename ObjectType, ObjectSelectivityType SelectivityType>
    static const std::vector<std::shared_ptr<ObjectType>>& getAllObjects();

    // get all selectable object depths
    // metadata for drawing scene objects list
    MRVIEWER_API static const std::vector<int>& getAllObjectsDepth();
private:
    MRVIEWER_API static SceneCache& instance_();
    SceneCache() {};

    using StoredType = std::vector<std::shared_ptr<Object>>;
    using CachedStoredType = std::optional<StoredType>;
    MRVIEWER_API static std::pair<StoredType, std::vector<int>> updateAllObjectsWithDepth_();

    std::vector<CachedStoredType> cachedData_;
    std::optional<std::vector<int>> allObjectDepths_;

    // Helper class to convert template params to unique numbers
    class MRVIEWER_CLASS TypeMap
    {
    public:
        template <typename ObjectType, ObjectSelectivityType SelectivityType>
        static int getId()
        {
            static int myId = instance_().idCounter_++;
            return myId;
        }

    private:
        MRVIEWER_API static TypeMap& instance_();

        int idCounter_ = 0;
    };
};

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
template <typename ObjectType, ObjectSelectivityType SelectivityType>
const std::vector<std::shared_ptr<ObjectType>>& SceneCache::getAllObjects()
{
    using SpecificDataType = std::vector<std::shared_ptr<ObjectType>>;
    const int templateParamsUniqueId = TypeMap::getId<ObjectType, SelectivityType>();
    if ( templateParamsUniqueId + 1 > instance_().cachedData_.size() )
        instance_().cachedData_.push_back( CachedStoredType() );
    if ( !instance_().cachedData_[templateParamsUniqueId] )
    {
        std::optional<SpecificDataType> specificData = getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType );
        std::optional<StoredType> storedData = *reinterpret_cast< std::optional<StoredType>* >( &specificData );
        instance_().cachedData_[templateParamsUniqueId] = storedData;
    }
    const SpecificDataType& resData = **reinterpret_cast< std::optional<SpecificDataType>* >( &instance_().cachedData_[templateParamsUniqueId] );
    return resData;
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// specialization getAllObjects to getAllObjects<Object, ObjectSelectivityType::Selectable>
// also calc allObjectDepths_
template <>
inline const std::vector<std::shared_ptr<Object>>& SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>()
{
    const int templateParamsUniqueId = TypeMap::getId<Object, ObjectSelectivityType::Selectable>();
    if ( templateParamsUniqueId + 1 > instance_().cachedData_.size() )
        instance_().cachedData_.push_back( CachedStoredType() );
    if ( !instance_().cachedData_[templateParamsUniqueId] )
    {
        auto data = updateAllObjectsWithDepth_();
        instance_().cachedData_[templateParamsUniqueId] = std::move( data.first );
        instance_().allObjectDepths_ = std::move( data.second );
    }
    return *instance_().cachedData_[templateParamsUniqueId];
}


}
