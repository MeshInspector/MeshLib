#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "exports.h"

namespace MR
{

// class to cached scene objects data
class MRVIEWER_CLASS SceneCache
{
public:
    // invalidate all cached data
    // call it in the beginning each frame
    MRVIEWER_API static void invalidateAll();

    // analog getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType ) but use cached data
    // return shared_pointr to data
    template <typename ObjectType, ObjectSelectivityType SelectivityType>
    static std::shared_ptr<std::vector<std::shared_ptr<ObjectType>>> getAllObjects();

    // get all selectable object depths
    // metadata for drawing scene objects list
    MRVIEWER_API static std::shared_ptr<std::vector<int>> getAllObjectsDepth();
private:
    MRVIEWER_API static SceneCache& instance_();
    SceneCache() {};

    using StoredType = std::vector<std::shared_ptr<Object>>;
    using CachedStoredType = std::shared_ptr<StoredType>; // store data as smart pointer to avoid data corruption
    MRVIEWER_API static std::pair<StoredType, std::vector<int>> updateAllObjectsWithDepth_();

    std::vector<CachedStoredType> cachedData_;
    std::shared_ptr<std::vector<int>> allObjectDepths_;

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
std::shared_ptr<std::vector<std::shared_ptr<ObjectType>>> SceneCache::getAllObjects()
{
    using SpecificType = std::vector<std::shared_ptr<ObjectType>>;
    using CachedSpecificType = std::shared_ptr<SpecificType>;
    const int templateParamsUniqueId = TypeMap::getId<ObjectType, SelectivityType>();
    if ( templateParamsUniqueId + 1 > instance_().cachedData_.size() )
        instance_().cachedData_.push_back( std::make_shared<StoredType>() );
    if ( !instance_().cachedData_[templateParamsUniqueId] )
    {
        CachedSpecificType specificData = std::make_shared<SpecificType>( getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType ) );
        CachedStoredType storedData = std::reinterpret_pointer_cast<StoredType>( specificData );
        instance_().cachedData_[templateParamsUniqueId] = storedData;
    }
    return std::reinterpret_pointer_cast<SpecificType>( instance_().cachedData_[templateParamsUniqueId] );
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// specialization getAllObjects to getAllObjects<Object, ObjectSelectivityType::Selectable>
// also calc allObjectDepths_
template <>
inline std::shared_ptr<std::vector<std::shared_ptr<Object>>> SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>()
{
    const int templateParamsUniqueId = TypeMap::getId<Object, ObjectSelectivityType::Selectable>();
    if ( templateParamsUniqueId + 1 > instance_().cachedData_.size() )
        instance_().cachedData_.push_back( std::make_shared<StoredType>() );
    if ( !instance_().cachedData_[templateParamsUniqueId] )
    {
        auto combinedData = updateAllObjectsWithDepth_();
        instance_().cachedData_[templateParamsUniqueId] = std::make_shared<StoredType>( std::move( combinedData.first ) );
        instance_().allObjectDepths_ = std::make_shared<std::vector<int>>( std::move( combinedData.second ) );
    }
    return instance_().cachedData_[templateParamsUniqueId];
}


}
