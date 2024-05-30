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

    template <typename ObjectType>
    using TemplateDataType = std::vector<std::shared_ptr<ObjectType>>;
    // analog getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType ) but use cached data
    // reference is valid until invalidateAll() is called
    template <typename ObjectType, ObjectSelectivityType SelectivityType>
    static const TemplateDataType<ObjectType>& getAllObjects();

    template <typename ObjectType>
    using CachedTemplateDataType = std::shared_ptr<TemplateDataType<ObjectType>>;
    // analog getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType ) but use cached data
    // shared ownership of cached data for current frame( not updated for new frame )
    template <typename ObjectType, ObjectSelectivityType SelectivityType>
    static CachedTemplateDataType<ObjectType> getAllObjectsCached();

    using DepthDataType = std::vector<int>;
    // get all selectable object depths
    // metadata for drawing scene objects list
    // reference is valid until invalidateAll() is called
    MRVIEWER_API static const DepthDataType& getAllObjectsDepth();
private:
    MRVIEWER_API static SceneCache& instance_();
    SceneCache() {};

    using StoredDataType = std::vector<std::shared_ptr<Object>>;
    using CachedStoredDataType = std::shared_ptr<StoredDataType>; // store data as smart pointer to avoid data corruption
    MRVIEWER_API static std::pair<StoredDataType, std::vector<int>> updateAllObjectsWithDepth_();

    std::vector<CachedStoredDataType> cachedData_;
    using CachedDepthDataType = std::shared_ptr<DepthDataType>;
    CachedDepthDataType allObjectDepths_;

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
SceneCache::CachedTemplateDataType<ObjectType> SceneCache::getAllObjectsCached()
{
    using SpecificType = TemplateDataType<ObjectType>;
    using CachedSpecificType = CachedTemplateDataType<ObjectType>;
    const int templateParamsUniqueId = TypeMap::getId<ObjectType, SelectivityType>();
    if ( templateParamsUniqueId + 1 > instance_().cachedData_.size() )
        instance_().cachedData_.push_back( std::make_shared<StoredDataType>() );
    if ( !instance_().cachedData_[templateParamsUniqueId] )
    {
        CachedSpecificType cachedSpecificData = std::make_shared<SpecificType>( getAllObjectsInTree<ObjectType>( &SceneRoot::get(), SelectivityType ) );
        CachedStoredDataType cachedStoredData = std::reinterpret_pointer_cast<StoredDataType>( cachedSpecificData );
        instance_().cachedData_[templateParamsUniqueId] = cachedStoredData;
    }
    return std::reinterpret_pointer_cast<SpecificType>( instance_().cachedData_[templateParamsUniqueId] );
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

template <typename ObjectType, ObjectSelectivityType SelectivityType>
const SceneCache::TemplateDataType<ObjectType>& SceneCache::getAllObjects()
{
    return *instance_().getAllObjectsCached<ObjectType, SelectivityType>();
}

// specialization getAllObjectsCached to getAllObjects<Object, ObjectSelectivityType::Selectable>
// also calc allObjectDepths_
template <>
inline SceneCache::CachedTemplateDataType<Object> SceneCache::getAllObjectsCached<Object, ObjectSelectivityType::Selectable>()
{
    const int templateParamsUniqueId = TypeMap::getId<Object, ObjectSelectivityType::Selectable>();
    if ( templateParamsUniqueId + 1 > instance_().cachedData_.size() )
        instance_().cachedData_.push_back( std::make_shared<StoredDataType>() );
    if ( !instance_().cachedData_[templateParamsUniqueId] )
    {
        auto combinedData = updateAllObjectsWithDepth_();
        instance_().cachedData_[templateParamsUniqueId] = std::make_shared<StoredDataType>( std::move( combinedData.first ) );
        instance_().allObjectDepths_ = std::make_shared<DepthDataType>( std::move( combinedData.second ) );
    }
    return instance_().cachedData_[templateParamsUniqueId];
}

template <>
inline const SceneCache::TemplateDataType<Object>& SceneCache::getAllObjects<Object, ObjectSelectivityType::Selectable>()
{
    return *instance_().getAllObjectsCached<Object, ObjectSelectivityType::Selectable>();
}

}
