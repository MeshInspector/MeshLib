#pragma once
#include "MRMesh/MRMeshFwd.h"
#include <optional>

namespace MR
{

// class to cached scene objects data
class SceneCache
{
public:
    // invalidate all cached data
    static void invalidateAll();

    // get all selectable object in tree
    // same as getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selectable )
    static const std::vector<std::shared_ptr<Object>>& getAllObjects();
    // get all selectable object depth
    // metadata for drawing scene objects list
    static const std::vector<int>& getAllObjectsDepth();
    // get all selected object in tree
    // same as getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected )
    static const std::vector<std::shared_ptr<Object>>& getSelectedObjects();
private:
    static SceneCache& instance_();
    SceneCache() {};

    static void updateAllObjectsWithDepth_();

    std::optional<std::vector<std::shared_ptr<Object>>> allObjects_;
    std::optional<std::vector<int>> allObjectsDepth_;
    std::optional<std::vector<std::shared_ptr<Object>>> selectedObjects_;
};

}
