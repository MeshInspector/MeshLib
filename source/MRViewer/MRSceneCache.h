#pragma once
#include "MRMesh/MRMeshFwd.h"
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

    // get all selectable object in tree
    // same as getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selectable )
    MRVIEWER_API static const std::vector<std::shared_ptr<Object>>& getAllObjects();
    // get all selectable object depth
    // metadata for drawing scene objects list
    MRVIEWER_API static const std::vector<int>& getAllObjectsDepth();
    // get all selected object in tree
    // same as getAllObjectsInTree<Object>( &SceneRoot::get(), ObjectSelectivityType::Selected )
    MRVIEWER_API static const std::vector<std::shared_ptr<Object>>& getSelectedObjects();
private:
    static SceneCache& instance_();
    SceneCache() {};

    static void updateAllObjectsWithDepth_();

    std::optional<std::vector<std::shared_ptr<Object>>> allObjects_;
    std::optional<std::vector<int>> allObjectsDepth_;
    std::optional<std::vector<std::shared_ptr<Object>>> selectedObjects_;
};

}
