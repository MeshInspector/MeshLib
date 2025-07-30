#pragma once

#include "exports.h"

#include "MRMesh/MRColor.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRViewportProperty.h"

namespace MR
{

/// modified color set for visual objects
struct VisualObjectTag
{
    std::string name;

    ViewportProperty<Color> selectedColor;
    ViewportProperty<Color> unselectedColor;

    /// canonical name for indexing
    MRVIEWER_API std::string canonicalName() const;
};

/// storage for visual object tags
class VisualObjectTagManager
{
public:
    /// get access to the global instance
    MRVIEWER_API static VisualObjectTagManager& instance();

    /// get direct access to the visual object tags' storage
    const std::unordered_map<std::string, VisualObjectTag>& storage() const { return storage_; }
    std::unordered_map<std::string, VisualObjectTag>& storage() { return storage_; }

private:
    std::unordered_map<std::string, VisualObjectTag> storage_;
};

MRVIEWER_API void deserializeFromJson( const Json::Value& root, VisualObjectTagManager& manager );
MRVIEWER_API void serializeToJson( const VisualObjectTagManager& manager, Json::Value& root );

} // namespace MR
