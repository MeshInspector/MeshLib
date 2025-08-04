#pragma once

#include "exports.h"

#include "MRMesh/MRColor.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRObjectsAccess.h"

namespace Json { class Value; }

namespace MR
{

/// modified color set for visual objects
struct VisualObjectTag
{
    std::string name;

    Color selectedColor;
    Color unselectedColor;

    /// canonical name for indexing
    MRVIEWER_API std::string canonicalName() const;
};

/// class for storing and changing visual object properties based on the object tags
class VisualObjectTagManager
{
public:
    /// get access to the global instance
    MRVIEWER_API static VisualObjectTagManager& instance();

    /// get direct access to the visual object tags' storage
    const std::unordered_map<std::string, VisualObjectTag>& storage() const { return storage_; }
    std::unordered_map<std::string, VisualObjectTag>& storage() { return storage_; }

    /// find all object in given object tree with the visual object tag
    MRVIEWER_API static std::vector<std::shared_ptr<Object>> getAllObjectsWithTag( Object* root, const std::string& visTagId, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable );

    /// check if given object has the visual object tag
    MRVIEWER_API static bool hasTag( const VisualObject& visObj, const std::string& visTagId );
    /// add the tag to the object and apply related visual object properties
    MRVIEWER_API static void applyTag( VisualObject& visObj, const std::string& visTagId, bool force = false );
    /// remove the tag from the object and reset the visual object properties
    MRVIEWER_API static void revertTag( VisualObject& visObj, const std::string& visTagId, bool force = false );

private:
    std::unordered_map<std::string, VisualObjectTag> storage_;
};

MRVIEWER_API void deserializeFromJson( const Json::Value& root, VisualObjectTagManager& manager );
MRVIEWER_API void serializeToJson( const VisualObjectTagManager& manager, Json::Value& root );

} // namespace MR
