#pragma once

#include "exports.h"

#include "MRMesh/MRColor.h"
#include "MRMesh/MRObjectsAccess.h"

#include <unordered_map>

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
    [[nodiscard]] MRVIEWER_API std::string canonicalName() const;
};

/// class for storing and changing visual object properties based on the object tags
class VisualObjectTagManager
{
public:
    /// get access to the global instance
    MRVIEWER_API static VisualObjectTagManager& instance();

    /// get read-only access to the visual object tags' storage
    MRVIEWER_API static const std::unordered_map<std::string, VisualObjectTag>& storage();

    /// add visual object tag
    MRVIEWER_API static std::string registerTag( VisualObjectTag tag );
    /// update visual object tag; linked objects are NOT updated automatically
    MRVIEWER_API static void updateTag( const std::string& visTagId, VisualObjectTag tag );
    /// remove visual object tag; linked objects are NOT updated automatically
    MRVIEWER_API static void unregisterTag( const std::string& visTagId );

    /// find all object in given object tree with the visual object tag
    MRVIEWER_API static std::vector<std::shared_ptr<Object>> getAllObjectsWithTag( Object* root, const std::string& visTagId, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable );

    /// update visual object properties according to whether given object has the visual object tag or not
    MRVIEWER_API static void update( VisualObject& visObj, const std::string& visTagId );

private:
    struct ProtectedTag {};
    explicit VisualObjectTagManager( ProtectedTag );

    boost::signals2::scoped_connection onTagAdded_;
    boost::signals2::scoped_connection onTagRemoved_;

    friend MRVIEWER_API void deserializeFromJson( const Json::Value&, VisualObjectTagManager& );

    std::unordered_map<std::string, VisualObjectTag> storage_;
};

MRVIEWER_API void deserializeFromJson( const Json::Value& root, VisualObjectTagManager& manager );
MRVIEWER_API void serializeToJson( const VisualObjectTagManager& manager, Json::Value& root );

} // namespace MR
