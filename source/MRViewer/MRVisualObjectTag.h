#pragma once

#include "exports.h"

#include "MRMesh/MRColor.h"
#include "MRMesh/MRObjectsAccess.h"

namespace Json { class Value; }

namespace MR
{

/// modified color set for visual objects
struct VisualObjectTag
{
    Color selectedColor;
    Color unselectedColor;
};

/// class for storing and changing visual object properties based on the object tags
class VisualObjectTagManager
{
public:
    /// get access to the global instance
    MRVIEWER_API static VisualObjectTagManager& instance();

    /// get read-only access to the visual object tags' storage
    MRVIEWER_API static const std::map<std::string, VisualObjectTag>& tags();

    /// add visual object tag
    MRVIEWER_API static void registerTag( std::string tag, VisualObjectTag visTag );
    /// update visual object tag; linked objects are NOT updated automatically
    MRVIEWER_API static void updateTag( const std::string& tag, VisualObjectTag visTag );
    /// remove visual object tag; linked objects are NOT updated automatically
    MRVIEWER_API static void unregisterTag( const std::string& tag );

    /// find all object in given object tree with the visual object tag
    MRVIEWER_API static std::vector<std::shared_ptr<Object>> getAllObjectsWithTag( Object* root, const std::string& tag, const ObjectSelectivityType& type = ObjectSelectivityType::Selectable );

    /// update visual object properties according to whether given object has the visual object tag or not
    MRVIEWER_API static void update( VisualObject& visObj, const std::string& tag );

private:
    struct ProtectedTag {};
    explicit VisualObjectTagManager( ProtectedTag );

    boost::signals2::scoped_connection onTagAdded_;
    boost::signals2::scoped_connection onTagRemoved_;

    friend MRVIEWER_API void deserializeFromJson( const Json::Value&, VisualObjectTagManager& );

    std::map<std::string, VisualObjectTag> visTags_;
};

MRVIEWER_API void deserializeFromJson( const Json::Value& root, VisualObjectTagManager& manager );
MRVIEWER_API void serializeToJson( const VisualObjectTagManager& manager, Json::Value& root );

} // namespace MR
