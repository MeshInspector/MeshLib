#include "MRVisualObjectTag.h"

#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRJson.h"

namespace MR
{

std::string VisualObjectTag::canonicalName() const
{
    return toLower( std::string{ trim( name ) } );
}

VisualObjectTagManager& VisualObjectTagManager::instance()
{
    static VisualObjectTagManager sInstance;
    return sInstance;
}

void deserializeFromJson( const Json::Value& root, VisualObjectTagManager& manager )
{
    if ( !root.isArray() )
        return;

    auto& storage = manager.storage();
    for ( const auto& tagObj : root )
    {
        if ( !tagObj["Id"].isString() )
            continue;
        if ( !tagObj["Name"].isString() )
            continue;

        const auto id = tagObj["Id"].asString();
        VisualObjectTag tag;
        tag.name = tagObj["Name"].asString();
        deserializeFromJson( tagObj["SelectedColor"], tag.selectedColor.get() );
        deserializeFromJson( tagObj["UnselectedColor"], tag.unselectedColor.get() );

        storage.emplace( id, tag );
    }
}

void serializeToJson( const VisualObjectTagManager& manager, Json::Value& root )
{
    root = Json::arrayValue;

    auto i = 0;
    for ( const auto& [id, tag] : manager.storage() )
    {
        auto& tagObj = root[i++] = Json::objectValue;
        tagObj["Id"] = id;
        tagObj["Name"] = tag.name;
        serializeToJson( tag.selectedColor.get(), tagObj["SelectedColor"] );
        serializeToJson( tag.unselectedColor.get(), tagObj["UnselectedColor"] );
    }
}

} // namespace MR
