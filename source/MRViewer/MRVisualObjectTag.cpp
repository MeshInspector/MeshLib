#include "MRVisualObjectTag.h"

#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRVisualObject.h"
#include "MRPch/MRJson.h"

namespace
{

constexpr const char* cVisualObjectTagPrefix = "visual-object-tag=";

} // namespace

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

std::vector<std::shared_ptr<Object>> VisualObjectTagManager::getAllObjectsWithTag( Object* root, const std::string& visTagId, const ObjectSelectivityType& type )
{
    // TODO: more efficient version
    const auto tagId = cVisualObjectTagPrefix + visTagId;
    auto results = getAllObjectsInTree( root, type );
    std::erase_if( results, [&] ( const std::shared_ptr<Object>& obj )
    {
        return !obj->getTags().contains( tagId );
    } );
    return results;
}

bool VisualObjectTagManager::hasTag( const VisualObject& visObj, const std::string& visTagId )
{
    const auto tagId = cVisualObjectTagPrefix + visTagId;
    return visObj.getTags().contains( tagId );
}

void VisualObjectTagManager::applyTag( VisualObject& visObj, const std::string& visTagId, bool force )
{
    const auto tagId = cVisualObjectTagPrefix + visTagId;
    if ( visObj.getTags().contains( tagId ) && !force )
        return;

    const auto& storage = instance().storage_;
    const auto visTagIt = storage.find( visTagId );
    if ( visTagIt == storage.end() )
        return;
    const auto& [_, tag] = *visTagIt;

    visObj.setFrontColor( tag.selectedColor, true );
    visObj.setFrontColor( tag.unselectedColor, false );
    visObj.getMutableTags().emplace( tagId );
}

void VisualObjectTagManager::revertTag( VisualObject& visObj, const std::string& visTagId, bool force )
{
    const auto tagId = cVisualObjectTagPrefix + visTagId;
    if ( !visObj.getTags().contains( tagId ) && !force )
        return;

    visObj.resetFrontColor();
    visObj.getMutableTags().erase( tagId );

    // re-apply existing tag
    const auto& storage = instance().storage_;
    for ( const auto& [id, _] : storage )
        if ( hasTag( visObj, id ) )
            return applyTag( visObj, id );
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
        deserializeFromJson( tagObj["SelectedColor"], tag.selectedColor );
        deserializeFromJson( tagObj["UnselectedColor"], tag.unselectedColor );

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
        serializeToJson( tag.selectedColor, tagObj["SelectedColor"] );
        serializeToJson( tag.unselectedColor, tagObj["UnselectedColor"] );
    }
}

} // namespace MR
