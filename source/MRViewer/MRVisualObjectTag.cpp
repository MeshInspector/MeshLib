#include "MRVisualObjectTag.h"

#include "MRMesh/MRObjectTagEventDispatcher.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRVisualObject.h"
#include "MRPch/MRJson.h"

namespace
{

constexpr const char* cVisualObjectTagPrefix = "visual-object-tag:";

} // namespace

namespace MR
{

std::string VisualObjectTag::canonicalName() const
{
    return toLower( std::string{ trim( name ) } );
}

VisualObjectTagManager& VisualObjectTagManager::instance()
{
    static VisualObjectTagManager sInstance{ ProtectedTag{} };
    return sInstance;
}

const std::unordered_map<std::string, VisualObjectTag>& VisualObjectTagManager::storage()
{
    return instance().storage_;
}

std::string VisualObjectTagManager::registerTag( VisualObjectTag tag )
{
    const auto id = cVisualObjectTagPrefix + tag.canonicalName();
    instance().storage_.emplace( id, std::move( tag ) );
    return id;
}

void VisualObjectTagManager::updateTag( const std::string& visTagId, VisualObjectTag tag )
{
    auto it = instance().storage_.find( visTagId );
    if ( it != instance().storage_.end() )
        it->second = std::move( tag );
}

void VisualObjectTagManager::unregisterTag( const std::string& visTagId )
{
    instance().storage_.erase( visTagId );
}

std::vector<std::shared_ptr<Object>> VisualObjectTagManager::getAllObjectsWithTag( Object* root, const std::string& visTagId, const ObjectSelectivityType& type )
{
    // TODO: more efficient version
    auto results = getAllObjectsInTree( root, type );
    std::erase_if( results, [&] ( const std::shared_ptr<Object>& obj )
    {
        return !obj->tags().contains( visTagId );
    } );
    return results;
}

void VisualObjectTagManager::update( VisualObject& visObj, const std::string& visTagId )
{
    const auto& storage = instance().storage_;
    if ( visObj.tags().contains( visTagId ) )
    {
        const auto visTagIt = storage.find( visTagId );
        if ( visTagIt == storage.end() )
            return;
        const auto& [_, tag] = *visTagIt;

        visObj.setFrontColor( tag.selectedColor, true );
        visObj.setFrontColor( tag.unselectedColor, false );
    }
    else
    {
        visObj.resetFrontColor();

        // re-apply existing tag
        for ( const auto& [id, _] : storage )
            if ( visObj.tags().contains( id ) )
                return update( visObj, id );
    }
}

VisualObjectTagManager::VisualObjectTagManager( ProtectedTag )
{
    auto& objectTagManager = ObjectTagEventDispatcher::instance();
    const auto slot = [this] ( Object* obj, const std::string& tag )
    {
        if ( !storage_.contains( tag ) )
            return;

        auto* visObj = dynamic_cast<VisualObject*>( obj );
        if ( !visObj )
            return;

        update( *visObj, tag );
    };
    onTagAdded_ = objectTagManager.tagAddedSignal.connect( slot );
    onTagRemoved_ = objectTagManager.tagRemovedSignal.connect( slot );
}

void deserializeFromJson( const Json::Value& root, VisualObjectTagManager& manager )
{
    if ( !root.isArray() )
        return;

    auto& storage = manager.storage_;
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
