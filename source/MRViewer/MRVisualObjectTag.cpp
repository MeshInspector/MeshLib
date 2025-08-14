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

const HashMap<std::string, VisualObjectTag>& VisualObjectTagManager::tags()
{
    return instance().tags_;
}

const std::map<std::string, std::string, VisualObjectTagManager::CaseInsensitiveCompare>& VisualObjectTagManager::tagIndex()
{
    return instance().tagIndex_;
}

std::string VisualObjectTagManager::registerTag( VisualObjectTag tag )
{
    const auto id = cVisualObjectTagPrefix + tag.canonicalName();
    registerTag( id, std::move( tag ) );
    return id;
}

void VisualObjectTagManager::registerTag( std::string id, VisualObjectTag tag )
{
    instance().tagIndex_.emplace( tag.name, id );
    instance().tags_.emplace( std::move( id ), std::move( tag ) );
}

void VisualObjectTagManager::updateTag( const std::string& visTagId, VisualObjectTag tag )
{
    auto it = instance().tags_.find( visTagId );
    if ( it == instance().tags_.end() )
        return;

    instance().tagIndex_.erase( it->second.name );
    instance().tagIndex_.emplace( tag.name, visTagId );

    it->second = std::move( tag );
}

void VisualObjectTagManager::unregisterTag( const std::string& visTagId )
{
    auto it = instance().tags_.find( visTagId );
    if ( it == instance().tags_.end() )
        return;

    instance().tagIndex_.erase( it->second.name );

    instance().tags_.erase( it );
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
    const auto& tags = instance().tags_;
    if ( visObj.tags().contains( visTagId ) )
    {
        const auto visTagIt = tags.find( visTagId );
        if ( visTagIt == tags.end() )
            return;
        const auto& [_, tag] = *visTagIt;

        visObj.setFrontColor( tag.selectedColor, true );
        visObj.setFrontColor( tag.unselectedColor, false );
    }
    else
    {
        visObj.resetFrontColor();

        // re-apply existing tag
        for ( const auto& [id, _] : tags )
            if ( visObj.tags().contains( id ) )
                return update( visObj, id );
    }
}

VisualObjectTagManager::VisualObjectTagManager( ProtectedTag )
{
    auto& objectTagManager = ObjectTagEventDispatcher::instance();
    const auto slot = [this] ( Object* obj, const std::string& tag )
    {
        if ( !tags_.contains( tag ) )
            return;

        auto* visObj = dynamic_cast<VisualObject*>( obj );
        if ( !visObj )
            return;

        update( *visObj, tag );
    };
    onTagAdded_ = objectTagManager.tagAddedSignal.connect( slot );
    onTagRemoved_ = objectTagManager.tagRemovedSignal.connect( slot );
}

bool VisualObjectTagManager::CaseInsensitiveCompare::operator()( const std::string& a, const std::string& b ) const
{
    return toLower( a ) < toLower( b );
}

void deserializeFromJson( const Json::Value& root, VisualObjectTagManager& manager )
{
    if ( !root.isArray() )
        return;

    auto& tags = manager.tags_;
    auto& tagIndex = manager.tagIndex_;
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

        tags.emplace( id, tag );
        tagIndex.emplace( tag.name, id );
    }
}

void serializeToJson( const VisualObjectTagManager& manager, Json::Value& root )
{
    root = Json::arrayValue;

    auto i = 0;
    for ( const auto& [id, tag] : manager.tags() )
    {
        auto& tagObj = root[i++] = Json::objectValue;
        tagObj["Id"] = id;
        tagObj["Name"] = tag.name;
        serializeToJson( tag.selectedColor, tagObj["SelectedColor"] );
        serializeToJson( tag.unselectedColor, tagObj["UnselectedColor"] );
    }
}

} // namespace MR
