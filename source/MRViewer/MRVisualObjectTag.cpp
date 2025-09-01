#include "MRVisualObjectTag.h"

#include "MRMesh/MRObjectTagEventDispatcher.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRVisualObject.h"
#include "MRPch/MRJson.h"

namespace MR
{

VisualObjectTagManager& VisualObjectTagManager::instance()
{
    static VisualObjectTagManager sInstance{ ProtectedTag{} };
    return sInstance;
}

const std::map<std::string, VisualObjectTag>& VisualObjectTagManager::tags()
{
    return instance().visTags_;
}

void VisualObjectTagManager::registerTag( std::string tag, VisualObjectTag visTag )
{
    instance().visTags_.emplace( std::move( tag ), std::move( visTag ) );
}

void VisualObjectTagManager::updateTag( const std::string& tag, VisualObjectTag visTag )
{
    auto it = instance().visTags_.find( tag );
    if ( it == instance().visTags_.end() )
        return;
    it->second = std::move( visTag );
}

void VisualObjectTagManager::unregisterTag( const std::string& tag )
{
    auto it = instance().visTags_.find( tag );
    if ( it == instance().visTags_.end() )
        return;
    instance().visTags_.erase( it );
}

std::vector<std::shared_ptr<Object>> VisualObjectTagManager::getAllObjectsWithTag( Object* root, const std::string& tag, const ObjectSelectivityType& type )
{
    // TODO: more efficient version
    auto results = getAllObjectsInTree( root, type );
    std::erase_if( results, [&] ( const std::shared_ptr<Object>& obj )
    {
        return !obj->tags().contains( tag );
    } );
    return results;
}

void VisualObjectTagManager::update( VisualObject& visObj, const std::string& tag )
{
    const auto& visTags = instance().visTags_;

    if ( visObj.tags().contains( tag ) )
    {
        if ( const auto visTagIt = visTags.find( tag ); visTagIt != visTags.end() )
        {
            const auto& [_, visTag] = *visTagIt;
            visObj.setFrontColor( visTag.selectedColor, true );
            visObj.setFrontColor( visTag.unselectedColor, false );
            return;
        }
    }

    visObj.resetFrontColor();

    // re-apply existing tag if any
    for ( const auto& [knownTag, visTag] : visTags )
    {
        if ( visObj.tags().contains( knownTag ) )
        {
            visObj.setFrontColor( visTag.selectedColor, true );
            visObj.setFrontColor( visTag.unselectedColor, false );
            return;
        }
    }
}

VisualObjectTagManager::VisualObjectTagManager( ProtectedTag )
{
    auto& objectTagManager = ObjectTagEventDispatcher::instance();
    const auto slot = [this] ( Object* obj, const std::string& tag )
    {
        if ( !visTags_.contains( tag ) )
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

    auto& visTags = manager.visTags_;
    for ( const auto& tagObj : root )
    {
        if ( !tagObj["Tag"].isString() )
            continue;
        const auto tag = tagObj["Tag"].asString();

        VisualObjectTag visTag;
        deserializeFromJson( tagObj["SelectedColor"], visTag.selectedColor );
        deserializeFromJson( tagObj["UnselectedColor"], visTag.unselectedColor );

        visTags.emplace( std::move( tag ), std::move( visTag ) );
    }
}

void serializeToJson( const VisualObjectTagManager& manager, Json::Value& root )
{
    root = Json::arrayValue;

    auto i = 0;
    for ( const auto& [tag, visTag] : manager.tags() )
    {
        auto& visTagObj = root[i++] = Json::objectValue;
        visTagObj["Tag"] = tag;
        serializeToJson( visTag.selectedColor, visTagObj["SelectedColor"] );
        serializeToJson( visTag.unselectedColor, visTagObj["UnselectedColor"] );
    }
}

} // namespace MR
