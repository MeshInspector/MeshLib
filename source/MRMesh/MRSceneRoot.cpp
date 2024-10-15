#include "MRSceneRoot.h"
#include "MRObjectFactory.h"
#include "MRStringConvert.h"
#include "MRPch/MRJson.h"
#include <regex>

namespace MR
{

#ifndef MR_SCENEROOT_CONST

SceneRootObject& SceneRoot::get()
{
    return *instance_().root_;
}

std::shared_ptr<SceneRootObject>& SceneRoot::getSharedPtr()
{
    return instance_().root_;
}

std::filesystem::path createNewFilePath( const std::filesystem::path& savePath )
{
    const std::regex pattern( R"(.*( \([0-9]+\))$)" );

    std::filesystem::path newPath = savePath;

    std::error_code ec;
    while ( std::filesystem::exists( newPath, ec ) )
    {
        auto name = utf8string( newPath.stem() );
        if ( std::regex_match( name, pattern ) )
        {
            auto endBracPos = name.rfind( ')' );
            auto startNumPos = name.rfind( '(' ) + 1;
            auto numStr = name.substr( startNumPos, endBracPos - startNumPos );
            int num = std::atoi( numStr.c_str() );
            name = name.substr( 0, startNumPos - 1 ) + "(" + std::to_string( num + 1 ) + ")";
        }
        else
        {
            name += " (1)";
        }
        newPath.replace_filename( asU8String( name ) + savePath.extension().u8string() );
    }
    return newPath;
}

void SceneRoot::setScenePath( const std::filesystem::path& scenePath )
{
    auto newPath = scenePath;
    if ( !scenePath.empty() && scenePath.extension().u8string() != u8".mru" )
    {
        newPath.replace_extension( ".mru" );
        newPath = createNewFilePath( newPath );
    }
    instance_().scenePath_ = newPath;
}

#endif

SceneRoot& SceneRoot::instance_()
{
    static SceneRoot scene;
    return scene;
}

SceneRoot::SceneRoot()
{
    root_ = std::make_shared<SceneRootObject>();
}

const SceneRootObject& SceneRoot::constGet()
{
    return *instance_().root_;
}

std::shared_ptr<const SceneRootObject> SceneRoot::constGetSharedPtr()
{
    return std::const_pointer_cast<const SceneRootObject >( instance_().root_ );
}

const std::filesystem::path& SceneRoot::getScenePath()
{
    return instance_().scenePath_;
}

MR_ADD_CLASS_FACTORY( SceneRootObject )

SceneRootObject::SceneRootObject()
{
    // these changes are required as root object has fixed properties
    setName( SceneRootObject::RootName() );
    setAncillary( false );
    select( false );
}

std::shared_ptr<Object> SceneRootObject::clone() const
{
    return std::make_shared<SceneRootObject>( ProtectedStruct{}, *this );
}

std::shared_ptr<SceneRootObject> SceneRootObject::cloneRoot() const
{
    return std::dynamic_pointer_cast< SceneRootObject >( clone() );
}

void SceneRootObject::serializeFields_( Json::Value& root ) const
{
    Object::serializeFields_( root );
    // append base type
    root["Type"].append( SceneRootObject::TypeName() );
}

void SceneRootObject::deserializeFields_( const Json::Value& root )
{
    Object::deserializeFields_( root );
    // these changes are required as root object has fixed properties
    setName( SceneRootObject::RootName() );
    setAncillary( false );
    select( false );
}

std::shared_ptr<SceneRootObject> createRootFormObject( std::shared_ptr<Object> obj )
{
    std::shared_ptr<SceneRootObject> root = std::make_shared<SceneRootObject>();
    auto children = obj->children();
    for ( auto child : children )
    {
        child->detachFromParent();
        root->addChild( child );
    }
    return root;
}

}