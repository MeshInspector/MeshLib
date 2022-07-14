#include "MROpen.h"
#include "MRMeshViewer.h"
#include "MRAppendHistory.h"
#include "MRRibbonMenu.h"
#include <MRMesh/MRObjectLoad.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshLoadObj.h>
#include <MRMesh/MRSerializer.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRObjectMesh.h>
#include <MRMesh/MRChangeSceneAction.h>
#include <MRMesh/MRStringConvert.h>
#include <MRViewer/MRSwapRootAction.h>
#include <MRPch/MRSpdlog.h>

namespace MR
{

void Open::init( Viewer* _viewer )
{
    viewer = _viewer;
    connect( _viewer );
}

void Open::shutdown()
{
    disconnect();
}

bool Open::load_( const std::filesystem::path & filename )
{
    std::string error;

    auto ext = filename.extension().u8string();
    for ( auto& c : ext )
        c = ( char )tolower( c );

    if ( ext == u8".obj" )
    {
        auto res = MeshLoad::fromSceneObjFile( filename, false );
        if ( res.has_value() )
        {
            SCOPED_HISTORY( "Load OBJ scene" );
            auto& objs = res.value();
            for ( int i = 0; i < objs.size(); ++i )
            {
                std::shared_ptr<ObjectMesh> objectMesh = std::make_shared<ObjectMesh>();
                if ( objs[i].name.empty() )
                    objectMesh->setName( utf8string( filename.stem() ) );
                else
                    objectMesh->setName( std::move( objs[i].name ) );
                objectMesh->setMesh( std::make_shared<Mesh>( std::move( objs[i].mesh ) ) );
                objectMesh->select( true );

                AppendHistory<ChangeSceneAction>( "Load Mesh", objectMesh, ChangeSceneAction::Type::AddObject );
                SceneRoot::get().addChild( objectMesh );
            }
        }
        else
        {
            error = res.error();
        }
    }
    else if ( !SceneFileFilters.empty() && filename.extension() == SceneFileFilters.front().extension.substr( 1 ) )
    {
        auto res = deserializeObjectTree( filename );
        if ( res.has_value() )
        {
            auto resVal = *res;
            if ( resVal && resVal->name() == SceneRoot::get().name() )
            {

                AppendHistory<SwapRootAction>( "Load Scene File" );
                std::swap( resVal, SceneRoot::getSharedPtr() );
            }
            else
            {
                AppendHistory<ChangeSceneAction>( "Load Scene File", resVal, ChangeSceneAction::Type::AddObject );
                SceneRoot::get().addChild( resVal );
            }
        }
        else
        {
            error = res.error();
        }
    }
    else
    {
        auto objectMesh = makeObjectMeshFromFile( filename );
        if ( objectMesh.has_value() )
        {
            objectMesh->select( true );
            auto obj = std::make_shared<ObjectMesh>( std::move( objectMesh.value() ) );
            AppendHistory<ChangeSceneAction>( "Load Mesh", obj, ChangeSceneAction::Type::AddObject );
            SceneRoot::get().addChild( obj );
        }
        else
        {
            error = objectMesh.error();

            auto objectPoints = makeObjectPointsFromFile( filename );
            if ( objectPoints.has_value() )
            {
                objectPoints->select( true );
                auto obj = std::make_shared<ObjectPoints>( std::move( objectPoints.value() ) );
                AppendHistory<ChangeSceneAction>( "Load Points", obj, ChangeSceneAction::Type::AddObject );
                SceneRoot::get().addChild( obj );
                error.clear();
            }
            else if ( error == "unsupported file extension" )
            {
                error = objectPoints.error();

                auto objectLines = makeObjectLinesFromFile( filename );
                if ( objectLines.has_value() )
                {
                    objectLines->select( true );
                    auto obj = std::make_shared<ObjectLines>( std::move( objectLines.value() ) );
                    AppendHistory<ChangeSceneAction>( "Load Lines", obj, ChangeSceneAction::Type::AddObject );
                    SceneRoot::get().addChild( obj );
                    error.clear();
                }
                else if ( error == "unsupported file extension" )
                {
                    error = objectLines.error();
                }
            }
        }
    }

    if ( error.empty() )
    {
        viewer->onSceneSaved( filename );
        return true;
    }

    if ( auto menu = viewer->getMenuPlugin() )
        menu->showErrorModal( error );

    spdlog::error( error );
    return false;
}

bool Open::dragDrop_( const std::vector<std::filesystem::path>& paths )
{
    // if drop to menu scene window -> add objects
    // if drop to viewport -> replace objects
    auto& viewerRef = getViewerInstance();
    SCOPED_HISTORY( "Drag and drop files" );
    auto menu = viewerRef.getMenuPluginAs<RibbonMenu>();
    if ( menu )
    {
        auto sceneBoxSize = menu->getSceneSize();
        auto mousePos = viewerRef.mouseController.getMousePos();
        auto headerHeight = viewerRef.window_height - sceneBoxSize.y;
        if ( mousePos.x > sceneBoxSize.x || mousePos.y < headerHeight )
        {
            auto children = SceneRoot::get().children();
            for ( auto child : children )
            {
                AppendHistory<ChangeSceneAction>( "Remove object", child, ChangeSceneAction::Type::RemoveObject );
                child->detachFromParent();
            }
        }
    }

    bool res = false;
    for ( const auto & f : paths )
        res = load_( f ) || res;

    if (paths.size() == 1 && !SceneFileFilters.empty() && paths[0].extension() == SceneFileFilters.front().extension.substr(1))
        SceneRoot::setScenePath(paths[0]);
    else
        SceneRoot::setScenePath("");

    viewerRef.makeTitleFromSceneRootPath();

    if ( res )
        viewerRef.viewport().preciseFitDataToScreenBorder( { 0.9f } );
    return res;
}

MRVIEWER_PLUGIN_REGISTRATION( Open )

} //namespace MR
