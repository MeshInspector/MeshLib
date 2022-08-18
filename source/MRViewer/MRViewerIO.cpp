#include "MRViewerIO.h"

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRVoxelsSave.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRMeshLoadObj.h"
#include "MRViewerInstance.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRViewer/MRAppendHistory.h"

namespace MR
{

tl::expected<void, std::string> saveObjectToFile( const Object& obj, const std::filesystem::path& filename, ProgressCallback callback )
{
    if ( callback && !callback( 0.f ) )
        return tl::make_unexpected( std::string( "Saving canceled" ) );

    tl::expected<void, std::string> result;

    if ( auto objPoints = obj.asType<ObjectPoints>() )
    {
        if ( objPoints->pointCloud() )
        {
            const auto& colors = objPoints->getVertsColorMap();
            result = PointsSave::toAnySupportedFormat( *objPoints->pointCloud(), filename,
                                                         colors.empty() ? nullptr : &colors, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectPoints has no PointCloud in it" ) );
    }
    else if ( auto objLines = obj.asType<ObjectLines>() )
    {
        if ( objLines->polyline() )
        {
            result = LinesSave::toAnySupportedFormat( *objLines->polyline(), filename, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectLines has no Polyline in it" ) );
    }
    else if ( auto objMesh = obj.asType<ObjectMesh>() )
    {
        if ( objMesh->mesh() )
        {
            const Vector<Color, VertId>* colors{ nullptr };
            if ( objMesh->getColoringType() == ColoringType::VertsColorMap )
                colors = &objMesh->getVertsColorMap();

            result = MeshSave::toAnySupportedFormat( *objMesh->mesh(), filename, colors, callback );
        }
        else
            result = tl::make_unexpected( std::string( "ObjectMesh has no Mesh in it" ) );
    }
#ifndef __EMSCRIPTEN__
    else if ( auto objVoxels = obj.asType<ObjectVoxels>() )
    {
        auto ext = filename.extension().u8string();
        for ( auto& c : ext )
            c = ( char )tolower( c );

        if ( ext == u8".raw" )
        {
            result = VoxelsSave::saveRAW( filename, *objVoxels, callback );
        }
    }
#endif

    if ( !result.has_value() )
        spdlog::error( result.error() );

    return result;
}

tl::expected<std::vector<std::shared_ptr<MR::Object>>, std::string> loadObjectFromFile( const std::filesystem::path& filename,
                                                                                        ProgressCallback callback )
{
    if ( callback && !callback( 0.f ) )
        return tl::make_unexpected( std::string( "Saving canceled" ) );

    tl::expected<std::vector<std::shared_ptr<Object>>, std::string> result;

    auto ext = filename.extension().u8string();
    for ( auto& c : ext )
        c = ( char )tolower( c );


    if ( ext == u8".obj" )
    {
        auto res = MeshLoad::fromSceneObjFile( filename, false, callback );
        if ( res.has_value() )
        {
            std::vector<std::shared_ptr<Object>> objects( res.value().size() );
            auto& resValue = *res;
            for ( int i = 0; i < objects.size(); ++i )
            {
                std::shared_ptr<ObjectMesh> objectMesh = std::make_shared<ObjectMesh>();
                if ( resValue[i].name.empty() )
                    objectMesh->setName( utf8string( filename.stem() ) );
                else
                    objectMesh->setName( std::move( resValue[i].name ) );
                objectMesh->select( true );
                objectMesh->setMesh( std::make_shared<Mesh>( std::move( resValue[i].mesh ) ) );
                objects[i] = std::dynamic_pointer_cast< Object >( objectMesh );
            }
            result = objects;
        }
        else
            result = tl::make_unexpected( res.error() );
    }
    else if ( !SceneFileFilters.empty() && filename.extension() == SceneFileFilters.front().extension.substr( 1 ) )
    {
        auto res = deserializeObjectTree( filename, {}, callback );
        if ( res.has_value() )
        {
            result = std::vector( { *res } );
            ( *result )[0]->setName( utf8string( filename.stem() ) );
        }
        else
            result = tl::make_unexpected( res.error() );
    }
    else
    {
        auto objectMesh = makeObjectMeshFromFile( filename, callback );
        if ( objectMesh.has_value() )
        {
            objectMesh->select( true );
            auto obj = std::make_shared<ObjectMesh>( std::move( *objectMesh ) );
            result = { obj };
        }
        else if ( objectMesh.error() == "Loading canceled" )
        {
            result = tl::make_unexpected( objectMesh.error() );
        }
        else
        {
            result = tl::make_unexpected( objectMesh.error() );

            auto objectPoints = makeObjectPointsFromFile( filename, callback );
            if ( objectPoints.has_value() )
            {
                objectPoints->select( true );
                auto obj = std::make_shared<ObjectPoints>( std::move( objectPoints.value() ) );
                result = { obj };
            }
            else if ( result.error() == "unsupported file extension" )
            {
                result = tl::make_unexpected( objectPoints.error() );

                auto objectLines = makeObjectLinesFromFile( filename, callback );
                if ( objectLines.has_value() )
                {
                    objectLines->select( true );
                    auto obj = std::make_shared<ObjectLines>( std::move( objectLines.value() ) );
                    result = { obj };
                }
                else if ( result.error() == "unsupported file extension" )
                {
                    result = tl::make_unexpected( objectLines.error() );
                }
            }
        }
    }

    if ( !result.has_value() )
        spdlog::error( result.error() );

    return result;
}

#ifdef __EMSCRIPTEN__

extern "C" {

EMSCRIPTEN_KEEPALIVE int load_files( int count, const char** filenames )
{
    using namespace MR;
    std::vector<std::filesystem::path> paths( count );
    for ( int i = 0; i < count; ++i )
    {
        paths[i] = MR::pathFromUtf8( filenames[i] );
    }

    if ( !paths.empty() )
    {
        SCOPED_HISTORY( "Open files" );
        for ( const auto& filename : paths )
        {
            if ( !filename.empty() )
                Viewer::instanceRef().load_file( filename );
        }

        if ( paths.size() == 1 && !SceneFileFilters.empty() && paths[0].extension() == SceneFileFilters.front().extension.substr( 1 ) )
            SceneRoot::setScenePath( paths[0] );
        else
            SceneRoot::setScenePath( "" );

        Viewer::instanceRef().makeTitleFromSceneRootPath();

        Viewer::instanceRef().viewport().preciseFitDataToScreenBorder( 0.9f );
    }
    return 1;
}

EMSCRIPTEN_KEEPALIVE int save_file( const char* filename )
{
    using namespace MR;
    std::filesystem::path savePath = std::string( filename );
    if ( !Viewer::instanceRef().save_mesh_to_file( savePath ) )
        return 0;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( save_file( UTF8ToString( $0 ) ), filename );
#pragma clang diagnostic pop
    return 1;
}

EMSCRIPTEN_KEEPALIVE int save_scene( const char* filename )
{
    using namespace MR;
    std::filesystem::path savePath = std::string( filename );
    auto res = serializeObjectTree( SceneRoot::get(), savePath );
    if ( !res.has_value() )
        spdlog::error( res.error() );
    else
        getViewerInstance().recentFilesStore.storeFile( savePath );

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( save_file( UTF8ToString( $0 ) ), filename );
#pragma clang diagnostic pop
    return 1;
}

}
#endif

}
