#include "MRSave.h"
#include "MRMeshViewer.h"
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
#include "MRMenu.h"

namespace MR
{

bool Save::save_( const std::filesystem::path & filename )
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selected.empty() )
        return false;

    std::string error;
#ifndef __EMSCRIPTEN__
    if ( auto objVoxels = selected.front()->asType<ObjectVoxels>() )
    {
        auto ext = filename.extension().u8string();
        for ( auto& c : ext )
            c = ( char )tolower( c );

        if ( ext == u8".raw" )
        {
            auto res = VoxelsSave::saveRAW( filename, *objVoxels );
            if ( !res.has_value() )
                error = res.error();
        }
    }
#endif

    if ( error.empty() )
    {
        if ( auto objPoints = selected.front()->asType<ObjectPoints>() )
        {
            if ( objPoints->pointCloud() )
            {
                const auto& colors = objPoints->getVertsColorMap();
                auto res = PointsSave::toAnySupportedFormat( *objPoints->pointCloud(), filename,
                                                             colors.empty() ? nullptr : &colors );
                if ( !res.has_value() )
                    error = res.error();
            }
            else
                error = "ObjectPoints has no PointCloud in it";
        }
    }

    if ( error.empty() )
    {
        if ( auto objLines = selected.front()->asType<ObjectLines>() )
        {
            if ( objLines->polyline() )
            {
                auto res = LinesSave::toAnySupportedFormat( *objLines->polyline(), filename );
                if ( !res.has_value() )
                    error = res.error();
            }
            else
                error = "ObjectLines has no Polyline in it";
        }
    }

    if ( error.empty() )
    {
        if ( auto objMesh = selected.front()->asType<ObjectMesh>() )
        {
            std::shared_ptr<const Mesh> out = objMesh->mesh();
            if ( selected.size() > 1 )
            {
                auto temp = std::make_shared<Mesh>();
                for ( const auto& data : selected )
                {
                    auto thisMesh = data->asType<ObjectMesh>();
                    if ( thisMesh && thisMesh->mesh() )
                        temp->addPart( *thisMesh->mesh() );
                }
                out = std::move( temp );
            }
            if ( !out || out->points.empty() )
                return false;

            const Vector<Color, VertId>* colors{ nullptr };
            if ( selected.size() == 1 && selected[0]->getColoringType() == ColoringType::VertsColorMap )
                colors = &selected[0]->getVertsColorMap();

            auto res = MeshSave::toAnySupportedFormat( *out, filename, colors );
            if ( !res.has_value() )
                error = res.error();
        }
    }

    if ( error.empty() )
    {
        viewer->recentFilesStore.storeFile( filename );
        return true;
    }

    if ( auto menu = viewer->getMenuPlugin() )
        menu->showErrorModal( error );

    spdlog::error( error );
    return false;
}

void Save::init( Viewer* _viewer )
{
    viewer = _viewer;
    connect( _viewer );
}

void Save::shutdown()
{
    disconnect();
}

#ifdef __EMSCRIPTEN__
MRVIEWER_PLUGIN_REGISTRATION(Save)
#endif

} //namespace MR
