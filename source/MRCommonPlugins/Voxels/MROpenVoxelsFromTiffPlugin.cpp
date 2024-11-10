#ifndef MESHLIB_NO_VOXELS
#include <MRVoxels/MRVoxelsFwd.h>
#ifndef MRVOXELS_NO_TIFF

#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRShowModal.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRFileDialog.h"
#include "MRViewer/MRProgressBar.h"
#include "MRViewer/MRViewport.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRVoxels/MRObjectVoxels.h"
#include "MRVoxels/MRVoxelsLoad.h"
#include "MRMesh/MRChangeSceneAction.h"
#include <MRMesh/MRSceneRoot.h>
#include "MRMesh/MRStringConvert.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{
class OpenVoxelsFromTiffPlugin : public StatePlugin
{
    Vector3f voxelSize_;
    bool invertSurfaceOrientation_ = false;
public:
    OpenVoxelsFromTiffPlugin()
        : StatePlugin( "Open Voxels From TIFF" )
    {}

    virtual bool onEnable_() override;

    virtual void drawDialog( float menuScaling, ImGuiContext* ) override;
};

bool OpenVoxelsFromTiffPlugin::onEnable_()
{
    voxelSize_ = Vector3f::diagonal( 1.0f );
    invertSurfaceOrientation_ = false;
    return true;
}

void OpenVoxelsFromTiffPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    const float menuWidth = 280.0f * menuScaling;

    if ( !ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );

    UI::drag<LengthUnit>( "Voxel Size", voxelSize_, 1e-3f, 1e-3f, 1000.f );

    UI::checkbox( "Invert Surface Orientation", &invertSurfaceOrientation_ );
    UI::setTooltipIfHovered( "By default result voxels has iso-surfaces oriented from bigger value to smaller which represents dense volume,"
                                "invert to have iso-surface oriented from smaller value to bigger to represent distances volume", menuScaling );
    if ( UI::button( "Open Directory", { -1, 0 } ) )
    {
        auto directory = openFolderDialog();
        if ( directory.empty() )
        {
            ImGui::PopStyleVar( 2 );
            ImGui::EndCustomStatePlugin();
            return;
        }

        ProgressBar::orderWithMainThreadPostProcessing( "Open Voxels From TIFF", [this, directory, viewer = Viewer::instance()]()->std::function<void()>
        {
            ProgressBar::nextTask( "Load TIFF Folder" );
            spdlog::info( "Loading TIFF Folder: {}", utf8string( directory ) );
            auto loadRes = VoxelsLoad::loadTiffDir
            ( {
                directory,
                voxelSize_,
                invertSurfaceOrientation_ ? VoxelsLoad::GridType::LevelSet : VoxelsLoad::GridType::DenseGrid,
                ProgressBar::callBackSetProgress
            } );

            const auto returnError = [directory, loadRes]() -> void
            {
                showError( ProgressBar::isCanceled() ? getCancelMessage( directory ) : loadRes.error() );
            };

            if ( ProgressBar::isCanceled() || !loadRes.has_value() )
                return returnError;

            std::shared_ptr<ObjectVoxels> voxelsObject = std::make_shared<ObjectVoxels>();
            auto name = utf8string( directory.filename() );
            if ( name.empty() )
                name = "Tiff Voxels";
            voxelsObject->setName( std::move( name ) );
            ProgressBar::setTaskCount( 2 );
            ProgressBar::nextTask( "Construct ObjectVoxels" );
            voxelsObject->construct( *loadRes );
            if ( ProgressBar::isCanceled() || !loadRes.has_value() )
                return returnError;

            const auto& bins = voxelsObject->histogram().getBins();
            auto minMax = voxelsObject->histogram().getBinMinMax( bins.size() / 3 );

            ProgressBar::nextTask( "Create ISO surface" );
            if ( ProgressBar::isCanceled() || !voxelsObject->setIsoValue( minMax.first, ProgressBar::callBackSetProgress ).has_value() )
                return returnError;

            voxelsObject->select( true );
            return [this, viewer, voxelsObject, directory] ()
            {
                AppendHistory<ChangeSceneAction>( "Open Voxels From TIFF", voxelsObject, ChangeSceneAction::Type::AddObject );
                SceneRoot::get().addChild( voxelsObject );
                viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
                std::filesystem::path scenePath = directory;
                scenePath += ".mru";
                getViewerInstance().onSceneSaved( scenePath, false );
                dialogIsOpen_ = false;
            };
        }, 2 );
    }

    ImGui::PopStyleVar( 2 );
    ImGui::EndCustomStatePlugin();
}

MR_REGISTER_RIBBON_ITEM( OpenVoxelsFromTiffPlugin )

}

#endif
#endif

#if defined( __EMSCRIPTEN__ ) && ( defined( MESHLIB_NO_VOXELS ) || defined( MRVOXELS_NO_TIFF ) )
#include "MRCommonPlugins/Basic/MRWasmUnavailablePlugin.h"
MR_REGISTER_WASM_UNAVAILABLE_ITEM( OpenVoxelsFromTiffPlugin, "Open Voxels From TIFF" )
#endif
