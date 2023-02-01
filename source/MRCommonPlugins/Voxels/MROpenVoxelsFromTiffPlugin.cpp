#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_TIFF) && !defined(MRMESH_NO_VOXEL)

#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRFileDialog.h"
#include "MRViewer/MRProgressBar.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRVoxelsLoad.h"
#include "MRMesh/MRStringConvert.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRChangeSceneAction.h"

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

    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );

    ImGui::DragFloatValid3( "Voxel Size", &voxelSize_.x, 1e-3f, 1e-3f, 1000 );

    RibbonButtonDrawer::GradientCheckbox( "Invert Surface Orientation", &invertSurfaceOrientation_ );
    ImGui::SetTooltipIfHovered( "By default result voxels has iso-surfaces oriented from bigger value to smaller which represents dense volume," 
                                "invert to have iso-surface oriented from smaller value to bigger to represent distances volume", menuScaling );
    if ( RibbonButtonDrawer::GradientButton( "Open Directory", { -1, 0 } ) )
    {
        auto directory = openFolderDialog();
        if ( directory.empty() )
        {
            ImGui::PopStyleVar( 2 );
            ImGui::EndCustomStatePlugin();
            return;
        }

        ProgressBar::orderWithMainThreadPostProcessing( "Open directory", [this, directory, viewer = Viewer::instance()]()->std::function<void()>
        {
            ProgressBar::nextTask( "Load TIFF Folder" );
            
            auto loadRes = VoxelsLoad::loadTiffDir
            ( {
                directory,
                voxelSize_,
                invertSurfaceOrientation_ ? VoxelsLoad::GridType::LevelSet : VoxelsLoad::GridType::DenseGrid,
                ProgressBar::callBackSetProgress
            } );

            const auto returnError = [directory, loadRes]() -> void
            {
                auto menu = getViewerInstance().getMenuPlugin();
                if ( !menu )
                    return;

                if ( ProgressBar::isCanceled() )
                {
                    menu->showErrorModal( getCancelMessage( directory ) );
                    return;
                }

                menu->showErrorModal( loadRes.error() );
            };            

            if ( ProgressBar::isCanceled() || !loadRes.has_value() )
                return returnError;

            std::shared_ptr<ObjectVoxels> voxelsObject = std::make_shared<ObjectVoxels>();
            voxelsObject->setName( "Loaded Voxels" );
            ProgressBar::setTaskCount( 2 );
            ProgressBar::nextTask( "Construct ObjectVoxels" );
            voxelsObject->construct( *loadRes, ProgressBar::callBackSetProgress );
            if ( ProgressBar::isCanceled() || !loadRes.has_value() )
                return returnError;

            const auto& bins = voxelsObject->histogram().getBins();
            auto minMax = voxelsObject->histogram().getBinMinMax( bins.size() / 3 );

            ProgressBar::nextTask( "Create ISO surface" );
            if ( ProgressBar::isCanceled() || !voxelsObject->setIsoValue( minMax.first, ProgressBar::callBackSetProgress ).has_value() )
                return returnError;
                
            voxelsObject->select( true );
            return [viewer, voxelsObject] ()
            {
                AppendHistory<ChangeSceneAction>( "Open Voxels", voxelsObject, ChangeSceneAction::Type::AddObject );
                SceneRoot::get().addChild( voxelsObject );
                viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
            };
        }, 2 );
    }

    ImGui::PopStyleVar( 2 );
    ImGui::EndCustomStatePlugin();
}

MR_REGISTER_RIBBON_ITEM( OpenVoxelsFromTiffPlugin )

}

#endif