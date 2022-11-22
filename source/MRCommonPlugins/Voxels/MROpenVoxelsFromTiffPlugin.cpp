#ifndef __EMSCRIPTEN__

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
    VoxelsLoad::GridType gridType_ = VoxelsLoad::GridType::DenseGrid;

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
    gridType_ = VoxelsLoad::GridType::DenseGrid;
    return true;
}

void OpenVoxelsFromTiffPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    const float menuWidth = 280.0f * menuScaling;

    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth, .menuScaling = menuScaling } ) )
        return;

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );

    ImGui::DragFloatValid3( "Voxel Size", &voxelSize_.x, 1e-3f, 0.0f );

    RibbonButtonDrawer::GradientRadioButton( "Dense Grid", ( int* )( &gridType_ ), 0 );
    ImGui::SameLine();
    RibbonButtonDrawer::GradientRadioButton( "Level Set", ( int* )( &gridType_ ), 1 );

    if ( RibbonButtonDrawer::GradientButton( "Open Directory" ) )
    {
        auto directory = openFolderDialog();
        if ( directory.empty() )
        {
            ImGui::PopStyleVar( 2 );
            ImGui::EndCustomStatePlugin();
        }

        ProgressBar::orderWithMainThreadPostProcessing( "Open directory", [this, directory, viewer = Viewer::instance()]()->std::function<void()>
        {
            ProgressBar::nextTask( "Load TIFF Folder" );
            auto loadRes = VoxelsLoad::loadTiffDir( directory, gridType_, voxelSize_, ProgressBar::callBackSetProgress );
            if ( loadRes.has_value() && !ProgressBar::isCanceled() )
            {
                std::shared_ptr<ObjectVoxels> voxelsObject = std::make_shared<ObjectVoxels>();
                voxelsObject->setName( "Loaded Voxels" );
                ProgressBar::setTaskCount( 2 );
                ProgressBar::nextTask( "Construct ObjectVoxels" );
                voxelsObject->construct( *loadRes, ProgressBar::callBackSetProgress );
                auto bins = voxelsObject->histogram().getBins();
                auto minMax = voxelsObject->histogram().getBinMinMax( bins.size() / 3 );

                ProgressBar::nextTask( "Create ISO surface" );
                voxelsObject->setIsoValue( minMax.first, ProgressBar::callBackSetProgress );
                voxelsObject->select( true );
                return [viewer, voxelsObject] ()
                {
                    AppendHistory<ChangeSceneAction>( "Open Voxels", voxelsObject, ChangeSceneAction::Type::AddObject );
                    SceneRoot::get().addChild( voxelsObject );
                    viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
                };
            }
            return[viewer, error = loadRes.error()]()
            {
                auto menu = viewer->getMenuPlugin();
                if ( menu )
                    menu->showErrorModal( error );
            };
        }, 2 );
    }

    ImGui::PopStyleVar( 2 );
    ImGui::EndCustomStatePlugin();
}

MR_REGISTER_RIBBON_ITEM( OpenVoxelsFromTiffPlugin )

}

#endif