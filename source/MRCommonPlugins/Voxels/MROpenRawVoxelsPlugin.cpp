#ifndef __EMSCRIPTEN__
#include "MROpenRawVoxelsPlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRFileDialog.h"
#include "MRViewer/MRProgressBar.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRStringConvert.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRChangeSceneAction.h"

namespace
{
constexpr std::array<const char*, size_t( MR::VoxelsLoad::RawParameters::ScalarType::Count )> cScalarTypeNames =
{
    "UInt8",
    "Int8",
    "UInt16",
    "Int16",
    "UInt32",
    "Int32",
    "UInt64",
    "Int64",
    "Float32",
    "Float64"
};
}

namespace MR
{

OpenRawVoxelsPlugin::OpenRawVoxelsPlugin():
    StatePlugin( "Open RAW Voxels" )
{
}

void OpenRawVoxelsPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 350.0f * menuScaling;
    if ( !ImGui::BeginCustomStatePlugin( plugin_name.c_str(), &dialogIsOpen_, { .collapsed = &dialogIsCollapsed_, .width = menuWidth, .menuScaling = menuScaling } ) )
        return;
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { cCheckboxPadding * menuScaling, cCheckboxPadding * menuScaling } );
    RibbonButtonDrawer::GradientCheckbox( "Auto parameters", &autoMode_ );
    ImGui::PopStyleVar();
    ImGui::SetTooltipIfHovered( "Use this flag to parse RAW parameters from filename.", menuScaling );
    ImGui::Separator();

    if ( !autoMode_ )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { ImGui::GetStyle().FramePadding.x, cInputPadding * menuScaling } );
        ImGui::PushItemWidth( menuScaling * 200.0f );
        ImGui::DragIntValid3( "Dimensions", &parameters_.dimensions.x, 1, 0, std::numeric_limits<int>::max() );
        ImGui::DragFloatValid3( "Voxel size", &parameters_.voxelSize.x, 1e-3f, 0.0f );
        ImGui::PopItemWidth();
        ImGui::Separator();
        ImGui::PopStyleVar();
        RibbonButtonDrawer::CustomCombo( "Scalar Type", ( int* )&parameters_.scalarType, MenuItemsList( std::begin( cScalarTypeNames ), std::end( cScalarTypeNames ) ) );
    }
    if ( RibbonButtonDrawer::GradientButton( "Open file", ImVec2( -1, 0 ) ) )
    {
        auto path = openFileDialog( { {},{},{{"RAW File","*.raw"}} } );
        if ( !path.empty() )
        {
            ProgressBar::orderWithMainThreadPostProcessing( "Load voxels", [params = parameters_, path, autoMode = autoMode_] ()->std::function<void()>
            {
                ProgressBar::nextTask( "Load file" );
                tl::expected<VdbVolume, std::string> res;
                auto error = std::make_shared<std::string>();

                const auto showError = [error] () -> void
                {
                    auto menu = getViewerInstance().getMenuPlugin();
                    if ( !menu || !error)
                        return;

                    menu->showErrorModal( *error );
                };

                
                if ( autoMode )
                    res = VoxelsLoad::loadRaw( path, ProgressBar::callBackSetProgress );
                else
                    res = VoxelsLoad::loadRaw( path, params, ProgressBar::callBackSetProgress );

                if ( ProgressBar::isCanceled() )
                {
                    *error = getCancelMessage( path );
                    return showError;
                }

                if ( res.has_value() )
                {
                    ProgressBar::nextTask( "Create object" );
                    std::shared_ptr<ObjectVoxels> object = std::make_shared<ObjectVoxels>();
                    object->setName( utf8string( path.stem() ) );
                    object->construct( res->data, res->voxelSize, ProgressBar::callBackSetProgress );
                    auto bins = object->histogram().getBins();
                    auto minMax = object->histogram().getBinMinMax( bins.size() / 3 );

                    if ( ProgressBar::isCanceled() )
                    {
                        *error = getCancelMessage( path );
                        return showError;
                    }

                    ProgressBar::nextTask( "Create ISO surface" );
                    object->setIsoValue( minMax.first, ProgressBar::callBackSetProgress );
                    object->select( true );

                    if ( ProgressBar::isCanceled() )
                    {
                        *error = getCancelMessage( path );
                        return showError;
                    }

                    return [object] ()
                    {
                        AppendHistory<ChangeSceneAction>( "Open Voxels", object, ChangeSceneAction::Type::AddObject );
                        SceneRoot::get().addChild( object );
                        getViewerInstance().viewport().preciseFitDataToScreenBorder( { 0.9f } );
                    };
                }
                else
                {
                    *error = res.error();
                    return showError;
                }
            }, 3 );
            dialogIsOpen_ = false;
        }
    }
    ImGui::PopStyleVar( 2 );
    ImGui::EndCustomStatePlugin();
}

bool OpenRawVoxelsPlugin::onEnable_()
{
    parameters_ = VoxelsLoad::RawParameters();
    autoMode_ = true;
    return true;
}

bool OpenRawVoxelsPlugin::onDisable_()
{
    return true;
}

MR_REGISTER_RIBBON_ITEM( OpenRawVoxelsPlugin )

}
#endif
