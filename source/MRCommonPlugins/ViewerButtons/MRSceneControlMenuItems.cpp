#include "MRSceneControlMenuItems.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRHistoryStore.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/MRSwapRootAction.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRFileDialog.h"
#include "MRMesh/MRSerializer.h"
#include "MRViewer/MRProgressBar.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRRibbonConstants.h"
#include <array>

namespace
{
using namespace MR;
constexpr const char* sGetViewTypeName( SetViewPresetMenuItem::Type type )
{
    constexpr std::array<const char*, size_t( SetViewPresetMenuItem::Type::Count )> names =
    {
        "Front View",
        "Top View",
        "",// 2 is undefined
        "Bottom View",
        "Left View",
        "Back View",
        "Right View",
        "Isometric View"
    };
    return names[int( type )];
}

constexpr const char* sGetViewportConfigName( SetViewportConfigPresetMenuItem::Type type )
{
    constexpr std::array<const char*, size_t( SetViewportConfigPresetMenuItem::Type::Count )> names =
    {
        "Single Viewport",
        "Horizontal Viewports",
        "Vertical Viewports",
        "Quad Viewports"
    };
    return names[int( type )];
}

}

namespace MR
{

ResetSceneMenuItem::ResetSceneMenuItem() :
    RibbonMenuItem( "New" )
{
    connect( &getViewerInstance() );
}

bool ResetSceneMenuItem::action()
{
    const auto& globalHistory = getViewerInstance().getGlobalHistoryStore();
    if ( globalHistory && globalHistory->isSceneModified() )
        openPopup_ = true;
    else
        resetScene_();
    return false;
}

void ResetSceneMenuItem::preDraw_()
{
    const auto& globalHistory = getViewerInstance().getGlobalHistoryStore();
    if ( !globalHistory )
        return;

    auto menuInstance = getViewerInstance().getMenuPlugin();
    if ( !menuInstance )
        return;
    const auto scaling = menuInstance->menu_scaling();

    if ( openPopup_ )
    {
        ImGui::OpenPopup( popupId_ );
        openPopup_ = false;
    }

    const ImVec2 windowSize{ cModalWindowWidth * scaling, -1 };
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    popupId_ = ImGui::GetID( "New scene##new scene" );

    
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * scaling, 3.0f * cDefaultItemSpacing * scaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * scaling, cModalWindowPaddingY * scaling } );
    if ( ImGui::BeginModalNoAnimation( "New scene##new scene", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            ImGui::PushFont( headerFont );

        const auto headerWidth = ImGui::CalcTextSize( "New Scene" ).x;
        ImGui::SetCursorPosX( ( windowSize.x - headerWidth ) * 0.5f );
        ImGui::Text( "New Scene" );

        if ( headerFont )
            ImGui::PopFont();

        const char* text = "Save your changes?";
        ImGui::SetCursorPosX( ( windowSize.x - ImGui::CalcTextSize( text ).x ) * 0.5f );
        ImGui::Text( "%s", text );

        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * scaling } );

        const float p = ImGui::GetStyle().ItemSpacing.x;
        const ImVec2 btnSize{ ( ImGui::GetContentRegionAvail().x - p * 2 ) / 3.f, 0 };
        if ( RibbonButtonDrawer::GradientButton( "Save", btnSize, ImGuiKey_Enter ) )
        {
            auto savePath = SceneRoot::getScenePath();
            if ( savePath.empty() )
                savePath = saveFileDialog( { {}, {},SceneFileFilters } );

            ImGui::CloseCurrentPopup();
            ProgressBar::orderWithMainThreadPostProcessing( "Saving scene", [this, savePath, &root = SceneRoot::get()]()->std::function<void()>
            {
                auto res = serializeObjectTree( root, savePath, ProgressBar::callBackSetProgress );
                if ( !res.has_value() )
                    spdlog::error( res.error() );

                return[this, savePath, success = res.has_value()]()
                {
                    if ( success )
                        resetScene_();
                };
            } );
        }

        ImGui::SetTooltipIfHovered( "Save current scene and then remove all objects", scaling );
        ImGui::SameLine();
        if ( RibbonButtonDrawer::GradientButtonCommonSize( "Don't Save", btnSize ) )
        {
            ImGui::CloseCurrentPopup();
            resetScene_();
        }
        ImGui::SetTooltipIfHovered( "Remove all objects without saving and ability to restore them", scaling );
        ImGui::SameLine();
        if ( RibbonButtonDrawer::GradientButtonCommonSize( "Cancel", btnSize, ImGuiKey_Escape ) )
            ImGui::CloseCurrentPopup();

        ImGui::SetTooltipIfHovered( "Do not remove any objects, return back", scaling );

        if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
            ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar( 2 );
}

void ResetSceneMenuItem::resetScene_()
{
    auto rootClone = SceneRoot::get().clone();
    std::swap( rootClone, SceneRoot::getSharedPtr() );
    if ( const auto& store = getViewerInstance().getGlobalHistoryStore() )
        store->clear();
    getViewerInstance().onSceneSaved( {} );
}

FitDataMenuItem::FitDataMenuItem() :
    RibbonMenuItem( "Fit data" )
{
}

bool FitDataMenuItem::action()
{
    Viewer::instanceRef().viewport().preciseFitDataToScreenBorder( { 0.9f, false, Viewport::FitMode::Visible } );
    return false;
}

std::string FitDataMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    auto allObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( const auto& obj : allObjs )
        if ( obj->globalVisibilty() )
            return "";

    return "There are no visible objects.";
}

FitSelectedObjectsMenuItem::FitSelectedObjectsMenuItem() :
    RibbonMenuItem( "Fit selected objects" )
{
}

bool FitSelectedObjectsMenuItem::action()
{
    Viewer::instanceRef().viewport().preciseFitDataToScreenBorder( { 0.9f, false, Viewport::FitMode::SelectedObjects } );
    return false;
}

std::string FitSelectedObjectsMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    auto allObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    for ( const auto& obj : allObjs )
        if ( obj->globalVisibilty() )
            return "";

    return "There are no visible selected objects.";
}

FitSelectedPrimitivesMenuItem::FitSelectedPrimitivesMenuItem() :
    RibbonMenuItem( "Fit selected primitives" )
{
}

bool FitSelectedPrimitivesMenuItem::action()
{
    Viewer::instanceRef().viewport().preciseFitDataToScreenBorder( { 0.9f, false, Viewport::FitMode::SelectedPrimitives } );
    return false;
}

std::string FitSelectedPrimitivesMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    auto allObjs = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( const auto& obj : allObjs )
        if ( obj->globalVisibilty() && obj->mesh() && ( obj->getSelectedEdges().any() || obj->getSelectedFaces().any() ) )
            return "";

    return "There are no visible selected primitives.";
}

SetViewPresetMenuItem::SetViewPresetMenuItem( Type type ) :
    RibbonMenuItem( sGetViewTypeName( type ) ),
    type_{type}
{
}

bool SetViewPresetMenuItem::action()
{
    auto& viewport = getViewerInstance().viewport();
    if ( type_ != Type::Isometric )
        viewport.setCameraTrackballAngle( getCanonicalQuaternions<float>()[int( type_ )] );
    else
        viewport.cameraLookAlong( Vector3f( -1.f, -1.f, -1.f ), Vector3f( -1.f, 2.f, -1.f ) );

    viewport.preciseFitDataToScreenBorder( { 0.9f } );
    return false;
}

template<SetViewPresetMenuItem::Type T>
class SetViewPresetMenuItemTemplate : public SetViewPresetMenuItem
{
public:
    SetViewPresetMenuItemTemplate() :
        SetViewPresetMenuItem( T )
    {
    }
};

using SetFrontViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Front>;
using SetTopViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Top>;
using SetButtomViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Buttom>;
using SetLeftViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Left>;
using SetBackViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Back>;
using SetRightViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Right>;
using SetIsometricViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Isometric>;

SetViewportConfigPresetMenuItem::SetViewportConfigPresetMenuItem( Type type ):
    RibbonMenuItem( sGetViewportConfigName( type ) ),
    type_{ type }
{
}

bool SetViewportConfigPresetMenuItem::action()
{
    auto& viewer = getViewerInstance();
    auto bounds = viewer.getViewportsBounds();

    float width = MR::width( bounds );
    float height = MR::height( bounds );

    for ( int i = int( viewer.viewport_list.size() ) - 1; i > 0; --i )
        viewer.erase_viewport( i );

    auto allObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );

    ViewportRectangle rect;
    const ViewportId activeViewportId = viewer.viewport().id;

    const auto updateMasks = [&allObjs, activeViewportId] ( ViewportId newVpId )
    {
        for ( auto& obj : allObjs )
        {
            auto masks = obj->getAllVisualizeProperties();
            for (auto& mask: masks )
                mask.set( newVpId, mask.contains( activeViewportId ) );

            obj->setAllVisualizeProperties( masks );
        }
    };

    switch ( type_ )
    {
        case Type::Vertical:
            rect.min.x = bounds.min.x;
            rect.min.y = bounds.min.y;
            rect.max.x = rect.min.x + width * 0.5f;
            rect.max.y = rect.min.y + height;
            viewer.viewport().setViewportRect( rect );

            rect.min.x = bounds.min.x + width * 0.5f;
            rect.min.y = bounds.min.y;
            rect.max.x = rect.min.x + width * 0.5f;
            rect.max.y = rect.min.y + height;
            updateMasks( viewer.append_viewport( rect ) );

            break;
        case Type::Horizontal:
            rect.min.x = bounds.min.x;
            rect.min.y = bounds.min.y;
            rect.max.x = rect.min.x + width;
            rect.max.y = rect.min.y + height * 0.5f;
            viewer.viewport().setViewportRect( rect );

            rect.min.x = bounds.min.x;
            rect.min.y = bounds.min.y + height * 0.5f;
            rect.max.x = rect.min.x + width;
            rect.max.y = rect.min.y + height * 0.5f;
            updateMasks( viewer.append_viewport( rect ) );

            break;
        case Type::Quad:
            rect.min.x = bounds.min.x;
            rect.min.y = bounds.min.y;
            rect.max.x = rect.min.x + width * 0.5f;
            rect.max.y = rect.min.y + height * 0.5f;
            viewer.viewport().setViewportRect( rect );

            rect.min.x = bounds.min.x;
            rect.min.y = bounds.min.y + height * 0.5f;
            rect.max.x = rect.min.x + width * 0.5f;
            rect.max.y = rect.min.y + height * 0.5f;
            updateMasks( viewer.append_viewport( rect ) );

            rect.min.x = bounds.min.x + width * 0.5f;
            rect.min.y = bounds.min.y;
            rect.max.x = rect.min.x + width * 0.5f;
            rect.max.y = rect.min.y + height * 0.5f;
            updateMasks( viewer.append_viewport( rect ) );

            rect.min.x = bounds.min.x + width * 0.5f;
            rect.min.y = bounds.min.y + height * 0.5f;
            rect.max.x = rect.min.x + width * 0.5f;
            rect.max.y = rect.min.y + height * 0.5f;
            updateMasks( viewer.append_viewport( rect ) );

            break;
        case Type::Single:
        default:
            rect.min.x = bounds.min.x;
            rect.min.y = bounds.min.y;
            rect.max.x = rect.min.x + width;
            rect.max.y = rect.min.y + height;
            viewer.viewport().setViewportRect( rect );
            break;
    }
    return false;
}

template<SetViewportConfigPresetMenuItem::Type T>
class SetViewportConfigPresetMenuItemTemplate : public SetViewportConfigPresetMenuItem
{
public:
    SetViewportConfigPresetMenuItemTemplate() :
        SetViewportConfigPresetMenuItem( T )
    {
    }
};

using SetSingleViewport = SetViewportConfigPresetMenuItemTemplate<SetViewportConfigPresetMenuItem::Type::Single>;
using SetHorizontalViewport = SetViewportConfigPresetMenuItemTemplate<SetViewportConfigPresetMenuItem::Type::Horizontal>;
using SetVerticalViewport = SetViewportConfigPresetMenuItemTemplate<SetViewportConfigPresetMenuItem::Type::Vertical>;
using SetQuadViewport = SetViewportConfigPresetMenuItemTemplate<SetViewportConfigPresetMenuItem::Type::Quad>;

MR_REGISTER_RIBBON_ITEM( ResetSceneMenuItem )

MR_REGISTER_RIBBON_ITEM( FitDataMenuItem )

MR_REGISTER_RIBBON_ITEM( FitSelectedObjectsMenuItem )

MR_REGISTER_RIBBON_ITEM( FitSelectedPrimitivesMenuItem )

MR_REGISTER_RIBBON_ITEM( SetFrontViewMenuItem )

MR_REGISTER_RIBBON_ITEM( SetTopViewMenuItem )

MR_REGISTER_RIBBON_ITEM( SetButtomViewMenuItem )

MR_REGISTER_RIBBON_ITEM( SetLeftViewMenuItem )

MR_REGISTER_RIBBON_ITEM( SetBackViewMenuItem )

MR_REGISTER_RIBBON_ITEM( SetRightViewMenuItem )

MR_REGISTER_RIBBON_ITEM( SetIsometricViewMenuItem )

MR_REGISTER_RIBBON_ITEM( SetSingleViewport )

MR_REGISTER_RIBBON_ITEM( SetHorizontalViewport )

MR_REGISTER_RIBBON_ITEM( SetVerticalViewport )

MR_REGISTER_RIBBON_ITEM( SetQuadViewport )

}