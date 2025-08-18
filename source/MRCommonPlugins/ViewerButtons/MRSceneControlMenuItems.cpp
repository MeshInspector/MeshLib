#include "MRSceneControlMenuItems.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRHistoryStore.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRViewer/MRSwapRootAction.h"
#include "MRViewer/MRRibbonRegisterItem.h"
#include "MRViewer/ImGuiMenu.h"
#include "MRViewer/MRRibbonFontManager.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectSave.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRFileDialog.h"
#include "MRMesh/MRSerializer.h"
#include "MRViewer/MRProgressBar.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRSceneCache.h"
#include "MRViewer/MRUISaveChangesPopup.h"
#include "MRViewer/MRViewportGlobalBasis.h"
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
        "Quad Viewports",
        "Hex Viewports"
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

    UI::SaveChangesPopupSettings settings;
    settings.scaling = scaling;
    settings.header = "New Scene";
    settings.shortCloseText = "New";
    settings.saveTooltip = "Save current scene and then remove all objects";
    settings.dontSaveTooltip = "Remove all objects without saving and ability to restore them";
    settings.cancelTooltip = "Do not remove any objects, return back";
    settings.onOk =  [this] () { resetScene_(); };
    UI::saveChangesPopup( "New scene##new scene", settings );
}

void ResetSceneMenuItem::resetScene_()
{
    auto rootClone = SceneRoot::get().cloneRoot();
    std::swap( rootClone, SceneRoot::getSharedPtr() );
    getViewerInstance().setSceneDirty();
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
    Viewer::instanceRef().viewport().preciseFitDataToScreenBorder( { 0.9f, false, FitMode::Visible } );
    return false;
}

std::string FitDataMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    auto allObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( const auto& obj : allObjs )
        if ( obj->globalVisibility() )
            return "";
    if ( getViewerInstance().globalBasis && getViewerInstance().globalBasis->isVisible() )
        return "";
    return "There are no visible objects.";
}

FitSelectedObjectsMenuItem::FitSelectedObjectsMenuItem() :
    RibbonMenuItem( "Fit selected objects" )
{
}

bool FitSelectedObjectsMenuItem::action()
{
    Viewer::instanceRef().viewport().preciseFitDataToScreenBorder( { 0.9f, false, FitMode::SelectedObjects } );
    return false;
}

std::string FitSelectedObjectsMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    auto allObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    for ( const auto& obj : allObjs )
        if ( obj->globalVisibility() )
            return "";

    return "There are no visible selected objects.";
}

FitSelectedPrimitivesMenuItem::FitSelectedPrimitivesMenuItem() :
    RibbonMenuItem( "Fit selected primitives" )
{
}

bool FitSelectedPrimitivesMenuItem::action()
{
    Viewer::instanceRef().viewport().preciseFitDataToScreenBorder( { 0.9f, false, FitMode::SelectedPrimitives } );
    return false;
}

std::string FitSelectedPrimitivesMenuItem::isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const
{
    auto allObjs = getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( const auto& obj : allObjs )
        if ( obj->globalVisibility() && obj->mesh() && ( obj->getSelectedEdges().any() || obj->getSelectedFaces().any() ) )
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

    static const Quaternionf quats[(int)Type::Isometric] =
    {
        Quaternionf( Vector3f::plusX(),  -PI2_F ),        // Front
        Quaternionf(),                                    // Top
        Quaternionf(), // unused
        Quaternionf( Vector3f::plusY(),  PI_F ),         // Bottom
        Quaternionf( Vector3f(-1, 1, 1 ), 2 * PI_F / 3 ), // Left
        Quaternionf( Vector3f( 0, 1, 1 ), PI_F ),         // Back
        Quaternionf( Vector3f(-1,-1,-1 ), 2 * PI_F / 3 )  // Right
    };

    if ( type_ < Type::Isometric )
        viewport.setCameraTrackballAngle( quats[int( type_ )] );
    else
        viewport.cameraLookAlong( Vector3f( -1.f, -1.f, -1.f ), Vector3f( -1, -1, 2 ) );

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
using SetButtomViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Bottom>;
using SetLeftViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Left>;
using SetBackViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Back>;
using SetRightViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Right>;
using SetIsometricViewMenuItem = SetViewPresetMenuItemTemplate<SetViewPresetMenuItem::Type::Isometric>;

SetViewportConfigPresetMenuItem::SetViewportConfigPresetMenuItem( Type type ):
    RibbonMenuItem( sGetViewportConfigName( type ) ),
    type_{ type }
{
    updateViewports_ = [] ( const ViewportMask appendedViewports, ViewportId oldActiveViewport )
    {
        auto allObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );

        for ( ViewportId newVpId : appendedViewports )
        {
            for ( auto& obj : allObjs )
            {
                auto masks = obj->getAllVisualizeProperties();
                for ( auto& mask : masks )
                    mask.set( newVpId, mask.contains( oldActiveViewport ) );

                obj->setAllVisualizeProperties( masks );
            }
        }
    };
}

bool SetViewportConfigPresetMenuItem::action()
{
    auto& viewer = getViewerInstance();
    auto bounds = viewer.getViewportsBounds();

    float width = MR::width( bounds );
    float height = MR::height( bounds );

    ViewportMask newViewportMask;
    for ( const auto& viewport : viewer.viewport_list )
        newViewportMask |= viewport.id;
    newViewportMask = ~newViewportMask;
    const ViewportId oldViewportId = viewer.viewport().id;
    for ( int i = int( viewer.viewport_list.size() ) - 1; i > 0; --i )
        viewer.erase_viewport( i );

    ViewportRectangle rect;

    switch ( type_ )
    {
        case Type::Vertical:
            rect.min = bounds.min;
            rect.max.x = std::ceil( bounds.min.x + width * 0.5f );
            rect.max.y = bounds.max.y;
            viewer.viewport().setViewportRect( rect );

            rect.min.x = rect.max.x;
            rect.min.y = bounds.min.y;
            rect.max = bounds.max;
            newViewportMask &= ViewportMask( viewer.append_viewport( rect ) );
            updateViewports_( newViewportMask, oldViewportId );

            break;
        case Type::Horizontal:
            rect.min = bounds.min;
            rect.max.x = bounds.max.x;
            rect.max.y = std::ceil( rect.min.y + height * 0.5f );
            viewer.viewport().setViewportRect( rect );

            rect.min.x = bounds.min.x;
            rect.min.y = rect.max.y;
            rect.max = bounds.max;
            newViewportMask &= ViewportMask( viewer.append_viewport( rect ) );
            updateViewports_( newViewportMask, oldViewportId );

            break;
        case Type::Quad:
        {
            rect.min = bounds.min;
            rect.max.x = std::ceil( bounds.min.x + width * 0.5f );
            rect.max.y = std::ceil( bounds.min.y + height * 0.5f );
            viewer.viewport().setViewportRect( rect );

            rect.min.y = rect.max.y;
            rect.max.x = rect.max.x;
            rect.max.y = bounds.max.y;
            ViewportMask appendedViewports = viewer.append_viewport( rect );

            rect.min.x = rect.max.x;
            rect.min.y = bounds.min.y;
            rect.max.x = bounds.max.x;
            rect.max.y = std::ceil( bounds.min.y + height * 0.5f );
            appendedViewports |= viewer.append_viewport( rect );

            rect.min.y = rect.max.y;
            rect.max = bounds.max;
            appendedViewports |= viewer.append_viewport( rect );
            newViewportMask &= appendedViewports;
            updateViewports_( newViewportMask, oldViewportId );
            break;
        }
        case Type::Hex:
        {
            rect.min = bounds.min;
            rect.max.x = std::ceil( bounds.min.x + width * 0.333f );
            rect.max.y = std::ceil( bounds.min.y + height * 0.5f );
            viewer.viewport().setViewportRect( rect );

            rect.min.y = rect.max.y;
            rect.max.x = rect.max.x;
            rect.max.y = bounds.max.y;
            ViewportMask appendedViewports = viewer.append_viewport( rect );

            rect.min.x = rect.max.x;
            rect.min.y = bounds.min.y;
            rect.max.x = std::ceil( bounds.min.x + width * 0.666f );
            rect.max.y = std::ceil( bounds.min.y + height * 0.5f );
            appendedViewports |= viewer.append_viewport( rect );

            rect.min.y = rect.max.y;
            rect.max.y = bounds.max.y;
            appendedViewports |= viewer.append_viewport( rect );

            rect.min.x = rect.max.x;
            rect.min.y = bounds.min.y;
            rect.max.x = bounds.max.x;
            rect.max.y = std::ceil( bounds.min.y + height * 0.5f );
            appendedViewports |= viewer.append_viewport( rect );

            rect.min.y = rect.max.y;
            rect.max = bounds.max;
            appendedViewports |= viewer.append_viewport( rect );
            newViewportMask &= appendedViewports;
            updateViewports_( newViewportMask, oldViewportId );
            break;
        }
        case Type::Single:
        default:
            rect.min.x = bounds.min.x;
            rect.min.y = bounds.min.y;
            rect.max.x = rect.min.x + width;
            rect.max.y = rect.min.y + height;
            viewer.viewport().setViewportRect( rect );
            updateViewports_( {}, {} );
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
using SetHexViewport = SetViewportConfigPresetMenuItemTemplate<SetViewportConfigPresetMenuItem::Type::Hex>;

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

MR_REGISTER_RIBBON_ITEM( SetHexViewport )

}
