#include "MRRibbonSceneObjectsListDrawer.h"
#include "MRRibbonIcons.h"
#include "MRMesh/MRObject.h"
#include "ImGuiMenu.h"
#include "MRRibbonMenu.h"
#include "MRRibbonFontManager.h"
// Object Types
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRSymbolMesh/MRObjectLabel.h"
#include "MRMesh/MRSphereObject.h"
#include "MRMesh/MRPointObject.h"
#include "MRMesh/MRPlaneObject.h"
#include "MRMesh/MRLineObject.h"
#include "MRMesh/MRCylinderObject.h"
#include "MRMesh/MRConeObject.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRVisualObject.h"
// end object types
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "MRSceneCache.h"
#include "imgui.h"
#include "imgui_internal.h"

#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#endif
#include "MRUIStyle.h"
#include "MRColorTheme.h"

namespace MR
{

void RibbonSceneObjectsListDrawer::drawCustomObjectPrefixInScene_( const Object& obj )
{
    if ( !ribbonMenu_ )
        return;

    const auto& fontManager = ribbonMenu_->getFontManager();

    auto imageSize = ImGui::GetFrameHeight();
    auto* imageIcon = RibbonIcons::findByName( obj.typeName(), imageSize,
                                               RibbonIcons::ColorType::White,
                                               RibbonIcons::IconType::ObjectTypeIcon );

    if ( !imageIcon )
    {
        auto font = fontManager.getFontByType( RibbonFontManager::FontType::Icons );
        font->Scale = fontManager.getFontSizeByType( RibbonFontManager::FontType::Default ) /
            fontManager.getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( font );

        ImGui::Text( "%s", getSceneItemIconByTypeName_( obj.typeName() ) );

        ImGui::PopFont();
        font->Scale = 1.0f;
    }
    else
    {
        auto multColor = ImGui::GetStyleColorVec4( ImGuiCol_Text );
        ImGui::Image( *imageIcon, ImVec2( imageSize, imageSize ), multColor );
    }
    ImGui::SameLine();
}

void RibbonSceneObjectsListDrawer::drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& selected, const std::string& uniqueStr )
{
    if ( !ribbonMenu_ )
        return;

    if ( ImGui::BeginPopupContextItem( ( "##SceneContext" + uniqueStr ).c_str() ) )
    {
        auto selectedMask = ribbonMenu_->calcSelectedTypesMask( selected );
        ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImGui::GetStyle().WindowPadding );
        [[maybe_unused]] bool wasChanged = false, wasAction = false;
        const auto& selectedVisualObjs = SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selected>();
        if ( selectedVisualObjs.empty() )
        {
            wasChanged |= ribbonMenu_->drawGeneralOptions( selected );
            wasAction |= ribbonMenu_->drawRemoveButton( selected );
            wasAction |= ribbonMenu_->drawGroupUngroupButton( selected );
            wasAction |= ribbonMenu_->drawSelectSubtreeButton( selected );
            wasAction |= ribbonMenu_->drawCloneButton( selected );
        }
        else if ( ImGui::BeginTable( "##DrawOptions", 2, ImGuiTableFlags_BordersInnerV ) )
        {
            ImGui::TableNextColumn();
            wasChanged |= ribbonMenu_->drawGeneralOptions( selected );
            wasChanged |= ribbonMenu_->drawDrawOptionsCheckboxes( selectedVisualObjs, selectedMask );
            wasChanged |= ribbonMenu_->drawCustomCheckBox( selected, selectedMask );
            wasChanged |= ribbonMenu_->drawAdvancedOptions( selectedVisualObjs, selectedMask );
            ImGui::TableNextColumn();
            wasChanged |= ribbonMenu_->drawDrawOptionsColors( selectedVisualObjs );
            wasAction |= ribbonMenu_->drawRemoveButton( selected );
            wasAction |= ribbonMenu_->drawGroupUngroupButton( selected );
            wasAction |= ribbonMenu_->drawSelectSubtreeButton( selected );
            wasAction |= ribbonMenu_->drawCloneButton( selected );
            wasAction |= ribbonMenu_->drawCloneSelectionButton( selected );
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();

        const bool needCloseCurrentPopup =
            ( ImGui::IsMouseDown( 2 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) ) ||
            ( wasAction || ( wasChanged && closeContextOnChange_ ) );
        if ( needCloseCurrentPopup )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

bool RibbonSceneObjectsListDrawer::collapsingHeader_( const std::string& uniqueName, ImGuiTreeNodeFlags flags )
{
    return RibbonButtonDrawer::CustomCollapsingHeader( uniqueName.c_str(), flags );
}

bool RibbonSceneObjectsListDrawer::drawObject_( Object& object, const std::string& uniqueStr )
{
    bool isOpen = false;
    const std::string uniqueId = object.name() + "##" + uniqueStr;
    const bool hasRealChildren = objectHasSelectableChildren( object );
    const int iconItemCount = 1 + hasRealChildren;
    ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImVec2() );
    if ( ImGui::BeginTable( uniqueId.c_str(), 1 + iconItemCount ) )
    {
        auto& style = ImGui::GetStyle();
        const float width = ImGui::GetContentRegionAvail().x;
        const float widthIconItem = ImGui::GetTextLineHeight() + style.FramePadding.y * 2;
        const float widthMainItem = std::max( width - ( widthIconItem ) * iconItemCount, widthIconItem * 2.f );
        if ( hasRealChildren )
            ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, widthIconItem );
        ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, widthMainItem );
        ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, widthIconItem );

        if ( hasRealChildren )
        {
            ImGui::TableNextColumn();
            isOpen = drawCollapsingArrow_( uniqueId );
        }

        ImGui::TableNextColumn();
        drawObjectIconAndName_( object, uniqueId );

        ImGui::TableNextColumn();
        drawVisibleButton_( object, uniqueId );

        ImGui::EndTable();
    }
    ImGui::PopStyleVar();

    return isOpen;
}

const char* RibbonSceneObjectsListDrawer::getSceneItemIconByTypeName_( const std::string& typeName ) const
{
    if ( typeName == ObjectMesh::TypeName() )
        return "\xef\x82\xac";
#ifndef MRVIEWER_NO_VOXELS
    if ( typeName == ObjectVoxels::TypeName() )
        return "\xef\x86\xb3";
#endif
    if ( typeName == ObjectPoints::TypeName() )
        return "\xef\x84\x90";
    if ( typeName == ObjectLines::TypeName() )
        return "\xef\x87\xa0";
    if ( typeName == ObjectDistanceMap::TypeName() )
        return "\xef\xa1\x8c";
    if ( typeName == ObjectLabel::TypeName() )
        return "\xef\x81\xb5";
    if ( ( typeName == SphereObject::TypeName() ) ||
        ( typeName == PointObject::TypeName() ) ||
        ( typeName == PlaneObject::TypeName() ) ||
        ( typeName == LineObject::TypeName() ) ||
        ( typeName == CylinderObject::TypeName() ) ||
        ( typeName == ConeObject::TypeName() )
        )
        return "\xef\x98\x9f";
    return "\xef\x88\xad";
}

bool RibbonSceneObjectsListDrawer::drawCollapsingArrow_( const std::string& uniqueId )
{
    ImGui::PushStyleColor( ImGuiCol_Header, 0 );
    ImGui::PushStyleColor( ImGuiCol_HeaderHovered, 0 );
    ImGui::PushStyleColor( ImGuiCol_HeaderActive, 0 );
    ImGui::PushStyleColor( ImGuiCol_Border, 0 );
    const bool isOpen = RibbonButtonDrawer::CustomCollapsingHeader( ( "##" + uniqueId ).c_str(), 0, -1 );
    ImGui::PopStyleColor( 4 );
    return isOpen;
}

void RibbonSceneObjectsListDrawer::drawObjectIconAndName_( Object& object, const std::string& uniqueId )
{
    const bool isSelected = object.isSelected();
    if ( isSelected )
        ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectFrame ).getUInt32() );
    else
        ImGui::PushStyleColor( ImGuiCol_Button, 0 );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_HeaderHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_HeaderActive ) );
    ImGui::PushStyleColor( ImGuiCol_Border, 0 );
    const float size = ImGui::GetTextLineHeight() + ImGui::GetStyle().FramePadding.y * 2.f;
    const ImVec2 posBegin = ImGui::GetCursorPos();
    ImGui::Button( ( "##header" + uniqueId ).c_str(), ImVec2{ -1, size } );
    ImGui::PopStyleColor( 4 );
    
    auto& style = ImGui::GetStyle();
    if ( isSelected )
        ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectText ).getUInt32() );
    ImGui::SetCursorPos( posBegin );
    drawCustomObjectPrefixInScene_( object );
    ImGui::SetCursorPos( ImVec2( ImGui::GetCursorPosX(), ImGui::GetCursorPosY() + style.FramePadding.y ) );
    ImGui::Text( "%s", object.name().c_str() );
    if ( isSelected )
        ImGui::PopStyleColor();

    makeDragDropSource_( SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>() );
    makeDragDropTarget_( object, false, false, "0" );
    if ( ImGui::IsItemHovered() )
        updateObjectByClick_( object );
}

void RibbonSceneObjectsListDrawer::drawVisibleButton_( Object& object, const std::string& uniqueId )
{
    ImGui::PushStyleColor( ImGuiCol_Button, 0 );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, 0 );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, 0 );
    ImGui::PushStyleColor( ImGuiCol_Border, 0 );
    const float size = ImGui::GetTextLineHeight() + ImGui::GetStyle().FramePadding.y * 2.f;
    const bool isVisible = object.isVisible();
    const std::string iconName = isVisible ? "object_visible" : "object_invisible";
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2() );
    if ( UI::buttonIconEx( iconName, { 32, 32 }, ( "##visible" + uniqueId ).c_str(), ImVec2( size, size ), { .flatBackgroundColor = true } ) )
    {
        object.setVisible( !isVisible );
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor( 4 );
}

}
