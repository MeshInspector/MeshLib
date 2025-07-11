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
#include "MRUIStyle.h"
#include "MRRibbonConstants.h"
#include "MRImGuiVectorOperators.h"
#include "MRColorTheme.h"
#include "MRViewport.h"
#include "MRImGuiImage.h"
#include "imgui_internal.h"

#ifndef MRVIEWER_NO_VOXELS
#include "MRVoxels/MRObjectVoxels.h"
#endif

namespace MR
{

void RibbonSceneObjectsListDrawer::draw( float height, float scaling )
{
    currentElementId_ = 1;
    lastDrawnSibling_.clear();
    SceneObjectsListDrawer::draw( height, scaling );
}

void RibbonSceneObjectsListDrawer::initRibbonMenu( RibbonMenu* ribbonMenu )
{
    // lets assume that will not have depth more than 32 most times
    lastDrawnSibling_.reserve( 32 );
    ribbonMenu_ = ribbonMenu;
}

void RibbonSceneObjectsListDrawer::drawCustomObjectPrefixInScene_( const Object& obj, bool opened )
{
    if ( !ribbonMenu_ )
        return;

    const auto& fontManager = ribbonMenu_->getFontManager();

    auto imageSize = ImGui::GetFrameHeight() - 2 * menuScaling_;
    std::string name = obj.typeName();
    if ( opened && name == Object::TypeName() )
        name += "_open";
    auto* imageIcon = RibbonIcons::findByName( name, imageSize,
                                               RibbonIcons::ColorType::White,
                                               RibbonIcons::IconType::ObjectTypeIcon );

    if ( !imageIcon )
    {
        auto font = fontManager.getFontByType( RibbonFontManager::FontType::Icons );
        font->Scale = fontManager.getFontSizeByType( RibbonFontManager::FontType::Default ) /
            fontManager.getFontSizeByType( RibbonFontManager::FontType::Icons );
        ImGui::PushFont( font );

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + ( imageSize - ImGui::GetFontSize() ) * 0.5f );
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
        onDrawContextSignal();

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
            wasAction |= ribbonMenu_->drawMergeSubtreeButton( selected );
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
            wasAction |= ribbonMenu_->drawMergeSubtreeButton( selected );
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

std::string RibbonSceneObjectsListDrawer::objectLineStrId_( const Object& object, const std::string& uniqueStr )
{
    return "##OpenState_" + object.name() + "_" + uniqueStr;
}

bool RibbonSceneObjectsListDrawer::drawObject_( Object& object, const std::string& uniqueStr, int depth )
{
    const bool hasRealChildren = objectHasSelectableChildren( object );

    auto isOpened = drawTreeOpenedState_( object, !hasRealChildren, uniqueStr, depth );
    ImGui::SameLine();
    drawObjectLine_( object, uniqueStr, isOpened );

    // update last sibling
    if ( lastDrawnSibling_.size() <= depth )
        lastDrawnSibling_.resize( depth + 1 );
    lastDrawnSibling_[depth] = { ImGui::GetCursorScreenPos().y,currentElementId_ };
    ++currentElementId_;

    return isOpened;
}

bool RibbonSceneObjectsListDrawer::drawSkippedObject_( Object& object, const std::string& uniqueStr, int depth )
{
    auto startScreenPos = ImGui::GetCursorScreenPos();
    drawHierarhyLine_( startScreenPos, depth, true );
    return SceneObjectsListDrawer::drawSkippedObject_( object, uniqueStr, depth );
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

bool RibbonSceneObjectsListDrawer::drawTreeOpenedState_( Object& object, bool leaf, const std::string& uniqueStr, int depth )
{
    auto openCommandIt = sceneOpenCommands_.find( &object );
    if ( openCommandIt != sceneOpenCommands_.end() )
        ImGui::SetNextItemOpen( openCommandIt->second );

    ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0, 0, 0, 0 ) );
    if ( leaf )
    {
        ImGui::PushStyleColor( ImGuiCol_HeaderHovered, ImVec4( 0, 0, 0, 0 ) );
        ImGui::PushStyleColor( ImGuiCol_HeaderActive, ImVec4( 0, 0, 0, 0 ) );
    }
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

    const ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_AllowOverlap |
        ( !leaf ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_Bullet );


    auto startScreenPos = ImGui::GetCursorScreenPos();

    const float cFrameHeight = ImGui::GetFrameHeight();
    // window->WorkRect.Max.x hardcoded inside ImGui as limit of width, so manual change it here
    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    float storedWorkRectMaxX = window->WorkRect.Max.x;
    window->WorkRect.Max.x = window->DC.CursorPos.x + cFrameHeight - 2 * menuScaling_;
    const bool isOpen = collapsingHeader_( objectLineStrId_( object, uniqueStr ).c_str(), flags );
    window->WorkRect.Max.x = storedWorkRectMaxX;

    ImGui::PopStyleColor( leaf ? 3 : 1 );
    ImGui::PopStyleVar();

    // draw hierarchy lines
    drawHierarhyLine_( startScreenPos, depth, false );

    return isOpen;
}

void RibbonSceneObjectsListDrawer::drawObjectLine_( Object& object, const std::string& uniqueStr, bool opened )
{
    const bool isSelected = object.isSelected();

    const auto& style = ImGui::GetStyle();
    const float cFrameHeight = ImGui::GetFrameHeight();

    auto context = ImGui::GetCurrentContext();
    auto window = context->CurrentWindow;
    auto drawList = window->DrawList;

    auto startPos = ImGui::GetCursorPos();
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0 );
    ImGui::PushStyleColor( ImGuiCol_Button, isSelected ? ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectFrame ) : ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Background ) );
    if ( !isSelected )
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, Color::gray().scaledAlpha( 0.2f ).getUInt32() );
    UI::ButtonCustomizationParams params;
    params.forceImGuiBackground = true;
    params.flags = ImGuiButtonFlags_AllowOverlap;
    UI::buttonEx( ( "##SelectBtn_" + object.name() + "_" + uniqueStr ).c_str(), Vector2f( -1, cFrameHeight ), params );
    if ( ImGui::IsItemHovered( ImGuiHoveredFlags_AllowWhenBlockedByActiveItem ) && needDragDropTarget_() )
    {
        auto rect = context->LastItemData.Rect;
        drawList->PushClipRect( window->InnerRect.Min, window->InnerRect.Max );
        drawList->AddRect( rect.Min, rect.Max, ImGui::GetColorU32( ImGuiCol_ButtonActive ), style.FrameRounding, 0, 2 * menuScaling_ );
        drawList->PopClipRect();

    }
    ImGui::PopStyleColor( !isSelected ? 2 : 1 );
    ImGui::PopStyleVar();
    
    const auto& selected = SceneCache::getAllObjects<Object, ObjectSelectivityType::Selected>();

    makeDragDropSource_( selected );
    makeDragDropTarget_( object, false, false, uniqueStr );

    context->LastItemData.InFlags |= ImGuiItemFlags_AllowOverlap; // needed so hover check respect overlap

    bool frameHovered = ImGui::IsItemHovered();
    if ( frameHovered )
        processItemClick_( object, selected );

    auto lineObjectData = context->LastItemData;

    // draw text
    if ( isSelected )
        ImGui::PushStyleColor( ImGuiCol_Text, 0xffffffff );
    drawList->PushClipRect( window->InnerClipRect.Min, window->InnerClipRect.Max - ImVec2( cFrameHeight, 0 ) );
    ImGui::SetCursorPos( startPos + ImVec2( style.FramePadding.x, 0 ) );
    drawCustomObjectPrefixInScene_( object, opened );
    ImGui::SetCursorPosY( startPos.y + style.FramePadding.y );
    ImGui::Text( "%s", object.name().c_str() );
    drawList->PopClipRect();

    // draw visibility button
    ImGui::SetCursorPos( ImVec2( window->InnerClipRect.Max.x - window->Pos.x - cFrameHeight - style.FramePadding.x, startPos.y ) );
    drawEyeButton_( object, uniqueStr, frameHovered );
    if ( isSelected )
        ImGui::PopStyleColor();

    // set back last item as if it was main line for further checks
    context->LastItemData = lineObjectData;
}

void RibbonSceneObjectsListDrawer::drawEyeButton_( Object& object, const std::string& uniqueStr, bool frameHovered )
{
    auto& viewer = getViewerInstance();
    auto& vp = viewer.viewport();
    bool isVisible = object.isVisible( vp.id );

    const float cFrameHeight = ImGui::GetFrameHeight();
    const float cImageHeight = 24 * menuScaling_;
    auto* imageIcon = RibbonIcons::findByName( isVisible ? "Ribbon Scene Show all" : "Ribbon Scene Hide all", cFrameHeight, RibbonIcons::ColorType::White, RibbonIcons::IconType::RibbonItemIcon );
    if ( !imageIcon )
    {
        drawObjectVisibilityCheckbox_( object, uniqueStr );
        ImGui::NewLine();
        return;
    }

    auto btnScreenPos = ImGui::GetCursorScreenPos();
    UI::ButtonCustomizationParams params;
    params.forceImGuiBackground = true;
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0 );
    ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4( 0, 0, 0, 0 ) );
    bool changed = UI::buttonEx( ( "##VisibilityBtn_" + object.name() + "_" + uniqueStr ).c_str(), ImVec2( -1, cFrameHeight ), params );
    ImGui::PopStyleColor( 3 );
    ImGui::PopStyleVar();

    bool isHovered = ImGui::IsItemHovered();

    Color imageColor = Color( ImGui::GetStyleColorVec4( ImGuiCol_Text ) );
    if ( !isHovered && !frameHovered )
    {
        if ( isVisible )
        {
            bool globalVisible = object.globalVisibility( vp.id );
            if ( !globalVisible )
                imageColor = imageColor.scaledAlpha( 0.5f );
            else if ( !object.isSelected() )
                imageColor = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectFrame );
        }
        else
            imageColor = imageColor.scaledAlpha( 0.5f );
    }

    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    auto drawList = window->DrawList;

    auto pos = btnScreenPos + ImVec2( 1, 1 ) * ( cFrameHeight - cImageHeight ) * 0.5f;
    drawList->AddImage( imageIcon->getImTextureId(), pos, pos + ImVec2( 1, 1 ) * cImageHeight, ImVec2( 0, 1 ), ImVec2( 1, 0 ), imageColor.getUInt32() );

    if ( changed )
    {
        object.setVisible( !isVisible, vp.id );
        if ( deselectNewHiddenObjects_ && !object.isVisible( viewer.getPresentViewports() ) )
            object.select( false );
    }
}

void RibbonSceneObjectsListDrawer::drawHierarhyLine_( const Vector2f& startScreenPos, int depth, bool skipped )
{
    if ( depth <= 0 )
        return;

    auto numDepths = lastDrawnSibling_.size();

    if ( skipped && numDepths < depth )
        return;

    int numSteps = 0;
    if ( numDepths > depth )
    {
        auto lastParent = lastDrawnSibling_[depth - 1].id;
        if ( lastParent == 0 )
        {
            // parent was skipped, so it is over the screen
            numSteps = -1;
        }
        else if ( lastParent + 1 != currentElementId_ )
        {
            // otherwise it first child
            numSteps = currentElementId_ - lastDrawnSibling_[depth].id;
        }
    }

    const float cFrameHeight = ImGui::GetFrameHeight();
    auto drawList = ImGui::GetWindowDrawList();

    auto pos0 = ImVec2( startScreenPos.x - ImGui::GetStyle().FramePadding.x * 0.75f, startScreenPos.y + cFrameHeight * 0.5f );
    auto pos1 = ImVec2( startScreenPos.x - cFrameHeight * 0.5f, pos0.y );
    auto pos2 = ImVec2( pos1.x, pos1.y );
    if ( numSteps < 0 )
        pos2.y = 0.0f;
    else if ( numSteps == 0 )
        pos2.y = numDepths < depth ? ( pos2.y - cFrameHeight * 0.5f ) : lastDrawnSibling_[depth - 1].screenPosY - cFrameHeight * 0.25f;
    else
        pos2.y = lastDrawnSibling_[depth].screenPosY - cFrameHeight;

    drawList->AddLine( pos0, pos1, Color::gray().getUInt32(), menuScaling_ );
    drawList->AddLine( pos1, pos2, Color::gray().getUInt32(), menuScaling_ );

    if ( skipped )
        lastDrawnSibling_.resize( depth - 1 );

}

}
