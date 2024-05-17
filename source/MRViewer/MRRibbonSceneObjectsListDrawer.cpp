#include "MRRibbonSceneObjectsListDrawer.h"
#include "MRRibbonIcons.h"
#include "MRMesh/MRObject.h"
#include "ImGuiMenu.h"
#include "MRRibbonMenu.h"
#include "MRRibbonFontManager.h"
// Object Types
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRMesh/MRObjectLabel.h"
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
#include "imgui.h"

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

void RibbonSceneObjectsListDrawer::drawSceneContextMenu_( const std::vector<std::shared_ptr<Object>>& selected )
{
    if ( !ribbonMenu_ )
        return;

    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( ImGui::BeginPopupContextItem( "SceneObjectsListSelectedContext" ) )
    {
        auto selectedMask = ribbonMenu_->calcSelectedTypesMask( selected );
        ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, ImGui::GetStyle().WindowPadding );
        [[maybe_unused]] bool wasChanged = false, wasAction = false;
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

const char* RibbonSceneObjectsListDrawer::getSceneItemIconByTypeName_( const std::string& typeName ) const
{
    if ( typeName == ObjectMesh::TypeName() )
        return "\xef\x82\xac";
#ifndef MRMESH_NO_OPENVDB
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

}
