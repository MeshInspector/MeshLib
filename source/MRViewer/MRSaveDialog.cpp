#include "MRSaveDialog.h"

#include "MRRibbonConstants.h"
#include "MRUIStyle.h"
#include "ImGuiHelpers.h"
#include "MRRibbonFontManager.h"
#include "MRFileDialog.h"
#include "MRProgressBar.h"
#include "MRShowModal.h"
#include "MRViewer.h"
#include "MRSceneCache.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRVisualObject.h"

namespace MR
{

void saveSceneDialog( float scaling, const std::string& name, const std::string& label, const std::function<void()>& customFunction )
{
    const ImVec2 windowSize{ cModalWindowWidth * scaling, -1 };
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * scaling, 3.0f * cDefaultItemSpacing * scaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * scaling, cModalWindowPaddingY * scaling } );
    if ( ImGui::BeginModalNoAnimation( name.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            ImGui::PushFont( headerFont );

        const auto headerWidth = ImGui::CalcTextSize( label.c_str() ).x;
        ImGui::SetCursorPosX( ( windowSize.x - headerWidth ) * 0.5f );
        ImGui::Text( label.c_str() );

        if ( headerFont )
            ImGui::PopFont();

        // do not suggest saving empty scene
        const bool showSave = !SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selectable>().empty();
        if ( showSave )
        {
            const char* text = "Save your changes?";
            ImGui::SetCursorPosX( ( windowSize.x - ImGui::CalcTextSize( text ).x ) * 0.5f );
            ImGui::Text( "%s", text );
        }

        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * scaling } );

        const float p = ImGui::GetStyle().ItemSpacing.x;
        const Vector2f btnSize{ showSave ? ( ImGui::GetContentRegionAvail().x - p * 2 ) / 3.f : ( ImGui::GetContentRegionAvail().x - p ) / 2.f, 0 };

        if ( showSave )
        {
            if ( UI::button( "Save", btnSize, ImGuiKey_Enter ) )
            {
                auto savePath = SceneRoot::getScenePath();
                if ( savePath.empty() )
                    savePath = saveFileDialog( { .filters = SceneSave::getFilters() } );

                ImGui::CloseCurrentPopup();
                if ( !savePath.empty() )
                    ProgressBar::orderWithMainThreadPostProcessing( "Saving scene", [customFunction_ = customFunction, savePath, &root = SceneRoot::get()] ()->std::function<void()>
                {
                    auto res = ObjectSave::toAnySupportedSceneFormat( root, savePath, ProgressBar::callBackSetProgress );

                    return[customFunction_ = customFunction_, savePath, res] ()
                    {
                        if ( res )
                        {
                            getViewerInstance().onSceneSaved( savePath );
                            customFunction_();
                        }
                        else
                            showError( "Error saving scene: " + res.error() );
                    };
                } );
            }
            UI::setTooltipIfHovered( "Save current scene and then remove all objects", scaling );
            ImGui::SameLine();
        }

        if ( UI::buttonCommonSize( showSave ? "Don't Save" : "Sign out", btnSize, ImGuiKey_N ) )
        {
            ImGui::CloseCurrentPopup();
            customFunction();
        }
        UI::setTooltipIfHovered( "Remove all objects without saving and ability to restore them", scaling );
        ImGui::SameLine();
        if ( UI::buttonCommonSize( "Cancel", btnSize, ImGuiKey_Escape ) )
            ImGui::CloseCurrentPopup();

        UI::setTooltipIfHovered( "Do not remove any objects, return back", scaling );

        if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
            ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar( 2 );
}

}
