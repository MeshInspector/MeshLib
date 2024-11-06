#include "MRUISaveChangesPopup.h"

#include "MRRibbonConstants.h"
#include "MRUIStyle.h"
#include "ImGuiHelpers.h"
#include "MRRibbonFontManager.h"
#include "MRFileDialog.h"
#include "MRProgressBar.h"
#include "MRShowModal.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "MRSceneCache.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MRVisualObject.h"

namespace MR
{

namespace UI
{

void saveChangesPopup( const char* str_id, const SaveChangesPopupSettings& settings )
{
    const ImVec2 windowSize{ cModalWindowWidth * settings.scaling, -1 };
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * settings.scaling, 3.0f * cDefaultItemSpacing * settings.scaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * settings.scaling, cModalWindowPaddingY * settings.scaling } );
    if ( ImGui::BeginModalNoAnimation( str_id, nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            ImGui::PushFont( headerFont );

        const auto headerWidth = ImGui::CalcTextSize( settings.header.c_str() ).x;
        ImGui::SetCursorPosX( ( windowSize.x - headerWidth ) * 0.5f );
        ImGui::Text( "%s", settings.header.c_str() );

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
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * settings.scaling } );

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
                    ProgressBar::orderWithMainThreadPostProcessing( "Saving scene", [customFunction = settings.onOk, savePath, &root = SceneRoot::get()] ()->std::function<void()>
                {
                    auto res = ObjectSave::toAnySupportedSceneFormat( root, savePath, ProgressBar::callBackSetProgress );

                    return[customFunction = customFunction, savePath, res] ()
                    {
                        if ( res )
                        {
                            getViewerInstance().onSceneSaved( savePath );
                            if ( customFunction )
                                customFunction();
                        }
                        else
                            showError( "Error saving scene: " + res.error() );
                    };
                } );
            }
            if( !settings.saveTooltip.empty() )
                UI::setTooltipIfHovered( settings.saveTooltip.c_str(), settings.scaling );
            ImGui::SameLine();
        }

        if ( UI::buttonCommonSize( showSave ? settings.dontSaveText.c_str() : settings.shortCloseText.c_str(), btnSize, ImGuiKey_N) )
        {
            ImGui::CloseCurrentPopup();
            if ( settings.onOk )
                settings.onOk();
        }
        if ( !settings.dontSaveTooltip.empty() )
            UI::setTooltipIfHovered( settings.dontSaveTooltip.c_str(), settings.scaling );
        ImGui::SameLine();
        if ( UI::buttonCommonSize( "Cancel", btnSize, ImGuiKey_Escape ) )
            ImGui::CloseCurrentPopup();

        if ( !settings.cancelTooltip.empty() )
        UI::setTooltipIfHovered( settings.cancelTooltip.c_str(), settings.scaling);

        if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
            ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar( 2 );
}

}

}
