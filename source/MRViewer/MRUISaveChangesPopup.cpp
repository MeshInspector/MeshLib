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
#include "MRModalDialog.h"
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
    // do not suggest saving empty scene
    const bool showSave = !SceneCache::getAllObjects<VisualObject, ObjectSelectivityType::Selectable>().empty();

    ModalDialog dialog( str_id, {
        .headline = settings.header,
        .text = showSave ? "Save your changes?" : "",
        .closeOnClickOutside = true,
    } );
    if ( dialog.beginPopup( settings.scaling ) )
    {
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
            UI::setTooltipIfHovered( settings.cancelTooltip.c_str(), settings.scaling );

        ImGui::PopStyleVar(); // ImGuiStyleVar_FramePadding
        dialog.endPopup( settings.scaling );
    }
}

}

}
