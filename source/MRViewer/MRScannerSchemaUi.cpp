#include "MRScannerSchemaUi.h"

#include "MRUIStyle.h"
#include "MRRibbonConstants.h"
#include "MRViewer.h"
#include "MRFileDialog.h"
#include "ImGuiHelpers.h"
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRVisualObject.h>
#include <MRPch/MRSpdlog.h>
#include "MRI18n.h"
#include <imgui.h>

#include <limits>

namespace MR
{

namespace
{

// a text from the backend: its own catalog or MeshLib's
std::string trBackend( const std::string& s )
{
    return Locale::translate( s, Locale::genericDomain );
}

// The failure of an operation, or empty after a success
template <typename T>
std::string failureOf( const Expected<T>& res )
{
    return res ? std::string() : res.error();
}

// Route a capture/action result: hand any produced object to the host, or log the failure
void routeResult( const Expected<std::shared_ptr<VisualObject>>& res, const ScannerOnScan& onScan, std::string& error, const std::string& what )
{
    error = failureOf( res );
    if ( !res )
        spdlog::error( "Scanner '{}' failed: {}", what, error );
    else if ( *res && onScan )
        onScan( *res );
}

// The file for an action or a path param, by its "filters" and "save"; empty when cancelled
std::filesystem::path pickFile( const Json::Value& item, const std::string& filterName )
{
    const FileParameters params{ .filters = { { filterName, item["filters"].asString() } } };
    return item.get( "save", false ).asBool() ? saveFileDialog( params ) : openFileDialog( params );
}

// A float param's drag: its range, drag speed and display precision in unit E
template <UnitEnum E>
bool dragFloat( const char* label, float& v, const Json::Value& p, UnitToStringParams<E> unitParams = {} )
{
    if ( p.isMember( "precision" ) )
        unitParams.precision = p["precision"].asInt();
    return UI::drag<E>( label, v, p.get( "step", 0.1 ).asFloat(), p.get( "min", std::numeric_limits<float>::lowest() ).asFloat(),
        p.get( "max", std::numeric_limits<float>::max() ).asFloat(), unitParams );
}

void drawParam( IScannerBackend& backend, const Json::Value& p )
{
    const std::string key = p["key"].asString();
    const std::string type = p["type"].asString();
    const std::string label = trBackend( p.get( "label", key ).asString() );
    const std::string id = label + "##" + key; // labels may repeat, keys don't
    const Json::Value& value = p["value"];

    if ( type == "info" )
    {
        UI::alignTextToButton();
        ImGui::Text( "%s: %s", label.c_str(), trBackend( value.asString() ).c_str() );
        return;
    }

    ImGui::BeginDisabled( !p.get( "enabled", true ).asBool() );
    if ( type == "bool" )
    {
        bool v = value.asBool();
        if ( UI::checkbox( id.c_str(), &v ) )
            backend.setParam( key, v );
    }
    else if ( type == "float" )
    {
        float v = value.asFloat();
        ImGui::SetNextItemWidth( 90 * UI::scale() );
        if ( p.get( "unit", "" ).asString() == "ms"
            ? dragFloat<TimeUnit>( id.c_str(), v, p, { .sourceUnit = TimeUnit::milliseconds, .targetUnit = TimeUnit::milliseconds } )
            : dragFloat<NoUnit>( id.c_str(), v, p ) )
            backend.setParam( key, double( v ) );
    }
    else if ( type == "int" )
    {
        int v = value.asInt();
        ImGui::SetNextItemWidth( 90 * UI::scale() );
        if ( UI::drag<NoUnit>( id.c_str(), v, p.get( "step", 1 ).asInt(), p.get( "min", std::numeric_limits<int>::lowest() ).asInt(),
                p.get( "max", std::numeric_limits<int>::max() ).asInt() ) )
            backend.setParam( key, v );
    }
    else if ( type == "enum" )
    {
        int v = value.asInt();
        std::vector<std::string> opts;
        for ( const auto& o : p["options"] )
            opts.push_back( trBackend( o.asString() ) );
        if ( UI::combo( id.c_str(), &v, opts ) )
            backend.setParam( key, v );
    }
    else if ( type == "path" )
    {
        std::string v = value.asString();
        ImGui::SetNextItemWidth( ImGui::GetContentRegionAvail().x - 30.0f * UI::scale() );
        UI::inputText( ( "##" + key ).c_str(), v, ImGuiInputTextFlags_ReadOnly );
        ImGui::SameLine( 0.0f, 1.0f );
        if ( UI::button( ( "...##" + key ).c_str(), Vector2f( 24.0f * UI::scale(), 24.0f * UI::scale() ) ) )
            if ( std::filesystem::path path = pickFile( p, label ); !path.empty() )
                backend.setParam( key, utf8string( path ) );
    }
    ImGui::EndDisabled();
    if ( p.isMember( "tooltip" ) )
        UI::setTooltipIfHovered( trBackend( p["tooltip"].asString() ) );
}

// A backend-specific action group: its title and status line, then its buttons in a 3-wide grid.
// Returns true when an action ran.
bool drawActionGroup( IScannerBackend& backend, const Json::Value& group, const ScannerOnScan& onScan, std::string& error )
{
    if ( group.isMember( "title" ) )
        drawScannerSeparator( trBackend( group["title"].asString() ) );
    if ( group.isMember( "status" ) )
    {
        UI::alignTextToButton();
        ImGui::TextUnformatted( fmt::format( fmt::runtime( trBackend( group["status"].asString() ) ), group.get( "statusArg", "" ).asString() ).c_str() );
    }

    const ImGuiStyle& style = ImGui::GetStyle();
    const float w3 = ( ImGui::GetContentRegionAvail().x - 2 * style.ItemSpacing.x ) / 3;
    const bool advancedShown = ImGui::IsKeyDown( ImGuiMod_Shift ) && ImGui::IsKeyDown( UI::getImGuiModPrimaryCtrl() );
    bool ran = false;
    int col = 0;
    for ( const auto& a : group["actions"] )
    {
        if ( ( a.get( "advanced", false ).asBool() && !advancedShown ) ||
             ( a.get( "experimental", false ).asBool() && !getViewerInstance().experimentalFeatures ) )
            continue;
        const std::string key = a["key"].asString();
        if ( col != 0 )
            ImGui::SameLine();
        const std::string id = trBackend( a.get( "label", key ).asString() ) + "##" + key;
        if ( UI::button( id.c_str(), a.get( "enabled", true ).asBool(), Vector2f( w3, 0 ) ) )
        {
            Json::Value arg;
            bool proceed = true;
            if ( a.isMember( "filters" ) )
            {
                std::filesystem::path path = pickFile( a, _tr( "File" ) );
                if ( path.empty() )
                    proceed = false;
                else
                    arg = utf8string( path );
            }
            if ( proceed )
            {
                routeResult( backend.runAction( key, arg ), onScan, error, key );
                ran = true;
            }
        }
        if ( a.isMember( "tooltip" ) )
            UI::setTooltipIfHovered( trBackend( a["tooltip"].asString() ) );
        col = ( col + 1 ) % 3;
    }
    return ran;
}

} // namespace

void drawScannerParams( IScannerBackend& backend )
{
    const Json::Value schema = backend.paramSchema();
    for ( const auto& group : schema["groups"] )
    {
        if ( group.isMember( "title" ) )
            drawScannerSeparator( trBackend( group["title"].asString() ) );
        for ( const auto& p : group["params"] )
            drawParam( backend, p );
    }
}

bool drawScannerActions( IScannerBackend& backend, const ScannerOnScan& onScan, std::string& error )
{
    const Json::Value schema = backend.paramSchema();
    bool ran = false;
    for ( const auto& group : schema["actionGroups"] )
        ran = drawActionGroup( backend, group, onScan, error ) || ran;
    return ran;
}

// Live preview and the connect buttons are gated by capabilities(); capture is always available.
bool drawScannerCapture( IScannerBackend& backend, const ScannerOnScan& onScan, std::string& error )
{
    const ImGuiStyle& style = ImGui::GetStyle();
    const float w2 = ( ImGui::GetContentRegionAvail().x - style.ItemSpacing.x ) / 2;
    const float w3 = ( ImGui::GetContentRegionAvail().x - 2 * style.ItemSpacing.x ) / 3;
    const bool ready = backend.isReady();
    const bool live = backend.capabilities().livePreview;

    drawScannerSeparator( _tr( "Capture" ) );
    // no shortcut while a popup is open, e.g. the Rename one
    const bool popupOpen = ImGui::IsPopupOpen( nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel );
    const bool captured = UI::button( _tr( "Capture" ), ready, Vector2f( live ? w3 : -1, 0 ), popupOpen ? ImGuiKey_None : ImGuiKey_Space );
    if ( captured )
        routeResult( backend.capture(), onScan, error, "capture" );
    UI::setTooltipIfHovered( _tr( "Scan and create 3D (Press Space)" ) );
    if ( live )
    {
        ImGui::SameLine();
        if ( backend.livePreviewActive() )
        {
            if ( UI::button( _tr( "Stop" ), Vector2f( -1, 0 ) ) )
                backend.stopLivePreview();
        }
        else
        {
            if ( UI::button( _tr( "Live" ), ready, Vector2f( w3, 0 ) ) )
                backend.startLivePreview( false );
            ImGui::SameLine();
            if ( UI::button( _tr( "Live + 3D" ), ready, Vector2f( w3, 0 ) ) )
                backend.startLivePreview( true );
        }
        if ( UI::button( _tr( "Capture frame" ), ready, Vector2f( w2, 0 ) ) )
            error = failureOf( backend.captureFrame() );
    }
    if ( backend.capabilities().connectButtons )
    {
        if ( live )
            ImGui::SameLine();
        if ( ready )
        {
            if ( UI::button( _tr( "Disconnect" ), Vector2f( -1, 0 ) ) )
            {
                backend.disconnect();
                error.clear();
            }
        }
        else if ( UI::button( _tr( "Connect" ), Vector2f( -1, 0 ) ) )
            error = failureOf( backend.connect() );
    }

    if ( !error.empty() )
    {
        ImGui::Separator();
        UI::transparentTextWrapped( "%s", trBackend( error ).c_str() );
    }
    return captured;
}

void drawScannerSeparator( const std::string& title )
{
    // UI::separator adds cSeparateBlocksSpacing above and below, too much between short blocks
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cSeparateBlocksSpacing * UI::scale() );
    UI::separator( title );
    ImGui::SetCursorPosY( ImGui::GetCursorPosY() - cSeparateBlocksSpacing * UI::scale() );
}

} // namespace MR
