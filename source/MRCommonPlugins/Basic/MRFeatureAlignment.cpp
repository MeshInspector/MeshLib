#include "MRFeatureAlignment.h"
#include "MRFeatureAlignmentReport.h"
#include <MRViewer/MRUIStyle.h>
#include <MRViewer/MRRibbonRegisterItem.h>
#include <MRViewer/MRI18n.h>
#include <MRViewer/MRUITestEngine.h>
#include <MRMesh/MRSceneRoot.h>
#include <json/json.h>
#include <algorithm>
#include <numbers>

namespace MR
{
namespace
{
using namespace FeatureAlignmentTool;
ImVec4 roleColor( bool moving )
{
    const auto bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    const bool dark = bg.x + bg.y + bg.z < 1.5f;
    return moving ? ( dark ? ImVec4( .45f, .72f, 1, 1 ) : ImVec4( .08f, .32f, .68f, 1 ) )
                  : ( dark ? ImVec4( .35f, .82f, .61f, 1 ) : ImVec4( .04f, .40f, .25f, 1 ) );
}
void beginToleranceField( bool failed )
{
    if ( !failed )
        return;
    const auto bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    const bool dark = bg.x + bg.y + bg.z < 1.5f;
    const ImVec4 fill = dark ? ImVec4( .42f, .12f, .10f, 1 ) : ImVec4( 1, .82f, .78f, 1 );
    ImGui::PushStyleColor( ImGuiCol_FrameBg, fill );
    ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, fill );
    ImGui::PushStyleColor( ImGuiCol_FrameBgActive, fill );
    ImGui::PushStyleColor( ImGuiCol_Border, ImVec4( 1, .35f, .25f, 1 ) );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 1.f );
}
void endToleranceField( bool failed )
{
    if ( failed )
    {
        ImGui::PopStyleVar();
        ImGui::PopStyleColor( 4 );
    }
}
bool choose( const char *label, ObjectPtr &value, const Catalog &options, bool moving, bool optional )
{
    ImGui::PushStyleColor( ImGuiCol_Text, roleColor( moving ) );
    // UI::beginCombo computes custom arrow/text bounds before ImGui expands
    // negative widths. Pass an explicit positive width, including in tables.
    ImGui::SetNextItemWidth( ImGui::GetContentRegionAvail().x );
    bool changed = false;
    const auto found =
        std::find_if( options.begin(), options.end(), [&]( const auto &e ) { return e.object == value; } );
    const auto text = found == options.end() ? ( value      ? s_tr( "<not available>" )
                                                 : optional ? s_tr( "None" )
                                                            : s_tr( "Choose..." ) )
                                             : value->name();
    if ( UI::beginCombo( label, text ) )
    {
        if ( optional && UI::comboElem( _tr( "None" ), !value ) )
        {
            value.reset();
            changed = true;
        }
        for ( const auto &e : options )
            if ( UI::comboElem( e.path.c_str(), e.object == value ) )
            {
                value = e.object;
                changed = true;
            }
        UI::endCombo();
    }
    ImGui::PopStyleColor();
    if ( found != options.end() )
        UI::setTooltipIfHovered( found->path );
    return changed;
}
using Predicate = bool ( * )( const ObjectPtr & );
Predicate primaryPredicate( PrimaryMode mode )
{
    return mode == PrimaryMode::Plane ? isPlane : mode == PrimaryMode::Axis ? isAxis : isCenter;
}
Catalog features( const Catalog &entries, const Selection &s, bool moving, Predicate eligible )
{
    Catalog result;
    const auto &owner = moving ? s.moving : s.fixed;
    for ( const auto &e : entries )
        if ( owner && eligible( e.object ) && below( e.object.get(), owner.get() ) &&
             ( moving || !below( e.object.get(), s.moving.get() ) ) )
            result.push_back( e );
    return result;
}
void defaultPrimary( const Catalog &entries, Selection &s, bool moving, PrimaryMode mode )
{
    const auto candidates = features( entries, s, moving, primaryPredicate( mode ) );
    auto &primary = moving ? s.movingPrimary : s.fixedPrimary;
    primary.reset();
    std::vector<ObjectPtr> selected;
    for ( const auto &e : candidates )
        if ( e.object->isSelected() )
            selected.push_back( e.object );
    if ( selected.size() == 1 )
        primary = selected.front();
    else if ( candidates.size() == 1 )
        primary = candidates.front().object;
    ( moving ? s.movingLocators : s.fixedLocators ) = {};
    ( moving ? s.movingClock : s.fixedClock ).reset();
}
} // namespace
FeatureAlignmentPlugin::FeatureAlignmentPlugin() : StatePlugin( "Feature Alignment", StatePluginTabs::Basic ) {}
std::string FeatureAlignmentPlugin::isAvailable( const std::vector<std::shared_ptr<const Object>> & ) const
{
    using namespace FeatureAlignmentTool;
    if ( isActive() )
        return {};
    const auto entries = catalog( SceneRoot::getSharedPtr() );
    auto roles = selectedHierarchyRoles( entries );
    if ( !roles )
        return "Select two parts, or fitted features on two different parts.";
    Selection selection;
    selection.fixed = roles->fixed;
    selection.moving = roles->moving;
    for ( auto mode : { PrimaryMode::Plane, PrimaryMode::Axis, PrimaryMode::Centers } )
        if ( !features( entries, selection, false, primaryPredicate( mode ) ).empty() &&
             !features( entries, selection, true, primaryPredicate( mode ) ).empty() )
            return {};
    return "Both selected parts need compatible existing fitted features.";
}
bool FeatureAlignmentPlugin::onEnable_()
{
    using namespace FeatureAlignmentTool;
    const auto entries = catalog( SceneRoot::getSharedPtr() );
    auto roles = selectedHierarchyRoles( entries );
    if ( !roles )
        return false;
    if ( !controller_ || controller_->root() != SceneRoot::getSharedPtr() )
        controller_ = std::make_unique<Controller>( SceneRoot::getSharedPtr(), history_ );
    auto settings = controller_->settings();
    Selection selection;
    selection.fixed = roles->fixed;
    selection.moving = roles->moving;
    std::vector<ObjectPtr> selectedFeatures;
    for ( const auto &entry : entries )
        if ( entry.object->isSelected() &&
             ( isPlane( entry.object ) || isAxis( entry.object ) || isCenter( entry.object ) ) )
            selectedFeatures.push_back( entry.object );
    if ( selectedFeatures.size() == 2 )
    {
        if ( isPlane( selectedFeatures[0] ) && isPlane( selectedFeatures[1] ) )
            settings.mode = PrimaryMode::Plane;
        else if ( isAxis( selectedFeatures[0] ) && isAxis( selectedFeatures[1] ) )
            settings.mode = PrimaryMode::Axis;
        else if ( isCenter( selectedFeatures[0] ) && isCenter( selectedFeatures[1] ) )
            settings.mode = PrimaryMode::Centers;
    }
    if ( features( entries, selection, false, primaryPredicate( settings.mode ) ).empty() ||
         features( entries, selection, true, primaryPredicate( settings.mode ) ).empty() )
        for ( auto mode : { PrimaryMode::Plane, PrimaryMode::Axis, PrimaryMode::Centers } )
            if ( !features( entries, selection, false, primaryPredicate( mode ) ).empty() &&
                 !features( entries, selection, true, primaryPredicate( mode ) ).empty() )
            {
                settings.mode = mode;
                break;
            }
    if ( settings.mode == PrimaryMode::Centers )
        settings.offset = 0;
    defaultPrimary( entries, selection, false, settings.mode );
    defaultPrimary( entries, selection, true, settings.mode );
    if ( auto configured = controller_->configure( std::move( selection ), settings ); !configured )
        return false;
    secondHole_ = false;
    diagnostics_ = false;
    message_.clear();
    error_ = false;
    return true;
}
bool FeatureAlignmentPlugin::onDisable_()
{
    if ( controller_ )
        if ( auto canceled = controller_->invalidate( true ); !canceled )
        {
            error_ = true;
            message_ = canceled.error();
            return false;
        }
    return true;
}
void FeatureAlignmentPlugin::drawDialog( ImGuiContext * )
{
    using namespace FeatureAlignmentTool;
    const float scale = UI::scale();
    const auto *viewport = ImGui::GetMainViewport();
    const Vector2f workPosition{ viewport->WorkPos.x, viewport->WorkPos.y };
    const Vector2f workSize{ viewport->WorkSize.x, viewport->WorkSize.y };
    {
        const UI::TestEngine::TreeGuard tree( name() );
        if ( auto value = UI::TestEngine::createValueTentative<float>( "Window width" ) )
            panelSize_.x = *value;
        if ( auto value = UI::TestEngine::createValueTentative<float>( "Window height" ) )
            panelSize_.y = *value;
    }
    const auto minimum = fitPanelBounds( workPosition, { 380 * scale, 320 * scale }, workPosition, workSize, 8 * scale,
                                         { 380 * scale, 320 * scale } )
                             .size;
    const auto maximum = fitPanelBounds( workPosition, workSize, workPosition, workSize, 8 * scale ).size;
    const auto desired = fitPanelBounds( workPosition, panelSize_ * scale, workPosition, workSize, 8 * scale, minimum );
    // Keep this assembly tool in the application's viewport. The generic saved
    // dialog allocator can retain an off-screen position after monitor/DPI changes.
    ImGui::SetNextWindowViewport( viewport->ID );
    ImGui::SetNextWindowSizeConstraints( { minimum.x, minimum.y }, { maximum.x, maximum.y } );
    ImVec2 changedSize{ desired.size.x, desired.size.y };
    if ( !ImGuiBeginWindow_( { .width = desired.size.x,
                               .height = desired.size.y,
                               .allowScrollbar = false,
                               .flags = 0,
                               .changedSize = &changedSize } ) )
        return;
    panelSize_ = Vector2f{ changedSize.x, changedSize.y } / scale;
    const auto pos = ImGui::GetWindowPos();
    const auto size = ImGui::GetWindowSize();
    const auto bounds =
        fitPanelBounds( { pos.x, pos.y }, { size.x, size.y }, workPosition, workSize, 8 * scale, minimum );
    if ( bounds.position != Vector2f{ pos.x, pos.y } )
        ImGui::SetWindowPos( { bounds.position.x, bounds.position.y } );
    layout_.window = bounds;
    layout_.viewport = { workPosition, workSize };
    // Automation uses the same size state as native border/corner dragging.
    (void)UI::TestEngine::createValue( "Window width", panelSize_.x, minimum.x / scale, maximum.x / scale );
    (void)UI::TestEngine::createValue( "Window height", panelSize_.y, minimum.y / scale, maximum.y / scale );
    const auto entries = catalog( controller_->root() );
    auto selection = controller_->selection();
    auto settings = controller_->settings();
    bool changed = false;
    if ( selection.movingLocators[1] || selection.fixedLocators[1] )
        secondHole_ = true;
    const float actionHeight = 32 * scale;
    const auto bodyStart = ImGui::GetCursorScreenPos();
    const float footerTop = ImGui::GetWindowPos().y + ImGui::GetWindowSize().y - ImGui::GetStyle().WindowPadding.y -
                            actionHeight - 12 * scale;
    if ( ImGui::BeginChild( "Setup and results", { 0, std::max( 1.f, footerTop - bodyStart.y - 16 * scale ) } ) )
    {
        int mode = int( settings.mode );
        ImGui::SetNextItemWidth( 185 * scale );
        if ( UI::combo( _tr( "Align by" ), &mode,
                        { s_tr( "Mating planes" ), s_tr( "Feature axes" ), s_tr( "Feature centers" ) } ) )
        {
            settings.mode = PrimaryMode( mode );
            if ( settings.mode == PrimaryMode::Centers )
                settings.offset = 0;
            defaultPrimary( entries, selection, false, settings.mode );
            defaultPrimary( entries, selection, true, settings.mode );
            changed = true;
        }
        if ( ImGui::BeginTable( "Part roles", 2, ImGuiTableFlags_SizingStretchSame ) )
        {
            float roleHeight = 0;
            const float nameWidth = std::max( 1.f, ImGui::GetContentRegionAvail().x / 2 -
                                                       2 * ImGui::GetStyle().WindowPadding.x - 12 * scale );
            for ( const auto &part : { selection.fixed, selection.moving } )
            {
                const auto name = part ? part->name() : std::string( "Not available" );
                roleHeight = std::max( roleHeight, ImGui::CalcTextSize( name.c_str(), nullptr, false, nameWidth ).y +
                                                       2 * ImGui::GetStyle().WindowPadding.y );
            }
            for ( bool moving : { false, true } )
            {
                ImGui::TableNextColumn();
                ImGui::TextColored( roleColor( moving ), "%s",
                                    moving ? _tr( "Floating obj" ) : _tr( "Reference obj" ) );
                ImGui::PushStyleColor( ImGuiCol_ChildBg, ImVec4( 0, 0, 0, 1 ) );
                ImGui::PushStyleColor( ImGuiCol_Border, roleColor( moving ) );
                ImGui::PushStyleVar( ImGuiStyleVar_ChildRounding, 4 * scale );
                const auto &part = moving ? selection.moving : selection.fixed;
                const auto name = part ? part->name() : std::string( "Not available" );
                std::string path;
                for ( const auto &entry : entries )
                    if ( entry.object == part )
                        path = entry.path;
                if ( ImGui::BeginChild( moving ? "Floating role" : "Reference role", { 0, roleHeight },
                                        ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar ) )
                {
                    ImGui::PushStyleColor( ImGuiCol_Text, roleColor( moving ) );
                    ImGui::TextWrapped( "%s", name.c_str() );
                    ImGui::PopStyleColor();
                    UI::setTooltipIfHovered( path );
                }
                ImGui::EndChild();
                ImGui::PopStyleVar();
                ImGui::PopStyleColor( 2 );
                (void)UI::TestEngine::createValue( moving ? "##Moving part" : "##Fixed reference", path, true,
                                                   std::nullopt,
                                                   { .disabledReason = "Assigned from scene selection" } );
            }
            ImGui::EndTable();
        }
        UI::separator();
        const auto locatorPredicate = settings.mode == PrimaryMode::Plane ? isHole : isCenter;
        if ( ImGui::BeginTable( "Features", 3, ImGuiTableFlags_SizingStretchProp ) )
        {
            ImGui::TableSetupColumn( "Reference", ImGuiTableColumnFlags_WidthFixed, 58 * scale );
            ImGui::TableSetupColumn( "Reference obj", ImGuiTableColumnFlags_WidthStretch );
            ImGui::TableSetupColumn( "Floating obj", ImGuiTableColumnFlags_WidthStretch );
            const auto row = [&]( const char *title, const char *fixedLabel, const char *movingLabel, ObjectPtr &a,
                                  ObjectPtr &b, Predicate eligible, bool optional )
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted( title );
                ImGui::TableNextColumn();
                changed |= choose( fixedLabel, a, features( entries, selection, false, eligible ), false, optional );
                ImGui::TableNextColumn();
                changed |= choose( movingLabel, b, features( entries, selection, true, eligible ), true, optional );
            };
            const auto primaryLabel = settings.mode == PrimaryMode::Plane  ? s_tr( "Plane" )
                                      : settings.mode == PrimaryMode::Axis ? s_tr( "Axis" )
                                                                           : s_tr( "Center 1" );
            row( primaryLabel.c_str(), "##Fixed primary", "##Moving primary", selection.fixedPrimary,
                 selection.movingPrimary, primaryPredicate( settings.mode ), false );
            const auto firstLabel = settings.mode == PrimaryMode::Plane  ? s_tr( "Hole 1" )
                                    : settings.mode == PrimaryMode::Axis ? s_tr( "Locator 1" )
                                                                         : s_tr( "Center 2" );
            row( firstLabel.c_str(), "##Fixed locator 1", "##Moving locator 1", selection.fixedLocators[0],
                 selection.movingLocators[0], locatorPredicate, true );
            if ( secondHole_ )
            {
                const auto secondLabel = settings.mode == PrimaryMode::Plane  ? s_tr( "Hole 2" )
                                         : settings.mode == PrimaryMode::Axis ? s_tr( "Locator 2" )
                                                                              : s_tr( "Center 3" );
                row( secondLabel.c_str(), "##Fixed locator 2", "##Moving locator 2", selection.fixedLocators[1],
                     selection.movingLocators[1], locatorPredicate, true );
            }
            ImGui::EndTable();
        }
        const auto extraLabel = settings.mode == PrimaryMode::Plane  ? s_tr( "Use a second hole pair" )
                                : settings.mode == PrimaryMode::Axis ? s_tr( "Use a second center locator" )
                                                                     : s_tr( "Use a third center pair" );
        if ( UI::checkbox( extraLabel.c_str(), &secondHole_ ) && !secondHole_ )
        {
            selection.movingLocators[1].reset();
            selection.fixedLocators[1].reset();
            changed = true;
        }
        if ( settings.mode == PrimaryMode::Plane )
        {
            int relation = settings.normals == NormalRelation::Opposing ? 0 : 1;
            ImGui::SetNextItemWidth( 150 * scale );
            if ( UI::combo( _tr( "Faces" ), &relation, { s_tr( "Opposing" ), s_tr( "Same" ) } ) )
            {
                settings.normals = relation == 0 ? NormalRelation::Opposing : NormalRelation::Same;
                changed = true;
            }
            ImGui::SetNextItemWidth( 150 * scale );
            changed |= UI::input<LengthUnit>( _tr( "Gap" ), settings.offset );
            UI::setTooltipIfHovered(
                s_tr( "Positive gap follows the fixed plane normal. Check the fitted normal direction." ) );
            int holeMode = settings.holeMode == HoleMode::Centers ? 0 : 1;
            ImGui::SetNextItemWidth( 150 * scale );
            if ( UI::combo( _tr( "Locate" ), &holeMode, { s_tr( "Hole centers" ), s_tr( "Hole axes" ) } ) )
            {
                settings.holeMode = holeMode == 0 ? HoleMode::Centers : HoleMode::Axes;
                changed = true;
            }
            UI::setTooltipIfHovered( _tr( "Centers locate within the mating plane; the plane gap sets depth. "
                                          "Axes locate their plane intersections and also check axis angles." ) );
            if ( secondHole_ )
            {
                changed |= UI::checkbox( _tr( "Balance hole pairs" ), &settings.balanceHoles );
                UI::setTooltipIfHovered(
                    _tr( "Share in-plane mismatch between both pairs without changing the plane gap "
                         "or rigid geometry. Off keeps the first hole exact." ) );
            }
            UI::transparentTextWrapped(
                settings.holeMode == HoleMode::Centers
                    ? _tr( "The plane gap fixes depth; hole centers locate within the plane." )
                    : _tr( "Hole axes locate bores at the plane and check their directions." ) );
        }
        else if ( settings.mode == PrimaryMode::Axis )
        {
            int direction = settings.axisDirection == AxisDirection::Closest ? 0
                            : settings.axisDirection == AxisDirection::Same  ? 1
                                                                             : 2;
            ImGui::SetNextItemWidth( 150 * scale );
            if ( UI::combo( _tr( "Axis direction" ), &direction,
                            { s_tr( "Unoriented / closest" ), s_tr( "Same" ), s_tr( "Opposing" ) } ) )
            {
                settings.axisDirection = direction == 0   ? AxisDirection::Closest
                                         : direction == 1 ? AxisDirection::Same
                                                          : AxisDirection::Opposing;
                changed = true;
            }
            changed |= UI::checkbox( _tr( "Match primary centers" ), &settings.matchAxialCenters );
            ImGui::SetNextItemWidth( 150 * scale );
            changed |= UI::input<LengthUnit>( _tr( "Axial offset" ), settings.offset );
            UI::setTooltipIfHovered( s_tr( "Offset follows the fixed fitted axis. Select a fitted rim circle when "
                                           "axial placement must refer to a cylinder edge." ) );
        }
        const auto *last =
            controller_->result() && !changed && !controller_->stalePreview() ? &*controller_->result() : nullptr;
        const bool positionFailed = last && last->maximumPositionError() > settings.linearTolerance;
        const bool angleFailed = last && last->maximumAngleError() > settings.angularToleranceDeg;
        beginToleranceField( positionFailed );
        ImGui::SetNextItemWidth( 150 * scale );
        changed |= UI::input<LengthUnit>( _tr( "Position tolerance" ), settings.linearTolerance, 0. );
        endToleranceField( positionFailed );
        if ( positionFailed && scrollToResult_ )
        {
            ImGui::SetScrollHereY( 0 );
            scrollToResult_ = false;
        }
        UI::setTooltipIfHovered( _tr( "Maximum permitted placement error. For planes and holes, depth comes from "
                                      "the plane gap and hole errors are measured within the plane." ) );
        if ( positionFailed )
        {
            const auto measured = valueToString<LengthUnit>( last->maximumPositionError() );
            const auto allowed = valueToString<LengthUnit>( settings.linearTolerance );
            ImGui::TextColored( ImVec4( 1, .45f, .35f, 1 ), _tr( "Measured %s; allowed %s" ), measured.c_str(),
                                allowed.c_str() );
        }
        if ( settings.mode != PrimaryMode::Centers )
        {
            double angle = settings.angularToleranceDeg * std::numbers::pi / 180.;
            beginToleranceField( angleFailed );
            ImGui::SetNextItemWidth( 150 * scale );
            if ( UI::input<AngleUnit>( _tr( "Angle tolerance" ), angle, 0., std::numbers::pi / 2 - 1e-6 ) )
            {
                settings.angularToleranceDeg = angle * 180. / std::numbers::pi;
                changed = true;
            }
            endToleranceField( angleFailed );
            if ( angleFailed && scrollToResult_ )
            {
                ImGui::SetScrollHereY( 0 );
                scrollToResult_ = false;
            }
            UI::setTooltipIfHovered( _tr( "Maximum permitted angle error. Hole-axis angles are checked only "
                                          "when Locate is set to Hole axes." ) );
            if ( angleFailed )
            {
                const auto measured = valueToString<AngleUnit>( last->maximumAngleError() * std::numbers::pi / 180. );
                const auto allowed = valueToString<AngleUnit>( angle );
                ImGui::TextColored( ImVec4( 1, .45f, .35f, 1 ), _tr( "Measured %s; allowed %s" ), measured.c_str(),
                                    allowed.c_str() );
            }
        }
        const auto defaultPosition = valueToString<LengthUnit>( Settings{}.linearTolerance );
        const auto defaultAngle = valueToString<AngleUnit>( Settings{}.angularToleranceDeg * std::numbers::pi / 180. );
        if ( settings.mode == PrimaryMode::Centers )
            UI::transparentTextWrapped( _tr( "Suggested start: %s. Adjust for your scans." ), defaultPosition.c_str() );
        else
            UI::transparentTextWrapped( _tr( "Suggested start: %s / %s. Adjust for your scans." ),
                                        defaultPosition.c_str(), defaultAngle.c_str() );
        if ( auto value = UI::TestEngine::createValueTentative<bool>( "Advanced" ) )
            ImGui::SetNextItemOpen( *value, ImGuiCond_Always );
        advanced_ = ImGui::CollapsingHeader( _tr( "Advanced" ) );
        (void)UI::TestEngine::createValue( "Advanced", advanced_, false, true );
        if ( advanced_ )
        {
            if ( settings.mode == PrimaryMode::Axis )
                changed |= UI::checkbox( _tr( "Make primary axes coaxial" ), &settings.coaxial );
            if ( settings.mode != PrimaryMode::Centers )
            {
                UI::transparentTextWrapped( _tr( "Optional directed rotation reference:" ) );
                if ( ImGui::BeginTable( "Clocking", 2, ImGuiTableFlags_SizingStretchSame ) )
                {
                    ImGui::TableNextColumn();
                    changed |= choose( "##Fixed clocking", selection.fixedClock,
                                       features( entries, selection, false, isDirection ), false, true );
                    ImGui::TableNextColumn();
                    changed |= choose( "##Moving clocking", selection.movingClock,
                                       features( entries, selection, true, isDirection ), true, true );
                    ImGui::EndTable();
                }
            }
        }
        if ( changed )
        {
            auto configured = controller_->configure( selection, settings );
            message_ = configured ? std::string() : configured.error();
            error_ = !configured;
        }
        UI::separator();
        if ( controller_->stalePreview() )
            UI::notificationFrame( NotificationType::Warning,
                                   s_tr( "The scene or tree order changed after Preview. Preview again." ) );
        if ( const auto &result = controller_->result() )
        {
            const auto &r = *result;
            const auto heading = !r.acceptable()                   ? s_tr( "Outside tolerance" )
                                 : controller_->previewDisplayed() ? s_tr( "Preview displayed" )
                                 : controller_->canRestore()       ? s_tr( "Placement applied" )
                                 : controller_->hasPreview()       ? s_tr( "Preview checked" )
                                                                   : s_tr( "Previous placement result" );
            ImGui::TextColored( r.acceptable() ? roleColor( false ) : ImVec4( 1, .45f, .35f, 1 ), "%s",
                                heading.c_str() );
            ImGui::TextWrapped( "%s", r.freeMotion.c_str() );
            const auto residual = valueToString<LengthUnit>( r.maximumPositionError() );
            ImGui::Text( _tr( "Max constrained residual: %s" ), residual.c_str() );
            if ( auto value = UI::TestEngine::createValueTentative<bool>( "Diagnostics" ) )
                diagnostics_ = *value;
            ImGui::SetNextItemOpen( diagnostics_, ImGuiCond_Always );
            diagnostics_ = ImGui::CollapsingHeader( _tr( "Diagnostics" ) );
            (void)UI::TestEngine::createValue( "Diagnostics", diagnostics_, false, true );
            if ( diagnostics_ )
            {
                if ( r.mode != PrimaryMode::Centers )
                {
                    const auto angle = valueToString<AngleUnit>( r.primaryAngleErrorDeg * std::numbers::pi / 180. );
                    ImGui::Text( _tr( "Primary angle residual: %s" ), angle.c_str() );
                }
                for ( size_t i = 0; i < r.holes.size(); ++i )
                {
                    const auto position = valueToString<LengthUnit>( r.holes[i].position );
                    if ( r.mode == PrimaryMode::Plane )
                        ImGui::Text( _tr( "Hole %d in plane: %s" ), int( i + 1 ), position.c_str() );
                    else
                        ImGui::Text( _tr( "Locator %d: %s" ), int( i + 1 ), position.c_str() );
                }
                for ( const auto &conflict : r.conflicts )
                    ImGui::TextWrapped( "%s", conflict.c_str() );
                for ( const auto &warning : r.warnings )
                    UI::transparentTextWrapped( "%s", warning.c_str() );
                if ( UI::button( _tr( "Copy report" ) ) )
                    ImGui::SetClipboardText( reportJson( *controller_ ).c_str() );
            }
        }
        if ( !message_.empty() )
        {
            if ( error_ )
                UI::notificationFrame( NotificationType::Warning, message_ );
            else
                UI::transparentTextWrapped( "%s", message_.c_str() );
        }
        else if ( !controller_->result() )
        {
            auto ready = readConstraints( controller_->root(), controller_->selection(), controller_->settings() );
            UI::transparentTextWrapped( "%s", ready ? _tr( "Ready to preview. Reference obj stays fixed." )
                                                    : ready.error().c_str() );
        }
        if ( scrollToResult_ )
        {
            ImGui::SetScrollHereY( 1 );
            scrollToResult_ = false;
        }
    }
    ImGui::EndChild();
    // Use the application's own automation surface, including in production.
    // Native widget diagnostics and clipboard reports cannot change the placement.
    (void)UI::TestEngine::createValue( "Report JSON", reportJson( *controller_ ), true, std::nullopt,
                                       { .disabledReason = "Read-only alignment report" } );
    ImGui::SetCursorScreenPos( { bodyStart.x, footerTop - 8 * scale } );
    ImGui::Separator();
    // UI::separator includes extra vertical padding. Anchor the button row
    // explicitly so the native custom-window clip cannot cut its lower edge.
    ImGui::SetCursorScreenPos( { bodyStart.x, footerTop } );
    // Keep actions visible while setup/results scroll. The footer never depends
    // on result length, advanced options, label wrapping or native UI scaling.
    const auto width = ( ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().ItemSpacing.x ) / 3;
    const bool ready =
        bool( readConstraints( controller_->root(), controller_->selection(), controller_->settings() ) );
    const auto clipMin = ImGui::GetWindowDrawList()->GetClipRectMin();
    const auto clipMax = ImGui::GetWindowDrawList()->GetClipRectMax();
    const auto actionVisible = [&]()
    {
        const auto a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        return a.x >= clipMin.x && a.y >= clipMin.y && b.x <= clipMax.x && b.y <= clipMax.y;
    };
    if ( UI::button( _tr( "Preview" ), ready, { width, actionHeight } ) )
    {
        auto result = controller_->preview( true );
        error_ = !result;
        message_ = result ? s_tr( "Temporary pose. Apply to keep it; Restore to cancel." ) : result.error();
        scrollToResult_ = true;
    }
    layout_.actionsInsideClip = actionVisible();
    ImGui::SameLine();
    if ( UI::button( _tr( "Apply" ), controller_->canApply(), { width, actionHeight } ) )
    {
        auto result = controller_->apply();
        error_ = !result;
        message_ = result ? s_tr( "Applied to Floating obj. Native Undo is available." ) : result.error();
        scrollToResult_ = true;
    }
    layout_.actionsInsideClip &= actionVisible();
    ImGui::SameLine();
    if ( UI::button( _tr( "Restore" ), controller_->canRestore(), { width, actionHeight } ) )
    {
        auto result = controller_->restore();
        error_ = !result;
        message_ = result ? s_tr( "Original placement restored." ) : result.error();
        scrollToResult_ = true;
    }
    layout_.actionsInsideClip &= actionVisible();
    Json::Value layout;
    layout["resize_enabled"] = true;
    layout["actions_inside_clip"] = layout_.actionsInsideClip;
    layout["ui_scale"] = scale;
    for ( int i = 0; i < 2; ++i )
    {
        layout["window_position"].append( bounds.position[i] );
        layout["window_size"].append( size[i] );
        layout["logical_size"].append( panelSize_[i] );
        layout["minimum_size"].append( minimum[i] );
        layout["maximum_size"].append( maximum[i] );
        layout["viewport_position"].append( workPosition[i] );
        layout["viewport_size"].append( workSize[i] );
    }
    (void)UI::TestEngine::createValue( "Panel layout", Json::writeString( Json::StreamWriterBuilder{}, layout ), true,
                                       std::nullopt, { .disabledReason = "Read-only panel layout" } );
    ImGui::EndCustomStatePlugin();
}
MR_REGISTER_RIBBON_ITEM( FeatureAlignmentPlugin )
} // namespace MR
