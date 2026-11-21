#include "MRFeatureAlignment.h"
#include <MRViewer/MRUIStyle.h>
#include <MRViewer/MRColorTheme.h>
#include <MRViewer/MRRibbonRegisterItem.h>
#include <MRViewer/MRI18n.h>
#include <MRMesh/MRSceneRoot.h>
#include <algorithm>
#include <numbers>

namespace MR
{
namespace
{
using namespace FeatureAlignmentTool;
ImVec4 roleColor( bool moving )
{
    const auto type = moving ? ColorTheme::RibbonColorsType::SelectedObjectText : ColorTheme::RibbonColorsType::Text;
    return ImGui::ColorConvertU32ToFloat4( ColorTheme::getRibbonColor( type ).getUInt32() );
}
ImVec4 errorColor()
{
    return ImGui::ColorConvertU32ToFloat4( UI::notificationChar( NotificationType::Error ).second );
}
void beginToleranceField( bool failed )
{
    if ( !failed )
        return;
    auto fill = errorColor();
    fill.w *= .18f;
    ImGui::PushStyleColor( ImGuiCol_FrameBg, fill );
    ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, fill );
    ImGui::PushStyleColor( ImGuiCol_FrameBgActive, fill );
    ImGui::PushStyleColor( ImGuiCol_Border, errorColor() );
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
std::string rotationSourceText( FeatureAlignment::RotationSource source )
{
    using FeatureAlignment::RotationSource;
    switch ( source )
    {
    case RotationSource::Unconstrained:
        return s_tr( "No rotation reference" );
    case RotationSource::OffAxisCenterLocator:
        return s_tr( "Off-axis center locator" );
    case RotationSource::DirectedReference:
        return s_tr( "Directed reference" );
    case RotationSource::SecondCenter:
        return s_tr( "Second center pair" );
    case RotationSource::ThirdCenter:
        return s_tr( "Third center pair" );
    case RotationSource::SecondHole:
        return s_tr( "Second hole pair" );
    case RotationSource::ObliquePrimaryAxis:
        return s_tr( "Oblique hole axis" );
    }
    return s_tr( "No rotation reference" );
}
std::string freeMotionText( const Solution &result, PrimaryMode mode )
{
    using FeatureAlignment::FreeMotion;
    switch ( result.freeMotion )
    {
    case FreeMotion::Unspecified:
        return s_tr( "Select references to inspect the remaining motion." );
    case FreeMotion::FullyConstrained:
        return result.acceptable() ? s_tr( "Fully located by the selected references." )
                                   : s_tr( "No motion remains, but the selected measurements conflict." );
    case FreeMotion::PlaneTranslationsAndSpin:
        return s_tr( "Two in-plane translations and rotation about the plane normal remain free." );
    case FreeMotion::PlaneTranslations:
        return s_tr( "Two in-plane translations remain free." );
    case FreeMotion::PrimarySpin:
        return mode == PrimaryMode::Plane ? s_tr( "Rotation about the plane normal remains free." )
                                                : s_tr( "Rotation about the primary axis remains free." );
    case FreeMotion::PrimarySlide:
        return s_tr( "Sliding along the primary axis remains free." );
    case FreeMotion::PrimarySlideAndSpin:
        return s_tr( "Sliding and rotation about the primary axis remain free." );
    case FreeMotion::Translations:
        return s_tr( "All three translations remain free." );
    case FreeMotion::TranslationsAndSpin:
        return s_tr( "All three translations and rotation about the primary axis remain free." );
    case FreeMotion::Rotations:
        return s_tr( "All three rotations about the matched primary center remain free." );
    case FreeMotion::BaselineSpin:
        return s_tr( "Rotation about the line through the first two centers remains free." );
    }
    return s_tr( "Review the selected references." );
}
void drawDiagnostic( const FeatureAlignment::Diagnostic &diagnostic, const Solution &result, PrimaryMode mode )
{
    using FeatureAlignment::DiagnosticCode;
    const int number = diagnostic.index + 1;
    switch ( diagnostic.code )
    {
    case DiagnosticCode::CenterPositionMismatch:
        ImGui::TextWrapped( mode == PrimaryMode::Centers
                                ? _tr( "Center %d residual exceeds Position tolerance." )
                                : _tr( "Locator %d residual exceeds Position tolerance." ), number );
        return;
    case DiagnosticCode::PrimaryCenterMismatch:
        ImGui::TextWrapped( "%s", _tr( "Primary feature centers conflict with the locators." ) );
        return;
    case DiagnosticCode::PrimaryAxisNotCoaxial:
        ImGui::TextWrapped( "%s", _tr( "Primary axes are not coaxial within Position tolerance." ) );
        return;
    case DiagnosticCode::PrimaryAxisAngleMismatch:
        ImGui::TextWrapped( "%s", _tr( "Primary axis angle exceeds Angle tolerance." ) );
        return;
    case DiagnosticCode::ClockingParallelToPrimary:
        ImGui::TextWrapped( "%s", mode == PrimaryMode::Plane
                                     ? _tr( "Clocking direction is parallel to the plane normal." )
                                     : _tr( "Clocking direction is parallel to the primary axis." ) );
        return;
    case DiagnosticCode::ClockingConflict:
        ImGui::TextWrapped( "%s", _tr( "Clocking reference exceeds Angle tolerance." ) );
        return;
    case DiagnosticCode::PlaneSeatingMismatch:
        ImGui::TextWrapped( "%s", _tr( "Plane seating exceeds Position or Angle tolerance." ) );
        return;
    case DiagnosticCode::HolePositionMismatch:
        ImGui::TextWrapped( _tr( "Hole %d in-plane mismatch exceeds Position tolerance." ), number );
        return;
    case DiagnosticCode::HoleAxisAngleMismatch:
        ImGui::TextWrapped( _tr( "Hole %d axis angle exceeds Angle tolerance." ), number );
        return;
    case DiagnosticCode::HoleRadiusMismatch:
        ImGui::TextWrapped( _tr( "Hole %d radii differ. Review the fitted features and clearance." ), number );
        return;
    case DiagnosticCode::Underconstrained:
        ImGui::TextWrapped( _tr( "%d degrees of freedom remain. Add a reference if needed." ), result.remainingDof );
        return;
    case DiagnosticCode::FittedAxialCenters:
        ImGui::TextWrapped( "%s", _tr( "Axial placement uses fitted feature centers. Select a rim circle to locate a cylinder edge." ) );
        return;
    case DiagnosticCode::AmbiguousAxisDirection:
        ImGui::TextWrapped( "%s", _tr( "Unoriented axes admit two direction choices. Use Same or Opposing to choose explicitly." ) );
        return;
    case DiagnosticCode::ShortLocatorLeverArm:
        ImGui::TextWrapped( mode == PrimaryMode::Centers
                                ? _tr( "Center %d is within Position tolerance of the baseline. Rotation is unreliable." )
                                : _tr( "Locator %d is within Position tolerance of the axis. Rotation is unreliable." ), number );
        return;
    case DiagnosticCode::CollinearCenters:
        ImGui::TextWrapped( "%s", _tr( "The centers are collinear. A noncollinear third pair can set rotation." ) );
        return;
    case DiagnosticCode::PlaneCentersProjected:
        ImGui::TextWrapped( "%s", _tr( "Hole centers locate within the mating plane; the plane gap sets depth." ) );
        return;
    case DiagnosticCode::BalancedHoleFit:
        ImGui::TextWrapped( "%s", _tr( "Balanced fit shares in-plane mismatch between both hole pairs." ) );
        return;
    case DiagnosticCode::ObliqueHoleAxis:
        ImGui::TextWrapped( "%s", _tr( "An oblique hole axis sets rotation. A second hole can provide a steadier reference." ) );
        return;
    case DiagnosticCode::OppositePlaneNormalsConvention:
        ImGui::TextWrapped( "%s", _tr( "Opposite normals use a rotation-axis convention. Add a clocking reference to set rotation." ) );
        return;
    case DiagnosticCode::Unspecified:
        break;
    }
    ImGui::TextWrapped( "%s", _tr( "Review the selected references and tolerances." ) );
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
        {
            ImGui::PushID( e.object.get() );
            if ( UI::comboElem( e.path.c_str(), e.object == value ) )
            {
                value = e.object;
                changed = true;
            }
            ImGui::PopID();
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
std::string FeatureAlignmentPlugin::isAvailable( const std::vector<std::shared_ptr<const Object>> &selectedObjects ) const
{
    using namespace FeatureAlignmentTool;
    if ( isActive() )
        return shouldClose_() ? s_tr( "The scene or selected parts changed. Reopen Feature Alignment." )
                              : std::string{};
    const auto roles = selectedHierarchyRoles( SceneRoot::getSharedPtr(), selectedObjects );
    if ( !roles )
        return s_tr( "Select two parts, or fitted features on two different parts." );
    if ( !haveCompatiblePrimaryFeatures( *roles ) )
        return s_tr( "Both selected parts need compatible existing fitted features." );
    return {};
}
void FeatureAlignmentPlugin::updateSelection( const std::vector<std::shared_ptr<const Object>> & )
{
    if ( controller_ )
        controller_->refresh();
}
bool FeatureAlignmentPlugin::onEnable_()
{
    using namespace FeatureAlignmentTool;
    const auto entries = catalog( SceneRoot::getSharedPtr() );
    auto roles = selectedHierarchyRoles( entries );
    if ( !roles )
        return false;
    auto settings = settings_;
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
    controller_ = std::make_unique<Controller>( SceneRoot::getSharedPtr() );
    controller_->configure( std::move( selection ), settings );
    secondHole_ = false;
    diagnostics_ = false;
    message_.clear();
    error_ = false;
    return true;
}
bool FeatureAlignmentPlugin::onDisable_()
{
    if ( controller_ )
    {
        settings_ = controller_->settings();
    }
    controller_.reset();
    return true;
}
bool FeatureAlignmentPlugin::shouldClose_() const
{
    if ( !controller_ )
        return false;
    const auto &root = SceneRoot::getSharedPtr();
    const auto &selection = controller_->selection();
    return controller_->root() != root || !FeatureAlignmentTool::below( selection.fixed.get(), root.get() ) ||
           !FeatureAlignmentTool::below( selection.moving.get(), root.get() );
}
void FeatureAlignmentPlugin::drawDialog( ImGuiContext * )
{
    using namespace FeatureAlignmentTool;
    if ( shouldClose_() )
    {
        (void)enable( false );
        return;
    }
    const float scale = UI::scale();
    ImVec2 changedSize{ panelSize_.x * scale, panelSize_.y * scale };
    if ( !ImGuiBeginWindow_( { .width = changedSize.x,
                               .height = changedSize.y,
                               .flags = 0,
                               .changedSize = &changedSize } ) )
        return;
    panelSize_ = Vector2f{ changedSize.x, changedSize.y } / scale;
    const auto &view = controller_->viewState();
    const auto &entries = view.entries;
    auto selection = controller_->selection();
    auto settings = controller_->settings();
    bool changed = false;
    if ( selection.movingLocators[1] || selection.fixedLocators[1] )
        secondHole_ = true;
    const float actionHeight = 32 * scale;
    if ( ImGui::BeginChild( "Setup and results", { 0, -( actionHeight + ImGui::GetStyle().ItemSpacing.y ) } ) )
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
            controller_->configure( selection, settings );
            message_.clear();
            error_ = false;
            selection = controller_->selection();
            settings = controller_->settings();
            secondHole_ = bool( selection.fixedLocators[1] || selection.movingLocators[1] );
            (void)controller_->viewState();
        }
        ImGui::SameLine();
        if ( UI::button( _tr( "Swap" ), view.canSwapRoles ) )
        {
            auto swapped = controller_->swapRoles();
            error_ = !swapped;
            message_ = swapped ? s_tr( "Roles exchanged. Review the placement before applying." ) : s_tr( swapped.error() );
            selection = controller_->selection();
            settings = controller_->settings();
            secondHole_ = bool( selection.fixedLocators[1] || selection.movingLocators[1] );
            if ( swapped )
                diagnostics_ = false;
            scrollToResult_ = false;
            changed = false;
            (void)controller_->viewState();
        }
        UI::setTooltipIfHovered(
            view.canSwapRoles
                ? _tr( "Exchange roles and their features. Gap and axial offset follow the new Reference object's direction." )
                : _tr( "Swap needs separate part branches. "
                       "Gap and axial offset follow the new Reference object's direction." ) );
        if ( ImGui::BeginTable( "Part roles", 2, ImGuiTableFlags_SizingStretchSame ) )
        {
            float roleHeight = 0;
            const float nameWidth = std::max( 1.f, ImGui::GetContentRegionAvail().x / 2 -
                                                       2 * ImGui::GetStyle().WindowPadding.x - 12 * scale );
            for ( const auto &part : { selection.fixed, selection.moving } )
            {
                const auto name = part ? part->name() : s_tr( "Not available" );
                roleHeight = std::max( roleHeight, ImGui::CalcTextSize( name.c_str(), nullptr, false, nameWidth ).y +
                                                       2 * ImGui::GetStyle().WindowPadding.y );
            }
            for ( bool moving : { false, true } )
            {
                ImGui::TableNextColumn();
                ImGui::TextColored( roleColor( moving ), "%s",
                                    moving ? _tr( "Floating object" ) : _tr( "Reference object" ) );
                ImGui::PushStyleColor( ImGuiCol_ChildBg,
                                       ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::FrameBackground ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_Border, roleColor( moving ) );
                ImGui::PushStyleVar( ImGuiStyleVar_ChildRounding, 4 * scale );
                const auto &part = moving ? selection.moving : selection.fixed;
                const auto name = part ? part->name() : s_tr( "Not available" );
                if ( ImGui::BeginChild( moving ? "Floating role" : "Reference role", { 0, roleHeight },
                                        ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar ) )
                {
                    ImGui::PushStyleColor( ImGuiCol_Text, roleColor( moving ) );
                    ImGui::TextWrapped( "%s", name.c_str() );
                    ImGui::PopStyleColor();
                }
                ImGui::EndChild();
                ImGui::PopStyleVar();
                ImGui::PopStyleColor( 2 );
            }
            ImGui::EndTable();
        }
        UI::separator();
        if ( ImGui::BeginTable( "Features", 3, ImGuiTableFlags_SizingStretchProp ) )
        {
            ImGui::TableSetupColumn( "Reference", ImGuiTableColumnFlags_WidthFixed, 58 * scale );
            ImGui::TableSetupColumn( "Reference object", ImGuiTableColumnFlags_WidthStretch );
            ImGui::TableSetupColumn( "Floating object", ImGuiTableColumnFlags_WidthStretch );
            const auto row = [&]( const char *title, const char *fixedLabel, const char *movingLabel, ObjectPtr &a,
                                  ObjectPtr &b, const std::array<Catalog, 2> &options, bool optional )
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted( title );
                ImGui::TableNextColumn();
                changed |= choose( fixedLabel, a, options[0], false, optional );
                ImGui::TableNextColumn();
                changed |= choose( movingLabel, b, options[1], true, optional );
            };
            const auto primaryLabel = settings.mode == PrimaryMode::Plane  ? s_tr( "Plane" )
                                      : settings.mode == PrimaryMode::Axis ? s_tr( "Axis" )
                                                                           : s_tr( "Center 1" );
            row( primaryLabel.c_str(), "##Fixed primary", "##Moving primary", selection.fixedPrimary,
                 selection.movingPrimary, view.primaryFeatures, false );
            const auto firstLabel = settings.mode == PrimaryMode::Plane  ? s_tr( "Hole 1" )
                                    : settings.mode == PrimaryMode::Axis ? s_tr( "Locator 1" )
                                                                         : s_tr( "Center 2" );
            row( firstLabel.c_str(), "##Fixed locator 1", "##Moving locator 1", selection.fixedLocators[0],
                 selection.movingLocators[0], view.locatorFeatures, true );
            if ( secondHole_ )
            {
                const auto secondLabel = settings.mode == PrimaryMode::Plane  ? s_tr( "Hole 2" )
                                         : settings.mode == PrimaryMode::Axis ? s_tr( "Locator 2" )
                                                                              : s_tr( "Center 3" );
                row( secondLabel.c_str(), "##Fixed locator 2", "##Moving locator 2", selection.fixedLocators[1],
                     selection.movingLocators[1], view.locatorFeatures, true );
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
            controller_->result() && !changed ? &*controller_->result() : nullptr;
        const bool positionFailed = last && last->maxPositionError > settings.linearTolerance;
        const bool angleFailed = last && last->maxAngleErrorDeg > settings.angularToleranceDeg;
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
            const auto measured = valueToString<LengthUnit>( last->maxPositionError );
            const auto allowed = valueToString<LengthUnit>( settings.linearTolerance );
            ImGui::TextColored( errorColor(), _tr( "Measured %s; allowed %s" ), measured.c_str(),
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
                const auto measured = valueToString<AngleUnit>( last->maxAngleErrorDeg * std::numbers::pi / 180. );
                const auto allowed = valueToString<AngleUnit>( angle );
                ImGui::TextColored( errorColor(), _tr( "Measured %s; allowed %s" ), measured.c_str(),
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
        advanced_ = ImGui::CollapsingHeader( _tr( "Advanced" ) );
        if ( advanced_ )
        {
            if ( settings.mode == PrimaryMode::Plane && settings.holeMode == HoleMode::Axes )
            {
                double minimumTilt = settings.minimumObliqueAngleDeg * std::numbers::pi / 180.;
                ImGui::SetNextItemWidth( 150 * scale );
                if ( UI::input<AngleUnit>( _tr( "Minimum oblique tilt" ), minimumTilt, 0., std::numbers::pi / 2 - 1e-6 ) )
                {
                    settings.minimumObliqueAngleDeg = minimumTilt * 180. / std::numbers::pi;
                    changed = true;
                }
                UI::setTooltipIfHovered( _tr( "Both hole axes must tilt this far from their plane normals to determine rotation. "
                                              "Set above fitting noise; independent of Angle tolerance." ) );
            }
            if ( settings.mode == PrimaryMode::Axis )
                changed |= UI::checkbox( _tr( "Make primary axes coaxial" ), &settings.coaxial );
            if ( settings.mode != PrimaryMode::Centers )
            {
                UI::transparentTextWrapped( _tr( "Optional directed rotation reference:" ) );
                if ( ImGui::BeginTable( "Clocking", 2, ImGuiTableFlags_SizingStretchSame ) )
                {
                    ImGui::TableNextColumn();
                    changed |= choose( "##Fixed clocking", selection.fixedClock,
                                       view.clockingFeatures[0], false, true );
                    ImGui::TableNextColumn();
                    changed |= choose( "##Moving clocking", selection.movingClock,
                                       view.clockingFeatures[1], true, true );
                    ImGui::EndTable();
                }
            }
        }
        if ( changed )
        {
            controller_->configure( selection, settings );
            message_.clear();
            error_ = false;
            (void)controller_->viewState();
        }
        UI::separator();
        if ( const auto &result = controller_->result() )
        {
            const auto &r = *result;
            const auto heading = r.acceptable() ? s_tr( "Calculated placement" ) : s_tr( "Outside tolerance" );
            ImGui::TextColored( r.acceptable() ? roleColor( false ) : errorColor(), "%s",
                                heading.c_str() );
            ImGui::TextWrapped( "%s", freeMotionText( r, settings.mode ).c_str() );
            const auto residual = valueToString<LengthUnit>( r.maxPositionError );
            ImGui::Text( _tr( "Max constrained residual: %s" ), residual.c_str() );
            ImGui::SetNextItemOpen( diagnostics_, ImGuiCond_Always );
            diagnostics_ = ImGui::CollapsingHeader( _tr( "Diagnostics" ) );
            if ( diagnostics_ )
            {
                ImGui::Text( _tr( "Rotation set by: %s" ), rotationSourceText( r.rotationSource ).c_str() );
                if ( settings.mode != PrimaryMode::Centers )
                {
                    const auto angle = valueToString<AngleUnit>( r.primaryAngleErrorDeg * std::numbers::pi / 180. );
                    ImGui::Text( _tr( "Primary angle residual: %s" ), angle.c_str() );
                }
                for ( size_t i = 0; i < r.holes.size(); ++i )
                {
                    const auto position = valueToString<LengthUnit>( r.holes[i].position );
                    if ( settings.mode == PrimaryMode::Plane )
                        ImGui::Text( _tr( "Hole %d in plane: %s" ), int( i + 1 ), position.c_str() );
                    else
                        ImGui::Text( _tr( "Locator %d: %s" ), int( i + 1 ), position.c_str() );
                }
                ImGui::PushStyleColor( ImGuiCol_Text, errorColor() );
                for ( const auto &conflict : r.conflicts )
                    drawDiagnostic( conflict, r, settings.mode );
                ImGui::PopStyleColor();
                ImGui::PushStyleColor( ImGuiCol_Text,
                                       ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextDisabled ).getUInt32() );
                for ( const auto &warning : r.warnings )
                    drawDiagnostic( warning, r, settings.mode );
                ImGui::PopStyleColor();
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
            UI::transparentTextWrapped( "%s", _tr( view.setupError ) );
        }
        if ( scrollToResult_ )
        {
            ImGui::SetScrollHereY( 1 );
            scrollToResult_ = false;
        }
    }
    ImGui::EndChild();
    if ( UI::button( _tr( "Apply" ), view.canApply, { 0, actionHeight } ) )
    {
        auto result = controller_->apply();
        error_ = !result;
        message_ = result ? s_tr( "Applied to Floating object. Native Undo is available." ) : s_tr( result.error() );
        scrollToResult_ = true;
    }
    if ( selection.moving && selection.moving->isLocked() )
        UI::setTooltipIfHovered( _tr( "Unlock the Floating object's transform before applying." ) );
    ImGui::EndCustomStatePlugin();
}
MR_REGISTER_RIBBON_ITEM( FeatureAlignmentPlugin )
} // namespace MR
