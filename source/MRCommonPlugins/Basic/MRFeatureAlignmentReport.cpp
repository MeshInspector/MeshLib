#include "MRFeatureAlignmentReport.h"
#include <json/json.h>
namespace MR::FeatureAlignmentTool
{
std::string reportJson( const Controller &controller )
{
    Json::Value report;
    report["schema"] = "meshlib.feature-alignment/1";
    report["preview_displayed"] = controller.previewDisplayed();
    report["preview_current"] = controller.canApply();
    report["restore_available"] = controller.canRestore();
    report["stale_preview"] = controller.stalePreview();
    const auto entries = catalog( controller.root() );
    const auto path = [&]( const ObjectPtr &object )
    {
        for ( const auto &e : entries )
            if ( e.object == object )
                return e.path;
        return std::string();
    };
    const auto &selection = controller.selection();
    report["moving"] = path( selection.moving );
    report["fixed"] = path( selection.fixed );
    report["moving_primary"] = path( selection.movingPrimary );
    report["fixed_primary"] = path( selection.fixedPrimary );
    report["locators"] = Json::Value( Json::arrayValue );
    for ( size_t i = 0; i < selection.fixedLocators.size(); ++i )
        if ( selection.fixedLocators[i] || selection.movingLocators[i] )
        {
            Json::Value pair;
            pair["fixed"] = path( selection.fixedLocators[i] );
            pair["moving"] = path( selection.movingLocators[i] );
            report["locators"].append( pair );
        }
    report["clocking"]["fixed"] = path( selection.fixedClock );
    report["clocking"]["moving"] = path( selection.movingClock );
    report["conflicts"] = Json::Value( Json::arrayValue );
    report["warnings"] = Json::Value( Json::arrayValue );
    const auto &settings = controller.settings();
    report["normal_relation"] = int( settings.normals );
    report["mode"] = settings.mode == PrimaryMode::Plane  ? "plane"
                     : settings.mode == PrimaryMode::Axis ? "axis"
                                                          : "centers";
    report["role_rule"] = "higher_scene_tree_part_fixed";
    report["axis_direction"] = int( settings.axisDirection );
    report["coaxial"] = settings.coaxial;
    report["match_primary_centers"] = settings.matchAxialCenters;
    report["hole_mode"] = settings.holeMode == HoleMode::Centers ? "centers" : "axes";
    report["balance_holes"] = settings.balanceHoles;
    report["offset"] = settings.offset;
    report["linear_tolerance"] = settings.linearTolerance;
    report["angular_tolerance_deg"] = settings.angularToleranceDeg;
    if ( const auto &result = controller.result() )
    {
        const auto &r = *result;
        report["acceptable"] = r.acceptable();
        report["remaining_dof"] = r.remainingDof;
        report["rotation_source"] = r.rotationSource;
        auto &m = report["metrics"];
        m["primary_position_error"] = r.primaryPositionError;
        if ( r.mode != PrimaryMode::Centers )
            m["primary_angle_error_deg"] = r.primaryAngleErrorDeg;
        if ( r.mode == PrimaryMode::Plane )
        {
            m["signed_plane_gap"] = r.signedPlaneGap;
            m["plane_offset_error"] = r.planeOffsetError;
            m["normal_error_deg"] = r.normalErrorDeg;
        }
        report["remaining_motion"] = r.freeMotion;
        m["maximum_position_error"] = r.maximumPositionError();
        m["maximum_angle_error_deg"] = r.maximumAngleError();
        m["position_within_tolerance"] = r.maximumPositionError() <= settings.linearTolerance;
        m["angle_within_tolerance"] = r.maximumAngleError() <= settings.angularToleranceDeg;
        m["locator_position_basis"] = r.mode == PrimaryMode::Plane ? "mating_plane" : "spatial";
        m["locator_axial_difference_is_constrained"] = r.mode != PrimaryMode::Plane;
        m["determinant"] = r.determinant;
        m["orthogonality_error"] = r.orthogonalityError;
        for ( const auto &h : r.holes )
        {
            Json::Value value;
            value["position_error"] = h.position;
            if ( h.hasAxis )
            {
                value["in_plane_error"] = h.inPlane;
                value["axial_error"] = h.axial;
                value["axis_angle_deg"] = h.axisAngleDeg;
                value["axis_angle_is_constrained"] = h.angleConstrained;
                value["radius_difference"] = h.radiusDifference;
            }
            value["raw_center_distance"] = h.rawCenterDistance;
            m["locators"].append( value );
        }
        if ( r.clockingErrorDeg )
            m["clocking_error_deg"] = *r.clockingErrorDeg;
        for ( const auto &message : r.conflicts )
            report["conflicts"].append( message );
        for ( const auto &message : r.warnings )
            report["warnings"].append( message );
        for ( int i = 0; i < 3; ++i )
        {
            Json::Value row;
            for ( int j = 0; j < 3; ++j )
                row.append( r.delta.A[i][j] );
            report["world_delta"]["rotation"].append( row );
            report["world_delta"]["translation"].append( r.delta.b[i] );
        }
    }
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    return Json::writeString( builder, report );
}
} // namespace MR::FeatureAlignmentTool
