#include "MRICPDialog.h"
#include "MRRibbonButtonDrawer.h"
#include "ImGuiHelpers.h"
#include "MRUIStyle.h"
#include "MRRibbonConstants.h"
#include <MRMesh/MRStringConvert.h>
#include "imgui.h"
#include "MRI18n.h"

namespace MR
{

namespace ICPDialog
{

ChangeFlags drawPairsFormingSettings(
    ICPProperties& params, float& samplingVoxelSize,
    float objectSize, bool* expanded, SampleTarget * sampleTarget, bool rigidModeOnly, bool drawCollapsingHeader,
    bool * registerWholeObjects, float * descriptorSpacing )
{
    if ( drawCollapsingHeader )
    {
        bool e = RibbonButtonDrawer::CustomCollapsingHeader( _tr("Pairs Forming Settings") );
        if ( expanded )
            *expanded = e;
        if ( !e )
            return ChangeFlags::NoChanges;
    }

    ChangeFlags flags = ChangeFlags::NoChanges;
    ImGui::PushItemWidth( 100.f * UI::scale() );

    if ( registerWholeObjects )
    {
        if ( UI::checkbox( _tr("Register Whole Objects"), registerWholeObjects ) )
            flags |= ChangeFlags::ParamsModified | ChangeFlags::PointPairsChanged;
        UI::setTooltipIfHovered( _tr("If selected then absolutely all point pairs between two objects will be considered during registration without filtering.") );
    }

    // pairs generation settings
    if ( !registerWholeObjects || !*registerWholeObjects )
    {
        float threshold = std::acos( std::clamp( params.cosThreshold, -1.f, 1.f ) );
        bool changed = UI::drag<AngleUnit>( _tr("Normals Max Angle"), threshold, PI_F / 180, 0.f, PI_F );
        if ( changed )
        {
            params.cosThreshold = std::cos( threshold );
            flags |= ChangeFlags::ParamsChanging;
        }
        UI::setTooltipIfHovered( _tr("A points pair is used only if the angle between the normals at the points is not larger than this value.") );
        if ( ImGui::IsItemDeactivatedAfterEdit() || ( changed && !ImGui::IsItemActive() ) )
            flags |= ChangeFlags::ParamsModified | ChangeFlags::PointPairsChanged;

        float distLimit = sqrt( params.distThresholdSq );
        changed = UI::drag<LengthUnit>( _tr("Max Distance"), distLimit,
                    distLimit > 0 ? distLimit / 100 : 1e-5f,
                    0.f, 10 * objectSize ); // 10 * objectSize to allow initial distances larger than object size
        if ( changed )
            flags |= ChangeFlags::ParamsChanging;
        params.distThresholdSq = distLimit * distLimit;
        UI::setTooltipIfHovered( _tr("A points pair is used only if the distance between two points is lower than this value.") );
        if ( ImGui::IsItemDeactivatedAfterEdit() || ( changed && !ImGui::IsItemActive() ) )
            flags |= ChangeFlags::ParamsModified | ChangeFlags::PointPairsChanged;

        changed = UI::drag<NoUnit>( _tr("Far Distance Factor"),
            params.farDistFactor, 0.001f, 1.f, 10.f );
        if ( changed )
            flags |= ChangeFlags::ParamsChanging;
        UI::setTooltipIfHovered( _tr("Points pair will be counted only if distance between points is lower than root-mean-square distance times this factor.") );
        if ( ImGui::IsItemDeactivatedAfterEdit() || ( changed && !ImGui::IsItemActive() ) )
            flags |= ChangeFlags::ParamsModified | ChangeFlags::PointPairsChanged;

        if ( UI::checkbox( _tr("Mutual Closest Points in Each Pair"), &params.mutualClosest ) )
            flags |= ChangeFlags::ParamsModified | ChangeFlags::PointPairsChanged;
        UI::setTooltipIfHovered( _tr("A pair of points is formed only if both points in the pair are mutually closest (reciprocity test passed).") );
    }

    const bool samplingChanged = UI::drag<LengthUnit>( _tr("Sampling Spacing"), samplingVoxelSize,
            samplingVoxelSize > 0 ? samplingVoxelSize / 100 : 1e-5f,
            objectSize * 1e-4f, objectSize * 0.5f, { .precision = 6 } );
    if ( samplingChanged )
        flags |= ChangeFlags::ParamsChanging;
    UI::setTooltipIfHovered( _tr("This value determines sampling voxel size to select only one point in each voxel.") );
    if ( ImGui::IsItemDeactivatedAfterEdit() || ( samplingChanged && !ImGui::IsItemActive() ) )
        flags |= ChangeFlags::ParamsModified | ChangeFlags::PointPairsChanged | ChangeFlags::SamplingChanged;

    if ( descriptorSpacing )
    {
        UI::drag<LengthUnit>( _tr("Descriptor Spacing"), *descriptorSpacing,
            *descriptorSpacing > 0 ? *descriptorSpacing / 100 : 1e-5f, objectSize * 1e-4f, objectSize * 0.5f, { .precision = 6 } );
        UI::setTooltipIfHovered( _tr("Auto Initial matches the objects by shape descriptors computed on samples with this spacing. "
            "Smaller spacing captures smaller details, but takes longer.") );
    }

    ImGui::PopItemWidth();

    if ( sampleTarget )
    {
        if ( UI::combo( _tr("Sample Target"), ( int* )sampleTarget,
            {
                "Both Objects",
                "Floating Only",
                "Reference Only"
            } ) )
            flags |= ChangeFlags::ParamsModified | ChangeFlags::PointPairsChanged | ChangeFlags::SamplingChanged;
        UI::setTooltipIfHovered( _tr("Select what object will be samples to form origins of point pairs.") );
    }

    if ( UI::combo( _tr("Method"), ( int* )( &params.method ), { _tr("Combined"), _tr("Point-to-Point"), _tr("Point-to-Plane") } ) )
        flags |= ChangeFlags::ParamsChanging | ChangeFlags::ParamsModified;
    UI::setTooltipIfHovered( _tr("Select target function to minimize.") );

    if ( !rigidModeOnly )
    {
        if ( UI::combo( _tr("Transformation"), ( int* ) ( &params.icpMode ),
                        {
                            "Rigid + Scale",
                            "Rigid Body",
                            "Orthogonal Axis Rotation",
                            "Fixed Axis Rotation",
                            "Translation Only"
                        } ) )
            flags |= ChangeFlags::ParamsChanging | ChangeFlags::ParamsModified;
        UI::setTooltipIfHovered( _tr("Select what kind of transformations are searched for.") );
    }
    else
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, UI::scale() * StyleConsts::CustomCombo::framePadding );
        UI::inputTextCenteredReadOnly( _tr("Transformation"), "Rigid Body", ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize( "Transformation" ).x );
        ImGui::PopStyleVar();
        if ( params.icpMode != ICPMode::AnyRigidXf )
        {
            params.icpMode = ICPMode::AnyRigidXf;
            flags |= ChangeFlags::ParamsChanging | ChangeFlags::ParamsModified;
        }
    }

    switch ( params.icpMode )
    {
    case ICPMode::OrthogonalAxis:
        if ( UI::drag<NoUnit>( _tr("Fixed rotation axis"), params.fixedRotationAxis, 0.1f, -10.f, 10.f ) )
            flags |= ChangeFlags::ParamsChanging | ChangeFlags::ParamsModified;
        break;
    case ICPMode::FixedAxis:
        if ( UI::drag<NoUnit>( _tr("Allowed rotation axis"), params.fixedRotationAxis, 0.1f, -10.f, 10.f ) )
            flags |= ChangeFlags::ParamsChanging | ChangeFlags::ParamsModified;
        break;
    default:
        break;
    }
    return flags;
}

float defaultDescriptorSpacing( float objectSize )
{
    return roundToPrecision( objectSize * 0.005f, 3 );
}

ChangeFlags drawStopConditions( ICPProperties& params, bool* expanded, bool drawCollapsingHeader )
{
    ChangeFlags flags = ChangeFlags::NoChanges;
    if ( drawCollapsingHeader )
    {
        bool e = RibbonButtonDrawer::CustomCollapsingHeader( _tr("Stop Conditions") );
        if ( expanded )
            *expanded = e;
        if ( !e )
            return ChangeFlags::NoChanges;
    }

    ImGui::PushItemWidth( 100.f * UI::scale() );
    if ( UI::drag<NoUnit>( _tr("Max Iterations"), params.iterLimit, 1, 1, 1000 ) )
        flags |= ChangeFlags::ParamsChanging | ChangeFlags::ParamsModified;
    UI::setTooltipIfHovered( _tr("Maximal possible number of ICP iterations.") );

    if ( UI::drag<NoUnit>( _tr("No Improvement Iterations"), params.badIterStopCount, 1, 1, 100 ) )
        flags |= ChangeFlags::ParamsChanging | ChangeFlags::ParamsModified;
    UI::setTooltipIfHovered( _tr("Stop ICP iterations if during given number of last iterations there was no improvement.") );

    if ( UI::drag<LengthUnit>( _tr("Deviation Reached"), params.exitVal, 1e-7f, 0.f, 10.f ) )
        flags |= ChangeFlags::ParamsChanging | ChangeFlags::ParamsModified;
    UI::setTooltipIfHovered( _tr("Algorithm target root-mean-square distance. "
        "As soon as it is reached, the algorithm stops.") );

    ImGui::PopItemWidth();
    return flags;
}

void drawInformation( size_t numSamples, size_t numPairs, float meanSqDistToPoint, float meanSqDistToPlane, bool* expanded )
{
    bool e = RibbonButtonDrawer::CustomCollapsingHeader( _tr("Information") );
    if ( expanded )
        *expanded = e;
    if ( !e )
        return;

    ImGui::Text( "%s: %zu", _tr("Samples"), numSamples );
    ImGui::Text( "%s: %zu", _tr("Active point pairs"), numPairs );
    if ( numPairs > 0 )
    {
        ImGui::Text( "%s: %g", _tr("RMS point-to-point distance"), meanSqDistToPoint );
        ImGui::Text( "%s: %g", _tr("RMS point-to-plane distance"), meanSqDistToPlane );
    }
}

void drawResult( std::string resultString )
{
    UI::transparentText( "%s", resultString.c_str() );
}

}
}
