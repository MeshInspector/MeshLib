#include <MRMesh/MRFeatureAlignment.h>
#include <MRMesh/MRPlaneObject.h>
#include <MRMesh/MRCircleObject.h>
#include <MRMesh/MRCylinderObject.h>
#include <MRMesh/MRPointObject.h>
#include <MRMesh/MRSphereObject.h>
#include <MRMesh/MRLineObject.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

namespace MR
{
namespace
{
using namespace FeatureAlignment;

void check( bool good, const char *message )
{
    if ( !good )
        throw std::runtime_error( message );
}
Constraints base()
{
    Constraints c;
    c.settings.normals = NormalRelation::Same;
    c.settings.holeMode = HoleMode::Axes;
    c.settings.balanceHoles = false;
    return c;
}
Hole h( Vector3d p, Vector3d axis = { 0, 0, 1 } )
{
    Hole r;
    r.center = p;
    r.axis = axis;
    return r;
}
Solution required( const Constraints &c )
{
    auto r = solve( c );
    if ( !r )
        throw std::runtime_error( r.error() );
    return *r;
}
void near( Vector3d a, Vector3d b, double eps = 1e-8 )
{
    check( ( a - b ).length() < eps, "position/direction mismatch" );
}
} // namespace

// scan tolerance defaults are editable starting values.
TEST( MRMesh, FeatureAlignmentScanToleranceDefaultsAreEditableStartingValues )
{
    Settings s;
    check( s.linearTolerance == 1. && s.angularToleranceDeg == 5., "scan defaults" );
    s.linearTolerance = .2;
    s.angularToleranceDeg = .5;
    auto c = base();
    c.settings = s;
    check( required( c ).acceptable(), "custom tolerances accepted" );
}

// plane-only preserves tangential position.
TEST( MRMesh, FeatureAlignmentPlaneOnlyPreservesTangentialPosition )
{
    auto c = base();
    c.movingPlane.center = { 8, 3, 20 };
    c.fixedPlane.center = { 100, -50, 2 };
    auto r = required( c );
    near( r.delta( c.movingPlane.center ), { 8, 3, 2 } );
    check( r.remainingDof == 3, "DOF" );
}

// tilted plane center convention.
TEST( MRMesh, FeatureAlignmentTiltedPlaneCenterConvention )
{
    auto c = base();
    c.movingPlane = { { 50, 20, 30 }, { 1, 0, 0 } };
    c.fixedPlane.center = { 0, 0, 4 };
    near( required( c ).delta( c.movingPlane.center ), { 50, 20, 4 } );
}

// opposing normals and signed gap.
TEST( MRMesh, FeatureAlignmentOpposingNormalsAndSignedGap )
{
    auto c = base();
    c.settings.normals = NormalRelation::Opposing;
    c.settings.offset = 2;
    c.fixedPlane.center = { 0, 0, 7 };
    auto r = required( c );
    near( r.delta.A * Vector3d{ 0, 0, 1 }, { 0, 0, -1 } );
    near( r.delta( { 0, 0, 0 } ), { 0, 0, 9 } );
    check( std::abs( r.signedPlaneGap - 2 ) < 1e-9, "gap" );
}

// one hole leaves rotation.
TEST( MRMesh, FeatureAlignmentOneHoleLeavesRotation )
{
    auto c = base();
    c.holes = { { h( { 3, 4, 0 } ), h( { 20, 30, 0 } ) } };
    auto r = required( c );
    near( r.delta( { 3, 4, 0 } ), { 20, 30, 0 } );
    check( r.remainingDof == 1, "DOF" );
}

// second hole sets rotation.
TEST( MRMesh, FeatureAlignmentSecondHoleSetsRotation )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 10, 20, 0 } ) }, { h( { 10, 0, 0 } ), h( { 10, 30, 0 } ) } };
    auto r = required( c );
    near( r.delta( { 10, 0, 0 } ), { 10, 30, 0 } );
    check( r.remainingDof == 0 && r.acceptable(), "constraints" );
}

// mismatch never scales.
TEST( MRMesh, FeatureAlignmentMismatchNeverScales )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 } ) }, { h( { 10, 0, 0 } ), h( { 0, 11, 0 } ) } };
    auto r = required( c );
    check( !r.acceptable() && std::abs( r.holes[1].position - 1 ) < 1e-8, "mismatch" );
    check( isRigid( r.delta ), "rigid" );
}

// cylinder axis ignores axial display center.
TEST( MRMesh, FeatureAlignmentCylinderAxisIgnoresAxialDisplayCenter )
{
    auto c = base();
    c.holes = { { h( { 1, 2, 100 } ), h( { 3, 4, -100 } ) } };
    auto r = required( c );
    check( r.acceptable() && r.maximumPositionError() < 1e-9, "axis center" );
}

// plane center locators preserve seating instead of matching feature depths.
TEST( MRMesh, FeatureAlignmentPlaneCenterLocatorsPreserveSeatingInsteadOfMatchingFeatureDepths )
{
    auto c = base();
    c.settings.holeMode = HoleMode::Centers;
    c.holes = { { h( { 1, 2, 5 } ), h( { 3, 4, 0 } ) } };
    auto r = required( c );
    check( r.acceptable() && r.maximumPositionError() < 1e-9, "depth is not a plane locator constraint" );
    check( std::abs( r.holes[0].axial - 5 ) < 1e-9, "retain depth diagnostic" );
    check( std::abs( r.signedPlaneGap ) < 1e-9, "plane seating" );
}

// plane center depths do not become constraints at signed gaps.
TEST( MRMesh, FeatureAlignmentPlaneCenterDepthsDoNotBecomeConstraintsAtSignedGaps )
{
    for ( auto relation : { NormalRelation::Same, NormalRelation::Opposing } )
        for ( double gap : { -10., 0., 10. } )
        {
            auto c = base();
            c.settings.holeMode = HoleMode::Centers;
            c.settings.normals = relation;
            c.settings.offset = gap;
            c.holes = { { h( { 0, 0, 7 } ), h( { 0, 0, -9 } ) }, { h( { 10, 0, -3 } ), h( { 10, 0, 4 } ) } };
            auto r = required( c );
            check( r.acceptable() && r.maximumPositionError() < 1e-8, "projected hole centers" );
            check( std::abs( r.signedPlaneGap - gap ) < 1e-9 && isRigid( r.delta ), "rigid plane priority" );
        }
}

// realistic two-hole mismatch rejects only the incompatible projected locator.
TEST( MRMesh, FeatureAlignmentRealisticTwoHoleMismatchRejectsOnlyTheIncompatibleProjectedLocator )
{
    auto c = base();
    c.settings.holeMode = HoleMode::Centers;
    c.settings.linearTolerance = .5;
    c.holes = { { h( { 0, 0, -8.9457646 } ), h( { 0, 0, 0 } ) },
                { h( { 10.708838, 0, 2.0678686 } ), h( { 10, 0, 0 } ) } };
    auto r = required( c );
    check( !r.acceptable() && r.conflicts.size() == 1, "only the second hole conflicts" );
    check( r.holes[0].position < 1e-8 && std::abs( r.holes[1].position - .708838 ) < 1e-8,
           "controlled position errors" );
    check( std::abs( r.maximumPositionError() - .708838 ) < 1e-8, "maximum controlled residual" );
}

// explicit balanced hole fit shares spacing error without changing plane or scale.
TEST( MRMesh, FeatureAlignmentExplicitBalancedHoleFitSharesSpacingErrorWithoutChangingPlaneOrScale )
{
    auto c = base();
    c.settings.holeMode = HoleMode::Centers;
    c.settings.balanceHoles = true;
    c.settings.linearTolerance = .5;
    c.settings.offset = 10;
    c.holes = { { h( { 0, 0, -8.9457646 } ), h( { 0, 0, 0 } ) },
                { h( { 10.708838, 0, 2.0678686 } ), h( { 10, 0, 0 } ) } };
    auto r = required( c );
    check( r.acceptable(), "both projected centers can meet tolerance" );
    check( std::abs( r.holes[0].position - .354419 ) < 1e-8 && std::abs( r.holes[1].position - .354419 ) < 1e-8,
           "symmetric residuals" );
    check( std::abs( r.signedPlaneGap - 10 ) < 1e-9 && isRigid( r.delta ), "plane and rigid geometry" );
}

// balanced fit still rejects incompatible spacing.
TEST( MRMesh, FeatureAlignmentBalancedFitStillRejectsIncompatibleSpacing )
{
    auto c = base();
    c.settings.holeMode = HoleMode::Centers;
    c.settings.balanceHoles = true;
    c.settings.linearTolerance = .5;
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 } ) }, { h( { 12, 0, 3 } ), h( { 10, 0, -4 } ) } };
    auto r = required( c );
    check( !r.acceptable() && std::abs( r.maximumPositionError() - 1 ) < 1e-8, "honest balanced residual" );
    check( isRigid( r.delta ), "no scale" );
}

// plane priority holds for single and paired mixed hole features.
TEST( MRMesh, FeatureAlignmentPlanePriorityHoldsForSingleAndPairedMixedHoleFeatures )
{
    const auto turn = Matrix3d::rotation( Vector3d{ 1, 2, 3 }, .61 );
    const AffineXf3d wanted{ Matrix3d::rotation( Vector3d{ 2, -1, 4 }, .83 ), { 70, -35, 18 } };
    const auto inverse = wanted.inverse();
    const auto n = turn * Vector3d{ 0, 0, 1 };
    const Vector3d origin{ 100, -40, 20 };
    for ( int count : { 1, 2 } )
        for ( auto kindA : { HoleKind::Circle, HoleKind::Cylinder } )
            for ( auto kindB : { HoleKind::Circle, HoleKind::Cylinder } )
                for ( auto mode : { HoleMode::Centers, HoleMode::Axes } )
                    for ( auto relation : { NormalRelation::Same, NormalRelation::Opposing } )
                        for ( double gap : { -10., 0., 10. } )
                        {
                            auto c = base();
                            c.settings.holeMode = mode;
                            c.settings.normals = relation;
                            c.settings.offset = gap;
                            c.fixedPlane = { origin, n };
                            c.movingPlane = { inverse( origin + gap * n ),
                                              inverse.A * ( double( int( relation ) ) * n ) };
                            for ( int i = 0; i < count; ++i )
                            {
                                const auto center = origin + turn * Vector3d{ i ? 25. : -30., 8., 0. };
                                auto moving = h( inverse( center + ( gap + ( i ? 4. : -7. ) ) * n ), inverse.A * n );
                                auto fixed = h( center + ( i ? -2. : 9. ) * n, n );
                                moving.kind = kindA;
                                fixed.kind = kindB;
                                c.holes.push_back( { moving, fixed } );
                            }
                            auto r = required( c );
                            check( r.acceptable() && r.maximumPositionError() < 1e-7, "mixed projected/axis locators" );
                            check( std::abs( r.signedPlaneGap - gap ) < 1e-7 && r.normalErrorDeg < 1e-5,
                                   "locked plane" );
                            check( r.remainingDof == ( count == 1 ? 1 : 0 ) && isRigid( r.delta ),
                                   "honest freedom and rigid placement" );
                        }
}

// oblique axis sets rotation.
TEST( MRMesh, FeatureAlignmentObliqueAxisSetsRotation )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 }, { 1, 0, 1 } ), h( { 4, 7, 0 }, { 0, 1, 1 } ) } };
    auto r = required( c );
    check( r.acceptable() && r.remainingDof == 0, "oblique" );
    near( r.delta.A * Vector3d{ 1, 0, 1 }, { 0, 1, 1 } );
}

// near-normal axis does not invent rotation.
TEST( MRMesh, FeatureAlignmentNearNormalAxisDoesNotInventRotation )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 }, { .0001, 0, 1 } ), h( { 0, 0, 0 }, { 0, .0001, 1 } ) } };
    check( required( c ).remainingDof == 1, "DOF" );
}

// axes are unoriented.
TEST( MRMesh, FeatureAlignmentAxesAreUnoriented )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 }, { 1, 0, 1 } ), h( { 0, 0, 0 }, { 0, -1, -1 } ) } };
    check( required( c ).acceptable(), "unoriented" );
}

// incompatible tilt blocks apply.
TEST( MRMesh, FeatureAlignmentIncompatibleTiltBlocksApply )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 }, { 1, 0, 1 } ) } };
    check( !required( c ).acceptable(), "tilt conflict" );
}

// directed clocking with one hole.
TEST( MRMesh, FeatureAlignmentDirectedClockingWithOneHole )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 4, 7, 0 } ) } };
    c.clocking = { { { 1, 0, 0 }, { 0, -1, 0 } } };
    auto r = required( c );
    near( r.delta.A * Vector3d{ 1, 0, 0 }, { 0, -1, 0 } );
    check( r.remainingDof == 0, "DOF" );
}

// directed reference conflict.
TEST( MRMesh, FeatureAlignmentDirectedReferenceConflict )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 } ) }, { h( { 10, 0, 0 } ), h( { 0, 10, 0 } ) } };
    c.clocking = { { { 1, 0, 0 }, { 1, 0, 0 } } };
    check( !required( c ).acceptable(), "clock conflict" );
}

// zero projected baseline rejected.
TEST( MRMesh, FeatureAlignmentZeroProjectedBaselineRejected )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 } ) }, { h( { 0, 0, 5 } ), h( { 10, 0, 0 } ) } };
    check( !solve( c ), "degenerate baseline" );
}

// parallel axis rejected.
TEST( MRMesh, FeatureAlignmentParallelAxisRejected )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 1 }, { 1, 0, 0 } ), h( { 0, 0, 0 } ) } };
    check( !solve( c ), "parallel axis" );
}

// nonfinite geometry rejected.
TEST( MRMesh, FeatureAlignmentNonfiniteGeometryRejected )
{
    auto c = base();
    c.movingPlane.center.x = std::numeric_limits<double>::quiet_NaN();
    check( !solve( c ), "nonfinite" );
}

// zero normal rejected.
TEST( MRMesh, FeatureAlignmentZeroNormalRejected )
{
    auto c = base();
    c.movingPlane.normal = {};
    check( !solve( c ), "normal" );
}

// negative radius rejected.
TEST( MRMesh, FeatureAlignmentNegativeRadiusRejected )
{
    auto c = base();
    c.holes = { { h( {} ), h( {} ) } };
    c.holes[0].moving.radius = -1;
    check( !solve( c ), "radius" );
}

// invalid tolerances rejected.
TEST( MRMesh, FeatureAlignmentInvalidTolerancesRejected )
{
    auto c = base();
    c.settings.linearTolerance = -1;
    check( !solve( c ), "negative tolerance" );
    c = base();
    c.settings.angularToleranceDeg = 90;
    check( !solve( c ), "angle tolerance" );
}

// rigid validation rejects mirror and scale.
TEST( MRMesh, FeatureAlignmentRigidValidationRejectsMirrorAndScale )
{
    AffineXf3d x;
    check( isRigid( x ), "identity" );
    x.A.x.x = 2;
    check( !isRigid( x ), "scale" );
    x.A.x.x = -1;
    check( !isRigid( x ), "reflection" );
}

// measure does not hide a supplied position error.
TEST( MRMesh, FeatureAlignmentMeasureDoesNotHideASuppliedPositionError )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.holes = { { h( {} ), h( {} ) } };
    auto r = measure( c, AffineXf3d::translation( { 1, 0, 0 } ) );
    check( r && !r->acceptable() && std::abs( r->holes[0].position - 1 ) < 1e-10, "measurement" );
}

// 150 randomized rigid recoveries.
TEST( MRMesh, FeatureAlignment150RandomizedRigidRecoveries )
{
    std::mt19937 random( 150 );
    std::uniform_real_distribution<double> d( -1, 1 );
    for ( int i = 0; i < 150; ++i )
    {
        auto c = base();
        c.settings.normals = i % 2 ? NormalRelation::Same : NormalRelation::Opposing;
        c.settings.holeMode = i % 3 ? HoleMode::Centers : HoleMode::Axes;
        c.settings.offset = d( random ) * 3;
        auto r = Matrix3d::rotation( Vector3d{ d( random ), d( random ), d( random ) }, d( random ) * 3 );
        AffineXf3d desired{ r, { d( random ) * 100, d( random ) * 100, d( random ) * 100 } };
        c.movingPlane = { { d( random ) * 40, d( random ) * 40, d( random ) * 40 },
                          Vector3d{ d( random ), d( random ), d( random ) }.normalized() };
        c.fixedPlane.normal = double( int( c.settings.normals ) ) * ( r * c.movingPlane.normal );
        c.fixedPlane.center = desired( c.movingPlane.center ) - c.settings.offset * c.fixedPlane.normal;
        auto tangent = cross( c.movingPlane.normal, c.movingPlane.normal.furthestBasisVector() ).normalized();
        for ( double step : { -30., 30. } )
        {
            auto source = h( c.movingPlane.center + step * tangent, c.movingPlane.normal );
            auto target = h( desired( source.center ) - c.settings.offset * c.fixedPlane.normal, r * source.axis );
            c.holes.push_back( { source, target } );
        }
        auto solution = required( c );
        check( solution.acceptable() && solution.remainingDof == 0, "random constraints" );
        near( solution.delta( { 4, 5, 6 } ), desired( { 4, 5, 6 } ), 1e-8 );
        check( isRigid( solution.delta ), "random rigidity" );
    }
}

// axis centers align without a plane.
TEST( MRMesh, FeatureAlignmentAxisCentersAlignWithoutAPlane )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.movingAxis = { { 3, 4, 9 }, { 1, 0, 0 } };
    c.fixedAxis = { { 10, 20, 0 }, { 0, 0, 1 } };
    auto r = required( c );
    check( r.acceptable() && r.remainingDof == 1, "axis DOF" );
    near( r.delta( c.movingAxis.center ), c.fixedAxis.center );
}

// coaxial axes retain slide and spin.
TEST( MRMesh, FeatureAlignmentCoaxialAxesRetainSlideAndSpin )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.movingAxis.center = { 3, 4, 9 };
    c.fixedAxis.center = { 20, 30, 0 };
    auto r = required( c );
    check( r.remainingDof == 2 && r.acceptable(), "axis free motions" );
    near( r.delta( c.movingAxis.center ), { 20, 30, 9 } );
}

// axis direction only retains four freedoms.
TEST( MRMesh, FeatureAlignmentAxisDirectionOnlyRetainsFourFreedoms )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.settings.coaxial = false;
    c.movingAxis = { { 3, 4, 9 }, { 1, 0, 0 } };
    auto r = required( c );
    check( r.remainingDof == 4, "direction DOF" );
    near( r.delta( c.movingAxis.center ), c.movingAxis.center );
}

// off-axis center sets cylinder rotation.
TEST( MRMesh, FeatureAlignmentOffAxisCenterSetsCylinderRotation )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.fixedAxis.center = { 20, 30, 0 };
    c.centers = { { { 10, 0, 5 }, { 20, 40, 5 } } };
    auto r = required( c );
    check( r.acceptable() && r.remainingDof == 0, "locator" );
    near( r.delta( { 10, 0, 5 } ), { 20, 40, 5 } );
}

// on-axis center sets slide but not spin.
TEST( MRMesh, FeatureAlignmentOnAxisCenterSetsSlideButNotSpin )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.fixedAxis.center = { 3, 4, 0 };
    c.centers = { { { 0, 0, 7 }, { 3, 4, 20 } } };
    auto r = required( c );
    check( r.acceptable() && r.remainingDof == 1, "on-axis locator" );
    near( r.delta( { 0, 0, 7 } ), { 3, 4, 20 } );
}

// inconsistent radial locator is a conflict.
TEST( MRMesh, FeatureAlignmentInconsistentRadialLocatorIsAConflict )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 5, 0, 0 }, { 6, 0, 0 } } };
    auto r = required( c );
    check( !r.acceptable() && isRigid( r.delta ), "radial mismatch" );
}

// primary axial center has priority.
TEST( MRMesh, FeatureAlignmentPrimaryAxialCenterHasPriority )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 5, 0, 4 }, { 5, 0, 6 } } };
    auto r = required( c );
    check( !r.acceptable() && r.primaryPositionError < 1e-8, "axial priority" );
}

// signed axial offset.
TEST( MRMesh, FeatureAlignmentSignedAxialOffset )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.offset = 2;
    auto r = required( c );
    near( r.delta( {} ), { 0, 0, 2 } );
    check( r.acceptable(), "axis gap" );
}

// directed axis clocking leaves slide.
TEST( MRMesh, FeatureAlignmentDirectedAxisClockingLeavesSlide )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.clocking = { { { 1, 0, 0 }, { 0, 1, 0 } } };
    auto r = required( c );
    check( r.remainingDof == 1 && r.acceptable(), "axis clocking" );
    near( r.delta.A * Vector3d{ 1, 0, 0 }, { 0, 1, 0 } );
}

// direction-only plus one center leaves spin.
TEST( MRMesh, FeatureAlignmentDirectionOnlyPlusOneCenterLeavesSpin )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.coaxial = false;
    c.settings.matchAxialCenters = false;
    c.centers = { { { 10, 0, 0 }, { 20, 30, 4 } } };
    auto r = required( c );
    check( r.remainingDof == 1 && r.acceptable(), "direction and center" );
}

// direction-only plus two centers sets rotation.
TEST( MRMesh, FeatureAlignmentDirectionOnlyPlusTwoCentersSetsRotation )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.coaxial = false;
    c.settings.matchAxialCenters = false;
    c.centers = { { { 10, 0, 0 }, { 20, 30, 4 } }, { { 20, 0, 0 }, { 20, 40, 4 } } };
    auto r = required( c );
    check( r.remainingDof == 0 && r.acceptable(), "direction and centers" );
}

// unoriented axes choose the sign consistent with locators.
TEST( MRMesh, FeatureAlignmentUnorientedAxesChooseTheSignConsistentWithLocators )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 5, 0, 3 }, { 5, 0, -3 } } };
    auto r = required( c );
    check( r.acceptable(), "axis sign must follow compatible measurements" );
    near( r.delta.A * Vector3d{ 0, 0, 1 }, { 0, 0, -1 } );
}

// zero axis rejected.
TEST( MRMesh, FeatureAlignmentZeroAxisRejected )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.movingAxis.direction = {};
    check( !solve( c ), "zero axis" );
}

// center-only one pair leaves three rotations.
TEST( MRMesh, FeatureAlignmentCenterOnlyOnePairLeavesThreeRotations )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 3, 4, 5 }, { 10, 20, 30 } } };
    auto r = required( c );
    check( r.remainingDof == 3 && r.acceptable(), "center DOF" );
    near( r.delta( { 3, 4, 5 } ), { 10, 20, 30 } );
}

// center-only two pairs leave one rotation.
TEST( MRMesh, FeatureAlignmentCenterOnlyTwoPairsLeaveOneRotation )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 10, 20, 30 } }, { { 10, 0, 0 }, { 10, 30, 30 } } };
    auto r = required( c );
    check( r.remainingDof == 1 && r.acceptable(), "two centers" );
}

// center-only three points fully locate.
TEST( MRMesh, FeatureAlignmentCenterOnlyThreePointsFullyLocate )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    AffineXf3d x{ Matrix3d::rotation( { 1, 2, -3 }, .8 ), { 10, 20, 30 } };
    for ( Vector3d p : std::vector<Vector3d>{ { 0, 0, 0 }, { 10, 0, 0 }, { 0, 5, 1 } } )
        c.centers.push_back( { p, x( p ) } );
    auto r = required( c );
    check( r.remainingDof == 0 && r.acceptable(), "three centers" );
    near( r.delta( { 3, 4, 5 } ), x( { 3, 4, 5 } ) );
}

// center-only mismatch never scales.
TEST( MRMesh, FeatureAlignmentCenterOnlyMismatchNeverScales )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } }, { { 10, 0, 0 }, { 0, 11, 0 } } };
    auto r = required( c );
    check( !r.acceptable() && isRigid( r.delta ), "center mismatch" );
}

// collinear centers report remaining rotation.
TEST( MRMesh, FeatureAlignmentCollinearCentersReportRemainingRotation )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } }, { { 10, 0, 0 }, { 0, 10, 0 } }, { { 20, 0, 0 }, { 0, 20, 0 } } };
    auto r = required( c );
    check( r.remainingDof == 1 && r.acceptable(), "collinear DOF" );
}

// duplicate secondary center rejected.
TEST( MRMesh, FeatureAlignmentDuplicateSecondaryCenterRejected )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } }, { { 0, 0, 0 }, { 1, 0, 0 } } };
    check( !solve( c ), "duplicate center" );
}

// center-only offset has no assumed direction.
TEST( MRMesh, FeatureAlignmentCenterOnlyOffsetHasNoAssumedDirection )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.settings.offset = 1;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } } };
    check( !solve( c ), "undefined offset" );
}

// axis locator and clocking conflict.
TEST( MRMesh, FeatureAlignmentAxisLocatorAndClockingConflict )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 5, 0, 0 }, { 0, 5, 0 } } };
    c.clocking = { { { 1, 0, 0 }, { 1, 0, 0 } } };
    check( !required( c ).acceptable(), "clocking conflict" );
}

// invalid primary mode rejected.
TEST( MRMesh, FeatureAlignmentInvalidPrimaryModeRejected )
{
    auto c = base();
    c.settings.mode = PrimaryMode( 99 );
    check( !solve( c ), "mode" );
}

// 150 axis and 150 center rigid recoveries.
TEST( MRMesh, FeatureAlignment150AxisAnd150CenterRigidRecoveries )
{
    std::mt19937 random( 300 );
    std::uniform_real_distribution<double> d( -1, 1 );
    for ( int i = 0; i < 150; ++i )
    {
        AffineXf3d x{ Matrix3d::rotation( Vector3d{ d( random ), d( random ), d( random ) }, d( random ) * 3 ),
                      { d( random ) * 100, d( random ) * 100, d( random ) * 100 } };
        auto c = base();
        c.settings.mode = PrimaryMode::Axis;
        c.settings.axisDirection = AxisDirection::Same;
        c.movingAxis = { { d( random ) * 20, d( random ) * 20, d( random ) * 20 },
                         Vector3d{ d( random ), d( random ), d( random ) }.normalized() };
        c.fixedAxis = { x( c.movingAxis.center ), x.A * c.movingAxis.direction };
        auto off = cross( c.movingAxis.direction, c.movingAxis.direction.furthestBasisVector() ).normalized() * 20;
        auto point = c.movingAxis.center + off + 7 * c.movingAxis.direction;
        c.centers = { { point, x( point ) } };
        auto axis = required( c );
        check( axis.acceptable() && axis.remainingDof == 0, "random axis" );
        near( axis.delta( { 2, 3, 4 } ), x( { 2, 3, 4 } ) );
        c = base();
        c.settings.mode = PrimaryMode::Centers;
        for ( Vector3d p : std::vector<Vector3d>{ { 0, 0, 0 }, { 20, 0, 0 }, { 0, 10, 3 } } )
            c.centers.push_back( { p, x( p ) } );
        auto center = required( c );
        check( center.acceptable() && center.remainingDof == 0, "random center" );
        near( center.delta( { 2, 3, 4 } ), x( { 2, 3, 4 } ) );
    }
}

namespace
{
std::shared_ptr<PlaneObject> fittedPlane( Vector3f center = {}, Vector3f normal = { 0, 0, 1 } )
{
    auto object = std::make_shared<PlaneObject>();
    object->setCenter( center );
    object->setNormal( normal );
    return object;
}
std::shared_ptr<CircleObject> fittedCircle( Vector3f center, Vector3f normal = { 0, 0, 1 } )
{
    auto object = std::make_shared<CircleObject>();
    object->setCenter( center );
    object->setNormal( normal );
    object->setRadius( 2 );
    return object;
}
std::shared_ptr<PointObject> fittedPoint( Vector3f center )
{
    auto object = std::make_shared<PointObject>();
    object->setLocalPoint( center );
    return object;
}
FeatureReferences planeReferences()
{
    FeatureReferences refs;
    refs.primary = { fittedPlane(), fittedPlane() };
    return refs;
}
} // namespace

TEST( MRMesh, FeatureAlignmentFeatureCentersKeepPlaneDepthAndBalanceMismatch )
{
    auto refs = planeReferences();
    refs.locators[0] = { fittedCircle( { 0, 0, -9 } ), fittedCircle( { 0, 0, 4 } ) };
    refs.locators[1] = { fittedCircle( { 12, 0, 3 } ), fittedCircle( { 10, 0, -5 } ) };
    auto settings = base().settings;
    settings.holeMode = HoleMode::Centers;
    settings.offset = -7;
    settings.linearTolerance = .5;
    const auto pinned = solve( refs, settings );
    ASSERT_TRUE( pinned ) << pinned.error();
    EXPECT_NEAR( pinned->signedPlaneGap, -7, 1e-9 );
    EXPECT_NEAR( pinned->holes[0].position, 0, 1e-9 );
    EXPECT_NEAR( pinned->holes[1].position, 2, 1e-9 );
    EXPECT_FALSE( pinned->acceptable() );
    settings.balanceHoles = true;
    const auto balanced = solve( refs, settings );
    ASSERT_TRUE( balanced ) << balanced.error();
    EXPECT_NEAR( balanced->signedPlaneGap, -7, 1e-9 );
    EXPECT_NEAR( balanced->holes[0].position, 1, 1e-9 );
    EXPECT_NEAR( balanced->holes[1].position, 1, 1e-9 );
    EXPECT_EQ( balanced->remainingDof, 0 );
    EXPECT_TRUE( isRigid( balanced->delta ) );
}

TEST( MRMesh, FeatureAlignmentFeatureHoleAxesIgnoreCylinderDisplayCenter )
{
    auto refs = planeReferences();
    auto cylinder = std::make_shared<CylinderObject>();
    cylinder->setCenter( { 3, 4, 120 } );
    cylinder->setRadius( 2 );
    cylinder->setLength( 90 );
    refs.locators[0] = { cylinder, fittedCircle( { 8, 10, -30 } ) };
    const auto settings = base().settings;
    const auto one = solve( refs, settings );
    ASSERT_TRUE( one ) << one.error();
    EXPECT_TRUE( one->acceptable() );
    EXPECT_EQ( one->remainingDof, 1 );
    EXPECT_NEAR( one->signedPlaneGap, 0, 1e-9 );
    EXPECT_NEAR( one->holes[0].inPlane, 0, 1e-9 );
    refs.locators[1] = { fittedCircle( { 13, 4, 75 } ), fittedCircle( { 18, 10, -80 } ) };
    const auto two = solve( refs, settings );
    ASSERT_TRUE( two ) << two.error();
    EXPECT_TRUE( two->acceptable() );
    EXPECT_EQ( two->remainingDof, 0 );
    EXPECT_NEAR( two->delta.A.det(), 1, 1e-12 );
}

TEST( MRMesh, FeatureAlignmentNestedFeatureFramesApplyParentTransformOnce )
{
    auto grand = std::make_shared<Object>();
    auto parent = std::make_shared<Object>();
    grand->setXf( { Matrix3f::rotation( Vector3f::plusZ(), float( PI / 2 ) ), { 100, 20, 30 } } );
    parent->setXf( AffineXf3f::translation( { 10, 0, 5 } ) );
    grand->addChild( parent );
    auto moving = fittedPlane( { 2, 3, 4 }, { 1, 0, 0 } );
    moving->setSizeX( 17 );
    moving->setSizeY( 9 );
    parent->addChild( moving );
    FeatureReferences refs;
    refs.primary = { moving, fittedPlane( { 97, 32, 39 }, { 0, 1, 0 } ) };
    auto settings = base().settings;
    const auto before = moving->xf();
    const auto constraints = readConstraints( refs, settings );
    ASSERT_TRUE( constraints ) << constraints.error();
    EXPECT_NEAR( ( constraints->movingPlane.center - Vector3d{ 97, 32, 39 } ).length(), 0, 5e-6 );
    EXPECT_NEAR( ( constraints->movingPlane.normal - Vector3d{ 0, 1, 0 } ).length(), 0, 5e-6 );
    const auto result = solve( refs, settings );
    ASSERT_TRUE( result ) << result.error();
    EXPECT_TRUE( result->acceptable() );
    EXPECT_NEAR( result->delta.b.length(), 0, 1e-5 );
    EXPECT_EQ( moving->xf(), before );
    EXPECT_EQ( parent->xf().b, ( Vector3f{ 10, 0, 5 } ) );
}

TEST( MRMesh, FeatureAlignmentFeaturesPreserveOpposingPlaneNormalAndGap )
{
    FeatureReferences refs;
    refs.primary = { fittedPlane( { 5, 6, 20 }, { 1, 0, 0 } ), fittedPlane( { 0, 0, 4 } ) };
    Settings settings;
    settings.offset = 3;
    const auto constraints = readConstraints( refs, settings );
    const auto result = solve( refs, settings );
    ASSERT_TRUE( constraints );
    ASSERT_TRUE( result ) << result.error();
    EXPECT_NEAR( result->signedPlaneGap, 3, 1e-9 );
    EXPECT_NEAR( ( result->delta.A * constraints->movingPlane.normal + constraints->fixedPlane.normal ).length(), 0,
                 1e-6 );
    EXPECT_EQ( result->remainingDof, 3 );
}

TEST( MRMesh, FeatureAlignmentFeatureAxisModePreservesPrimaryAndReportsLocatorConflict )
{
    auto moving = std::make_shared<LineObject>();
    auto fixed = std::make_shared<CylinderObject>();
    moving->setCenter( { 1, 2, 3 } );
    moving->setDirection( { 1, 0, 0 } );
    fixed->setCenter( { 10, 20, 30 } );
    FeatureReferences refs;
    refs.primary = { moving, fixed };
    refs.locators[0] = { fittedPoint( { 1, 7, 3 } ), fittedPoint( { 16, 20, 34 } ) };
    auto settings = base().settings;
    settings.mode = PrimaryMode::Axis;
    settings.linearTolerance = .01;
    const auto result = solve( refs, settings );
    ASSERT_TRUE( result ) << result.error();
    EXPECT_FALSE( result->acceptable() );
    EXPECT_NEAR( result->primaryPositionError, 0, 1e-9 );
    EXPECT_NEAR( result->primaryAngleErrorDeg, 0, 1e-6 );
    EXPECT_EQ( result->remainingDof, 0 );
    EXPECT_TRUE( isRigid( result->delta ) );
}

TEST( MRMesh, FeatureAlignmentFeatureCenterModePreservesPrimaryAndCountsRank )
{
    FeatureReferences refs;
    auto sphere = std::make_shared<SphereObject>();
    sphere->setCenter( { 10, 20, 30 } );
    refs.primary = { fittedPoint( {} ), sphere };
    auto settings = base().settings;
    settings.mode = PrimaryMode::Centers;
    auto result = solve( refs, settings );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->remainingDof, 3 );
    refs.locators[0] = { fittedPoint( { 10, 0, 0 } ), fittedCircle( { 10, 30, 30 } ) };
    result = solve( refs, settings );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->remainingDof, 1 );
    refs.locators[1] = { fittedPoint( { 0, 4, 0 } ), fittedPoint( { 6, 20, 30 } ) };
    result = solve( refs, settings );
    ASSERT_TRUE( result );
    EXPECT_TRUE( result->acceptable() );
    EXPECT_EQ( result->remainingDof, 0 );
    EXPECT_NEAR( ( result->delta( { 0, 0, 0 } ) - Vector3d{ 10, 20, 30 } ).length(), 0, 1e-9 );
}

TEST( MRMesh, FeatureAlignmentFeaturesRejectPartialDuplicateAndUnsupportedReferences )
{
    auto refs = planeReferences();
    const auto settings = base().settings;
    refs.locators[0].moving = fittedCircle( {} );
    EXPECT_FALSE( readConstraints( refs, settings ) );
    refs.locators[0] = {};
    refs.locators[1] = { fittedCircle( {} ), fittedCircle( {} ) };
    EXPECT_FALSE( readConstraints( refs, settings ) );
    refs.locators[0] = refs.locators[1];
    EXPECT_FALSE( readConstraints( refs, settings ) );
    refs = planeReferences();
    refs.primary.moving = fittedPoint( {} );
    EXPECT_FALSE( readConstraints( refs, settings ) );
    refs = planeReferences();
    refs.clocking.moving = std::make_shared<LineObject>();
    EXPECT_FALSE( readConstraints( refs, settings ) );
    refs = planeReferences();
    refs.locators[0] = { refs.primary.moving, refs.primary.fixed };
    EXPECT_FALSE( readConstraints( refs, settings ) );
}

TEST( MRMesh, FeatureAlignmentFeaturesRejectNonrigidAncestryAndDegenerateGeometry )
{
    auto refs = planeReferences();
    auto parent = std::make_shared<Object>();
    parent->addChild( std::const_pointer_cast<FeatureObject>( refs.primary.moving ) );
    parent->setXf( { Matrix3f::scale( 2 ), {} } );
    EXPECT_FALSE( readConstraints( refs, base().settings ) );
    parent->setXf( {} );
    auto cylinder = std::make_shared<CylinderObject>();
    cylinder->setLength( 0 );
    refs.locators[0] = { cylinder, fittedCircle( {} ) };
    EXPECT_FALSE( solve( refs, base().settings ) );
    auto plane = fittedPlane();
    plane->setSizeX( 0 );
    refs = planeReferences();
    refs.primary.moving = plane;
    EXPECT_FALSE( solve( refs, base().settings ) );
    refs = planeReferences();
    refs.locators[0] = { fittedCircle( {}, { 1, 0, 0 } ), fittedCircle( {} ) };
    EXPECT_FALSE( solve( refs, base().settings ) );
}

TEST( MRMesh, FeatureAlignmentFeatureClockingUsesDirectedFittedLine )
{
    auto refs = planeReferences();
    auto moving = std::make_shared<LineObject>();
    auto fixed = std::make_shared<LineObject>();
    moving->setDirection( { 1, 0, 0 } );
    fixed->setDirection( { 0, -1, 0 } );
    refs.clocking = { moving, fixed };
    const auto result = solve( refs, base().settings );
    ASSERT_TRUE( result ) << result.error();
    EXPECT_EQ( result->remainingDof, 2 );
    EXPECT_NEAR( ( result->delta.A * Vector3d::plusX() - Vector3d{ 0, -1, 0 } ).length(), 0, 1e-6 );
}

TEST( MRMesh, FeatureAlignmentMeasureReportsAxisConstraintRankWithoutChangingPlacement )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    const AffineXf3d supplied = AffineXf3d::translation( { 2, 3, 4 } );
    const auto expectRank = [&]( int dof )
    {
        const auto result = measure( c, supplied );
        ASSERT_TRUE( result ) << result.error();
        EXPECT_EQ( result->remainingDof, dof );
        EXPECT_FALSE( result->freeMotion.empty() );
        EXPECT_EQ( result->delta, supplied );
    };
    expectRank( 1 );
    c.settings.matchAxialCenters = false;
    expectRank( 2 );
    c.settings.coaxial = false;
    expectRank( 4 );
    c.centers = { { { 10, 0, 0 }, { 20, 30, 4 } } };
    expectRank( 1 );
    c.centers.push_back( { { 20, 0, 0 }, { 20, 40, 4 } } );
    expectRank( 0 );
    c.centers.clear();
    c.settings.coaxial = true;
    c.clocking = { { { 1, 0, 0 }, { 0, 1, 0 } } };
    expectRank( 1 );
}

TEST( MRMesh, FeatureAlignmentMeasureReportsCenterConstraintRankWithoutHidingResiduals )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { {}, {} } };
    const AffineXf3d supplied = AffineXf3d::translation( { 2, 3, 4 } );
    const auto expectRank = [&]( int dof )
    {
        const auto result = measure( c, supplied );
        ASSERT_TRUE( result ) << result.error();
        EXPECT_EQ( result->remainingDof, dof );
        EXPECT_FALSE( result->freeMotion.empty() );
        EXPECT_EQ( result->delta, supplied );
        EXPECT_NEAR( result->primaryPositionError, std::sqrt( 29. ), 1e-12 );
        EXPECT_FALSE( result->acceptable() );
    };
    expectRank( 3 );
    c.centers.push_back( { { 10, 0, 0 }, { 10, 0, 0 } } );
    expectRank( 1 );
    c.centers.push_back( { { 0, 10, 0 }, { 0, 10, 0 } } );
    expectRank( 0 );
    c.centers.back() = { { 20, 0, 0 }, { 20, 0, 0 } };
    expectRank( 1 );
}

TEST( MRMesh, FeatureAlignmentMeasureReportsPlaneConstraintRank )
{
    auto c = base();
    const auto expectRank = [&]( int dof )
    {
        const auto result = measure( c, {} );
        ASSERT_TRUE( result ) << result.error();
        EXPECT_EQ( result->remainingDof, dof );
        EXPECT_FALSE( result->freeMotion.empty() );
    };
    expectRank( 3 );
    c.clocking = { { { 1, 0, 0 }, { 0, 1, 0 } } };
    expectRank( 2 );
    c.clocking.reset();
    c.holes = { { h( {} ), h( {} ) } };
    expectRank( 1 );
    c.holes.push_back( { h( { 10, 0, 0 } ), h( { 0, 10, 0 } ) } );
    expectRank( 0 );
    c.holes.resize( 1 );
    c.holes[0] = { h( {}, { 1, 0, 1 } ), h( {}, { 0, 1, 1 } ) };
    expectRank( 0 );
}

TEST( MRMesh, FeatureAlignmentRejectsOverflowingSolvedPlacementsInEveryMode )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection::Same;
    c.movingAxis = { { 1.5e308, 1.5e308, 0 }, { 1, 0, 0 } };
    c.fixedAxis = { {}, { 1, 1, 0 } };
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 1.5e308, 0, 0 }, { -1.5e308, 0, 0 } } };
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.fixedPlane.center = { 0, 0, 1.5e308 };
    c.settings.offset = 1.5e308;
    EXPECT_FALSE( solve( c ) );
}

TEST( MRMesh, FeatureAlignmentRejectsOverflowingMeasurementsInEveryMode )
{
    for ( const auto mode : { PrimaryMode::Plane, PrimaryMode::Axis, PrimaryMode::Centers } )
    {
        SCOPED_TRACE( int( mode ) );
        auto c = base();
        c.settings.mode = mode;
        c.movingPlane.center = c.movingAxis.center = { 1.5e308, 0, 0 };
        c.fixedPlane.center = c.fixedAxis.center = { -1.5e308, 0, 0 };
        c.centers = { { { 1.5e308, 0, 0 }, { -1.5e308, 0, 0 } } };
        EXPECT_FALSE( measure( c, {} ) );
        c.fixedPlane.center = c.fixedAxis.center = {};
        c.centers.front().fixed = {};
        EXPECT_FALSE( measure( c, AffineXf3d::translation( { 1.5e308, 0, 0 } ) ) );
    }
}

TEST( MRMesh, FeatureAlignmentRejectsUnrepresentableLocatorNorms )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { {}, {} }, { { 1e200, 0, 0 }, { 1e200, 0, 0 } }, { { 0, 1e200, 0 }, { 0, 1e200, 0 } } };
    EXPECT_FALSE( solve( c ) );
    EXPECT_FALSE( measure( c, {} ) );
    c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 1e200, 0, 0 }, { 1e200, 0, 0 } } };
    EXPECT_FALSE( solve( c ) );
    EXPECT_FALSE( measure( c, {} ) );
}

} // namespace MR
