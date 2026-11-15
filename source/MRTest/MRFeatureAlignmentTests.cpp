#include <MRMesh/MRFeatureAlignment.h>
#include <MRMesh/MRPlaneObject.h>
#include <MRMesh/MRCircleObject.h>
#include <MRMesh/MRCylinderObject.h>
#include <MRMesh/MRPointObject.h>
#include <MRMesh/MRSphereObject.h>
#include <MRMesh/MRLineObject.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>

namespace MR
{
namespace
{
using namespace FeatureAlignment;

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
void expectVectorNear( Vector3d a, Vector3d b, double eps = 1e-8 )
{
    EXPECT_NEAR( ( a - b ).length(), 0, eps );
}
} // namespace

TEST( MRMesh, FeatureAlignmentScanToleranceDefaultsAreEditableStartingValues )
{
    Settings s;
    ASSERT_TRUE( s.linearTolerance == 1. && s.angularToleranceDeg == 5. ) << "scan defaults";
    s.linearTolerance = .2;
    s.angularToleranceDeg = .5;
    auto c = base();
    c.settings = s;
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    ASSERT_TRUE( solved->acceptable() ) << "custom tolerances accepted";
}

TEST( MRMesh, FeatureAlignmentPlaneOnlyPreservesTangentialPosition )
{
    auto c = base();
    c.movingPlane.center = { 8, 3, 20 };
    c.fixedPlane.center = { 100, -50, 2 };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    expectVectorNear( r.delta( c.movingPlane.center ), { 8, 3, 2 } );
    ASSERT_TRUE( r.remainingDof == 3 ) << "DOF";
}

TEST( MRMesh, FeatureAlignmentTiltedPlaneCenterConvention )
{
    auto c = base();
    c.movingPlane = { { 50, 20, 30 }, { 1, 0, 0 } };
    c.fixedPlane.center = { 0, 0, 4 };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    expectVectorNear( solved->delta( c.movingPlane.center ), { 50, 20, 4 } );
}

TEST( MRMesh, FeatureAlignmentOpposingNormalsAndSignedGap )
{
    auto c = base();
    c.settings.normals = NormalRelation::Opposing;
    c.settings.offset = 2;
    c.fixedPlane.center = { 0, 0, 7 };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    expectVectorNear( r.delta.A * Vector3d{ 0, 0, 1 }, { 0, 0, -1 } );
    expectVectorNear( r.delta( { 0, 0, 0 } ), { 0, 0, 9 } );
    ASSERT_TRUE( std::abs( r.signedPlaneGap - 2 ) < 1e-9 ) << "gap";
}

TEST( MRMesh, FeatureAlignmentOneHoleLeavesRotation )
{
    auto c = base();
    c.holes = { { h( { 3, 4, 0 } ), h( { 20, 30, 0 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    expectVectorNear( r.delta( { 3, 4, 0 } ), { 20, 30, 0 } );
    ASSERT_TRUE( r.remainingDof == 1 ) << "DOF";
}

TEST( MRMesh, FeatureAlignmentSecondHoleSetsRotation )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 10, 20, 0 } ) }, { h( { 10, 0, 0 } ), h( { 10, 30, 0 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    expectVectorNear( r.delta( { 10, 0, 0 } ), { 10, 30, 0 } );
    ASSERT_TRUE( r.remainingDof == 0 && r.acceptable() ) << "constraints";
}

TEST( MRMesh, FeatureAlignmentMismatchNeverScales )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 } ) }, { h( { 10, 0, 0 } ), h( { 0, 11, 0 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( !r.acceptable() && std::abs( r.holes[1].position - 1 ) < 1e-8 ) << "mismatch";
    ASSERT_TRUE( isRigid( r.delta ) ) << "rigid";
}

TEST( MRMesh, FeatureAlignmentCylinderAxisIgnoresAxialDisplayCenter )
{
    auto c = base();
    c.holes = { { h( { 1, 2, 100 } ), h( { 3, 4, -100 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.acceptable() && r.maximumPositionError() < 1e-9 ) << "axis center";
}

TEST( MRMesh, FeatureAlignmentPlaneCenterLocatorsPreserveSeatingInsteadOfMatchingFeatureDepths )
{
    auto c = base();
    c.settings.holeMode = HoleMode::Centers;
    c.holes = { { h( { 1, 2, 5 } ), h( { 3, 4, 0 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.acceptable() && r.maximumPositionError() < 1e-9 ) << "depth is not a plane locator constraint";
    ASSERT_TRUE( std::abs( r.holes[0].axial - 5 ) < 1e-9 ) << "retain depth diagnostic";
    ASSERT_TRUE( std::abs( r.signedPlaneGap ) < 1e-9 ) << "plane seating";
}

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
            const auto solved = solve( c );
            ASSERT_TRUE( solved ) << solved.error();
            const auto& r = *solved;
            ASSERT_TRUE( r.acceptable() && r.maximumPositionError() < 1e-8 ) << "projected hole centers";
            ASSERT_TRUE( std::abs( r.signedPlaneGap - gap ) < 1e-9 && isRigid( r.delta ) ) << "rigid plane priority";
        }
}

TEST( MRMesh, FeatureAlignmentRealisticTwoHoleMismatchRejectsOnlyTheIncompatibleProjectedLocator )
{
    auto c = base();
    c.settings.holeMode = HoleMode::Centers;
    c.settings.linearTolerance = .5;
    c.holes = { { h( { 0, 0, -8.9457646 } ), h( { 0, 0, 0 } ) },
                { h( { 10.708838, 0, 2.0678686 } ), h( { 10, 0, 0 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( !r.acceptable() && r.conflicts.size() == 1 ) << "only the second hole conflicts";
    ASSERT_TRUE( r.holes[0].position < 1e-8 && std::abs( r.holes[1].position - .708838 ) < 1e-8 ) << "controlled position errors";
    ASSERT_TRUE( std::abs( r.maximumPositionError() - .708838 ) < 1e-8 ) << "maximum controlled residual";
}

TEST( MRMesh, FeatureAlignmentExplicitBalancedHoleFitSharesSpacingErrorWithoutChangingPlaneOrScale )
{
    auto c = base();
    c.settings.holeMode = HoleMode::Centers;
    c.settings.balanceHoles = true;
    c.settings.linearTolerance = .5;
    c.settings.offset = 10;
    c.holes = { { h( { 0, 0, -8.9457646 } ), h( { 0, 0, 0 } ) },
                { h( { 10.708838, 0, 2.0678686 } ), h( { 10, 0, 0 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.acceptable() ) << "both projected centers can meet tolerance";
    ASSERT_TRUE( std::abs( r.holes[0].position - .354419 ) < 1e-8 && std::abs( r.holes[1].position - .354419 ) < 1e-8 ) << "symmetric residuals";
    ASSERT_TRUE( std::abs( r.signedPlaneGap - 10 ) < 1e-9 && isRigid( r.delta ) ) << "plane and rigid geometry";
}

TEST( MRMesh, FeatureAlignmentBalancedFitStillRejectsIncompatibleSpacing )
{
    auto c = base();
    c.settings.holeMode = HoleMode::Centers;
    c.settings.balanceHoles = true;
    c.settings.linearTolerance = .5;
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 } ) }, { h( { 12, 0, 3 } ), h( { 10, 0, -4 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( !r.acceptable() && std::abs( r.maximumPositionError() - 1 ) < 1e-8 ) << "honest balanced residual";
    ASSERT_TRUE( isRigid( r.delta ) ) << "no scale";
}

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
                            const auto solved = solve( c );
                            ASSERT_TRUE( solved ) << solved.error();
                            const auto& r = *solved;
                            ASSERT_TRUE( r.acceptable() && r.maximumPositionError() < 1e-7 ) << "mixed projected/axis locators";
                            ASSERT_TRUE( std::abs( r.signedPlaneGap - gap ) < 1e-7 && r.normalErrorDeg < 1e-5 ) << "locked plane";
                            ASSERT_TRUE( r.remainingDof == ( count == 1 ? 1 : 0 ) && isRigid( r.delta ) ) << "honest freedom and rigid placement";
                        }
}

TEST( MRMesh, FeatureAlignmentObliqueAxisSetsRotation )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 }, { 1, 0, 1 } ), h( { 4, 7, 0 }, { 0, 1, 1 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.acceptable() && r.remainingDof == 0 ) << "oblique";
    expectVectorNear( r.delta.A * Vector3d{ 1, 0, 1 }, { 0, 1, 1 } );
}

TEST( MRMesh, FeatureAlignmentNearNormalUsableAxisDeterminesRotation )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 }, { .0001, 0, 1 } ), h( { 0, 0, 0 }, { 0, .0001, 1 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    EXPECT_EQ( solved->remainingDof, 0 );
    EXPECT_NEAR( solved->yawDeg, 90, 1e-10 );
    expectVectorNear( solved->delta.A * c.holes[0].moving.axis, c.holes[0].fixed.axis, 1e-12 );
}

TEST( MRMesh, FeatureAlignmentAxesAreUnoriented )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 }, { 1, 0, 1 } ), h( { 0, 0, 0 }, { 0, -1, -1 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    ASSERT_TRUE( solved->acceptable() ) << "unoriented";
}

TEST( MRMesh, FeatureAlignmentIncompatibleTiltBlocksApply )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 }, { 1, 0, 1 } ) } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    ASSERT_TRUE( !solved->acceptable() ) << "tilt conflict";
}

TEST( MRMesh, FeatureAlignmentDirectedClockingWithOneHole )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 4, 7, 0 } ) } };
    c.clocking = { { { 1, 0, 0 }, { 0, -1, 0 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    expectVectorNear( r.delta.A * Vector3d{ 1, 0, 0 }, { 0, -1, 0 } );
    ASSERT_TRUE( r.remainingDof == 0 ) << "DOF";

    // Direct vector targets verify signed residual spin across the branch cut and rotated frames.
    for ( const double angle : { -std::numbers::pi, -2.7, -1.2, -0.01, 0., 0.01, 1.2, 2.7, std::numbers::pi } )
        for ( const auto normal : { Vector3d{ 0, 0, 1 }, Vector3d{ 1, -2, 3 }.normalized() } )
        {
            SCOPED_TRACE( angle );
            auto sweep = base();
            sweep.movingPlane.normal = normal;
            sweep.fixedPlane.normal = normal;
            const auto direction = cross( normal, Vector3d{ 1, 1, 0 } ).normalized();
            const auto target = std::cos( angle ) * direction + std::sin( angle ) * cross( normal, direction );
            sweep.clocking = { { direction, target } };
            const auto rotated = solve( sweep );
            ASSERT_TRUE( rotated ) << rotated.error();
            EXPECT_TRUE( rotated->acceptable() );
            expectVectorNear( rotated->delta.A * direction, target, 1e-10 );
            expectVectorNear( rotated->delta.A * normal, normal, 1e-10 );
            EXPECT_NEAR( std::abs( rotated->yawDeg ), std::abs( angle ) * 180. / std::numbers::pi, 1e-9 );
        }
}

TEST( MRMesh, FeatureAlignmentDirectedReferenceConflict )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 } ) }, { h( { 10, 0, 0 } ), h( { 0, 10, 0 } ) } };
    c.clocking = { { { 1, 0, 0 }, { 1, 0, 0 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    ASSERT_TRUE( !solved->acceptable() ) << "clock conflict";
}

TEST( MRMesh, FeatureAlignmentZeroProjectedBaselineRejected )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 0 } ), h( { 0, 0, 0 } ) }, { h( { 0, 0, 5 } ), h( { 10, 0, 0 } ) } };
    ASSERT_TRUE( !solve( c ) ) << "degenerate baseline";
}

TEST( MRMesh, FeatureAlignmentParallelAxisRejected )
{
    auto c = base();
    c.holes = { { h( { 0, 0, 1 }, { 1, 0, 0 } ), h( { 0, 0, 0 } ) } };
    ASSERT_TRUE( !solve( c ) ) << "parallel axis";
}

TEST( MRMesh, FeatureAlignmentNonfiniteGeometryRejected )
{
    auto c = base();
    c.movingPlane.center.x = std::numeric_limits<double>::quiet_NaN();
    ASSERT_TRUE( !solve( c ) ) << "nonfinite";

    // Unused finite center data cannot introduce a plane locator or alter its free motion.
    c = base();
    c.centers = { { { 10, -5, 7 }, { 99, 8, -4 } } };
    const auto finiteCenters = solve( c );
    ASSERT_TRUE( finiteCenters ) << finiteCenters.error();
    EXPECT_TRUE( finiteCenters->acceptable() );
    EXPECT_EQ( finiteCenters->remainingDof, 3 );
    expectVectorNear( finiteCenters->delta.b, {} );
    expectVectorNear( finiteCenters->delta.A.x, { 1, 0, 0 } );
    expectVectorNear( finiteCenters->delta.A.y, { 0, 1, 0 } );
    expectVectorNear( finiteCenters->delta.A.z, { 0, 0, 1 } );
    EXPECT_TRUE( measure( c, {} ) );
    // Every supplied datum must be finite, including a center unused by the selected mode.
    for ( const double value : { std::numeric_limits<double>::infinity(),
                                -std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN() } )
    {
        c.centers.front().moving.x = value;
        EXPECT_FALSE( solve( c ) );
        EXPECT_FALSE( measure( c, {} ) );
    }
}

TEST( MRMesh, FeatureAlignmentZeroNormalRejected )
{
    auto c = base();
    c.movingPlane.normal = {};
    ASSERT_TRUE( !solve( c ) ) << "normal";
}

TEST( MRMesh, FeatureAlignmentNegativeRadiusRejected )
{
    auto c = base();
    c.holes = { { h( {} ), h( {} ) } };
    c.holes[0].moving.radius = -1;
    ASSERT_TRUE( !solve( c ) ) << "radius";
}

TEST( MRMesh, FeatureAlignmentInvalidTolerancesRejected )
{
    auto c = base();
    c.settings.linearTolerance = -1;
    ASSERT_TRUE( !solve( c ) ) << "negative tolerance";
    c = base();
    c.settings.angularToleranceDeg = 90;
    ASSERT_TRUE( !solve( c ) ) << "angle tolerance";
}

TEST( MRMesh, FeatureAlignmentRigidValidationRejectsMirrorAndScale )
{
    AffineXf3d x;
    ASSERT_TRUE( isRigid( x ) ) << "identity";
    x.A.x.x = 2;
    ASSERT_TRUE( !isRigid( x ) ) << "scale";
    x.A.x.x = -1;
    ASSERT_TRUE( !isRigid( x ) ) << "reflection";
}

TEST( MRMesh, FeatureAlignmentMeasureDoesNotHideASuppliedPositionError )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.holes = { { h( {} ), h( {} ) } };
    auto r = measure( c, AffineXf3d::translation( { 1, 0, 0 } ) );
    ASSERT_TRUE( r && !r->acceptable() && std::abs( r->holes[0].position - 1 ) < 1e-10 ) << "measurement";
}

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
        const auto solved = solve( c );
        ASSERT_TRUE( solved ) << solved.error();
        const auto& solution = *solved;
        ASSERT_TRUE( solution.acceptable() && solution.remainingDof == 0 ) << "random constraints";
        expectVectorNear( solution.delta( { 4, 5, 6 } ), desired( { 4, 5, 6 } ), 1e-8 );
        ASSERT_TRUE( isRigid( solution.delta ) ) << "random rigidity";
    }
}

TEST( MRMesh, FeatureAlignmentAxisCentersAlignWithoutAPlane )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.movingAxis = { { 3, 4, 9 }, { 1, 0, 0 } };
    c.fixedAxis = { { 10, 20, 0 }, { 0, 0, 1 } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.acceptable() && r.remainingDof == 1 ) << "axis DOF";
    expectVectorNear( r.delta( c.movingAxis.center ), c.fixedAxis.center );
}

TEST( MRMesh, FeatureAlignmentCoaxialAxesRetainSlideAndSpin )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.movingAxis.center = { 3, 4, 9 };
    c.fixedAxis.center = { 20, 30, 0 };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 2 && r.acceptable() ) << "axis free motions";
    expectVectorNear( r.delta( c.movingAxis.center ), { 20, 30, 9 } );
}

TEST( MRMesh, FeatureAlignmentAxisDirectionOnlyRetainsFourFreedoms )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.settings.coaxial = false;
    c.movingAxis = { { 3, 4, 9 }, { 1, 0, 0 } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 4 ) << "direction DOF";
    expectVectorNear( r.delta( c.movingAxis.center ), c.movingAxis.center );
}

TEST( MRMesh, FeatureAlignmentOffAxisCenterSetsCylinderRotation )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.fixedAxis.center = { 20, 30, 0 };
    c.centers = { { { 10, 0, 5 }, { 20, 40, 5 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.acceptable() && r.remainingDof == 0 ) << "locator";
    expectVectorNear( r.delta( { 10, 0, 5 } ), { 20, 40, 5 } );
}

TEST( MRMesh, FeatureAlignmentOnAxisCenterSetsSlideButNotSpin )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.fixedAxis.center = { 3, 4, 0 };
    c.centers = { { { 0, 0, 7 }, { 3, 4, 20 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.acceptable() && r.remainingDof == 1 ) << "on-axis locator";
    expectVectorNear( r.delta( { 0, 0, 7 } ), { 3, 4, 20 } );
}

TEST( MRMesh, FeatureAlignmentInconsistentRadialLocatorIsAConflict )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 5, 0, 0 }, { 6, 0, 0 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( !r.acceptable() && isRigid( r.delta ) ) << "radial mismatch";
}

TEST( MRMesh, FeatureAlignmentPrimaryAxialCenterHasPriority )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 5, 0, 4 }, { 5, 0, 6 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( !r.acceptable() && r.primaryPositionError < 1e-8 ) << "axial priority";
}

TEST( MRMesh, FeatureAlignmentSignedAxialOffset )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.offset = 2;
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    expectVectorNear( r.delta( {} ), { 0, 0, 2 } );
    ASSERT_TRUE( r.acceptable() ) << "axis gap";
}

TEST( MRMesh, FeatureAlignmentDirectedAxisClockingLeavesSlide )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.clocking = { { { 1, 0, 0 }, { 0, 1, 0 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 1 && r.acceptable() ) << "axis clocking";
    expectVectorNear( r.delta.A * Vector3d{ 1, 0, 0 }, { 0, 1, 0 } );
}

TEST( MRMesh, FeatureAlignmentDirectionOnlyPlusOneCenterLeavesSpin )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.coaxial = false;
    c.settings.matchAxialCenters = false;
    c.centers = { { { 10, 0, 0 }, { 20, 30, 4 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 1 && r.acceptable() ) << "direction and center";
}

TEST( MRMesh, FeatureAlignmentDirectionOnlyPlusTwoCentersSetsRotation )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.coaxial = false;
    c.settings.matchAxialCenters = false;
    c.centers = { { { 10, 0, 0 }, { 20, 30, 4 } }, { { 20, 0, 0 }, { 20, 40, 4 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 0 && r.acceptable() ) << "direction and centers";
}

TEST( MRMesh, FeatureAlignmentUnorientedAxesChooseTheSignConsistentWithLocators )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 5, 0, 3 }, { 5, 0, -3 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.acceptable() ) << "axis sign must follow compatible measurements";
    expectVectorNear( r.delta.A * Vector3d{ 0, 0, 1 }, { 0, 0, -1 } );
}

TEST( MRMesh, FeatureAlignmentZeroAxisRejected )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.movingAxis.direction = {};
    ASSERT_TRUE( !solve( c ) ) << "zero axis";
}

TEST( MRMesh, FeatureAlignmentCenterOnlyOnePairLeavesThreeRotations )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 3, 4, 5 }, { 10, 20, 30 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 3 && r.acceptable() ) << "center DOF";
    expectVectorNear( r.delta( { 3, 4, 5 } ), { 10, 20, 30 } );
}

TEST( MRMesh, FeatureAlignmentCenterOnlyTwoPairsLeaveOneRotation )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 10, 20, 30 } }, { { 10, 0, 0 }, { 10, 30, 30 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 1 && r.acceptable() ) << "two centers";
}

TEST( MRMesh, FeatureAlignmentCenterOnlyThreePointsFullyLocate )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    AffineXf3d x{ Matrix3d::rotation( { 1, 2, -3 }, .8 ), { 10, 20, 30 } };
    for ( Vector3d p : std::vector<Vector3d>{ { 0, 0, 0 }, { 10, 0, 0 }, { 0, 5, 1 } } )
        c.centers.push_back( { p, x( p ) } );
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 0 && r.acceptable() ) << "three centers";
    expectVectorNear( r.delta( { 3, 4, 5 } ), x( { 3, 4, 5 } ) );
}

TEST( MRMesh, FeatureAlignmentCenterOnlyMismatchNeverScales )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } }, { { 10, 0, 0 }, { 0, 11, 0 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( !r.acceptable() && isRigid( r.delta ) ) << "center mismatch";
}

TEST( MRMesh, FeatureAlignmentCollinearCentersReportRemainingRotation )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } }, { { 10, 0, 0 }, { 0, 10, 0 } }, { { 20, 0, 0 }, { 0, 20, 0 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    ASSERT_TRUE( r.remainingDof == 1 && r.acceptable() ) << "collinear DOF";
}

TEST( MRMesh, FeatureAlignmentDuplicateSecondaryCenterRejected )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } }, { { 0, 0, 0 }, { 1, 0, 0 } } };
    ASSERT_TRUE( !solve( c ) ) << "duplicate center";
}

TEST( MRMesh, FeatureAlignmentCenterOnlyOffsetHasNoAssumedDirection )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.settings.offset = 1;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } } };
    ASSERT_TRUE( !solve( c ) ) << "undefined offset";
}

TEST( MRMesh, FeatureAlignmentAxisLocatorAndClockingConflict )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 5, 0, 0 }, { 0, 5, 0 } } };
    c.clocking = { { { 1, 0, 0 }, { 1, 0, 0 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    ASSERT_TRUE( !solved->acceptable() ) << "clocking conflict";
}

TEST( MRMesh, FeatureAlignmentInvalidPrimaryModeRejected )
{
    auto c = base();
    c.settings.mode = PrimaryMode( 99 );
    ASSERT_TRUE( !solve( c ) ) << "mode";
}

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
        const auto axisSolved = solve( c );
        ASSERT_TRUE( axisSolved ) << axisSolved.error();
        const auto& axis = *axisSolved;
        ASSERT_TRUE( axis.acceptable() && axis.remainingDof == 0 ) << "random axis";
        expectVectorNear( axis.delta( { 2, 3, 4 } ), x( { 2, 3, 4 } ) );
        c = base();
        c.settings.mode = PrimaryMode::Centers;
        for ( Vector3d p : std::vector<Vector3d>{ { 0, 0, 0 }, { 20, 0, 0 }, { 0, 10, 3 } } )
            c.centers.push_back( { p, x( p ) } );
        const auto centerSolved = solve( c );
        ASSERT_TRUE( centerSolved ) << centerSolved.error();
        const auto& center = *centerSolved;
        ASSERT_TRUE( center.acceptable() && center.remainingDof == 0 ) << "random center";
        expectVectorNear( center.delta( { 2, 3, 4 } ), x( { 2, 3, 4 } ) );
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
Expected<Solution> solveReferences( const FeatureReferences &references, const Settings &settings )
{
    const auto constraints = readConstraints( references, settings );
    if ( !constraints )
        return unexpected( constraints.error() );
    return solve( *constraints );
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
    const auto pinned = solveReferences( refs, settings );
    ASSERT_TRUE( pinned ) << pinned.error();
    EXPECT_NEAR( pinned->signedPlaneGap, -7, 1e-9 );
    EXPECT_NEAR( pinned->holes[0].position, 0, 1e-9 );
    EXPECT_NEAR( pinned->holes[1].position, 2, 1e-9 );
    EXPECT_FALSE( pinned->acceptable() );
    settings.balanceHoles = true;
    const auto balanced = solveReferences( refs, settings );
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
    const auto one = solveReferences( refs, settings );
    ASSERT_TRUE( one ) << one.error();
    EXPECT_TRUE( one->acceptable() );
    EXPECT_EQ( one->remainingDof, 1 );
    EXPECT_NEAR( one->signedPlaneGap, 0, 1e-9 );
    EXPECT_NEAR( one->holes[0].inPlane, 0, 1e-9 );
    refs.locators[1] = { fittedCircle( { 13, 4, 75 } ), fittedCircle( { 18, 10, -80 } ) };
    const auto two = solveReferences( refs, settings );
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
    const auto result = solveReferences( refs, settings );
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
    const auto result = solveReferences( refs, settings );
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
    const auto result = solveReferences( refs, settings );
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
    auto result = solveReferences( refs, settings );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->remainingDof, 3 );
    refs.locators[0] = { fittedPoint( { 10, 0, 0 } ), fittedCircle( { 10, 30, 30 } ) };
    result = solveReferences( refs, settings );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->remainingDof, 1 );
    refs.locators[1] = { fittedPoint( { 0, 4, 0 } ), fittedPoint( { 6, 20, 30 } ) };
    result = solveReferences( refs, settings );
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
    cylinder->setLength( 1e-12f );
    ASSERT_GT( cylinder->xf().A.det(), 0 );
    refs.locators[0] = { cylinder, fittedCircle( {} ) };
    EXPECT_FALSE( solveReferences( refs, base().settings ) );
    auto plane = fittedPlane();
    plane->setSizeX( 1e-12f );
    ASSERT_GT( plane->xf().A.det(), 0 );
    refs = planeReferences();
    refs.primary.moving = plane;
    EXPECT_FALSE( solveReferences( refs, base().settings ) );
    refs = planeReferences();
    refs.locators[0] = { fittedCircle( {}, { 1, 0, 0 } ), fittedCircle( {} ) };
    EXPECT_FALSE( solveReferences( refs, base().settings ) );
}

TEST( MRMesh, FeatureAlignmentFeatureClockingUsesDirectedFittedLine )
{
    auto refs = planeReferences();
    auto moving = std::make_shared<LineObject>();
    auto fixed = std::make_shared<LineObject>();
    moving->setDirection( { 1, 0, 0 } );
    fixed->setDirection( { 0, -1, 0 } );
    refs.clocking = { moving, fixed };
    const auto result = solveReferences( refs, base().settings );
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
        EXPECT_NE( result->freeMotion, FreeMotion::Unspecified );
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
        EXPECT_NE( result->freeMotion, FreeMotion::Unspecified );
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
        EXPECT_NE( result->freeMotion, FreeMotion::Unspecified );
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

TEST( MRMesh, FeatureAlignmentObliqueAxesSetQuarterTurnIndependentlyOfTolerance )
{
    auto c = base();
    c.holes = { { h( {}, { 1, 0, 1 } ), h( { 4, 7, 0 }, { 0, 1, 1 } ) } };
    for ( const double tolerance : { .25, 5., 20., 60. } )
    {
        SCOPED_TRACE( tolerance );
        c.settings.angularToleranceDeg = tolerance;
        const auto result = solve( c );
        ASSERT_TRUE( result ) << result.error();
        EXPECT_EQ( result->remainingDof, 0 );
        EXPECT_NEAR( result->yawDeg, 90, 1e-10 );
        EXPECT_NEAR( ( result->delta.A * Vector3d{ 1, 0, 1 } - Vector3d{ 0, 1, 1 } ).length(), 0, 1e-12 );
        EXPECT_NEAR( result->signedPlaneGap, 0, 1e-12 );
        EXPECT_TRUE( result->acceptable() );
    }
}

TEST( MRMesh, FeatureAlignmentAxisRawCenterDistanceDoesNotSubtractRequestedOffset )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection::Same;
    c.centers = { { { 5, 0, 0 }, { 5, 0, 0 } } };
    for ( const double offset : { -7., 0., 7. } )
    {
        SCOPED_TRACE( offset );
        c.settings.offset = offset;
        const auto result = solve( c );
        ASSERT_TRUE( result ) << result.error();
        ASSERT_EQ( result->holes.size(), 1 );
        EXPECT_NEAR( result->holes[0].position, 0, 1e-12 );
        EXPECT_NEAR( result->holes[0].rawCenterDistance, std::abs( offset ), 1e-12 );
        const auto measured = measure( c, AffineXf3d::translation( { 0, 0, offset } ) );
        ASSERT_TRUE( measured ) << measured.error();
        ASSERT_EQ( measured->holes.size(), 1 );
        EXPECT_NEAR( measured->holes[0].rawCenterDistance, std::abs( offset ), 1e-12 );
    }
}

TEST( MRMesh, FeatureAlignmentParallelMeasuredPlaneClockingIsAnExplicitFiniteConflict )
{
    auto c = base();
    c.clocking = { { Vector3d::plusX(), Vector3d::plusX() } };
    const AffineXf3d supplied{ Matrix3d{ { 0, 0, -1 }, { 0, 1, 0 }, { 1, 0, 0 } }, {} };
    const auto result = measure( c, supplied );
    ASSERT_TRUE( result ) << result.error();
    EXPECT_FALSE( result->acceptable() );
    EXPECT_FALSE( result->clockingErrorDeg.has_value() );
    EXPECT_TRUE( std::any_of( result->conflicts.begin(), result->conflicts.end(), []( const Diagnostic& conflict )
    {
        return conflict == Diagnostic{ DiagnosticCode::ClockingParallelToPrimary };
    } ) );
    EXPECT_TRUE( std::isfinite( result->maximumAngleError() ) );
    EXPECT_EQ( result->delta, supplied );
}

namespace
{
AffineXf3d randomRigidFrame( std::mt19937& generator )
{
    std::uniform_real_distribution<double> unit( -1, 1 );
    Vector3d axis{ unit( generator ), unit( generator ), unit( generator ) };
    if ( axis.length() < .1 )
        axis = Vector3d::plusZ();
    return { Matrix3d::rotation( axis, unit( generator ) * 3 ),
             { unit( generator ) * 100, unit( generator ) * 100, unit( generator ) * 100 } };
}
void expectProperRigid( const AffineXf3d& placement )
{
    EXPECT_NEAR( placement.A.det(), 1, 1e-10 );
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_NEAR( dot( placement.A.col( i ), placement.A.col( j ) ), i == j ? 1. : 0., 1e-10 );
    EXPECT_TRUE( std::isfinite( placement.b.x ) && std::isfinite( placement.b.y ) && std::isfinite( placement.b.z ) );
}
void expectFiniteReport( const Solution& result )
{
    for ( const double value : { result.primaryPositionError, result.primaryAngleErrorDeg, result.yawDeg,
          result.signedPlaneGap, result.planeOffsetError, result.normalErrorDeg, result.orthogonalityError, result.determinant } )
        EXPECT_TRUE( std::isfinite( value ) );
    if ( result.clockingErrorDeg )
        EXPECT_TRUE( std::isfinite( *result.clockingErrorDeg ) );
    for ( const auto& residual : result.holes )
        for ( const double value : { residual.position, residual.inPlane, residual.axial, residual.axisAngleDeg,
              residual.radiusDifference, residual.rawCenterDistance } )
            EXPECT_TRUE( std::isfinite( value ) );
}
} // namespace

TEST( MRMesh, FeatureAlignmentSeededPlaneOraclesCoverDepthModesNormalsOffsetsAndTolerance )
{
    std::mt19937 generator( 73191 );
    std::uniform_real_distribution<double> unit( -1, 1 );
    for ( int iteration = 0; iteration < 160; ++iteration )
    {
        SCOPED_TRACE( iteration );
        const auto frame = randomRigidFrame( generator );
        const auto desired = randomRigidFrame( generator );
        const auto inverse = desired.inverse();
        const auto normal = frame.A * Vector3d::plusZ();
        auto c = base();
        c.settings.normals = iteration % 2 ? NormalRelation::Same : NormalRelation::Opposing;
        c.settings.holeMode = iteration % 3 ? HoleMode::Centers : HoleMode::Axes;
        c.settings.offset = unit( generator ) * 12;
        c.settings.balanceHoles = iteration % 2 != 0;
        c.fixedPlane = { frame.b, normal };
        c.movingPlane = { inverse( frame.b + c.settings.offset * normal ),
                          inverse.A * ( double( int( c.settings.normals ) ) * normal ) };
        const int count = iteration % 4 == 0 ? 1 : 2;
        for ( int i = 0; i < count; ++i )
        {
            const auto anchor = frame( { i == 0 ? -20. : 30., i == 0 ? 4. : -7., 0 } );
            const auto axis = frame.A * ( c.settings.holeMode == HoleMode::Axes ? Vector3d{ .6, .2, 1 }.normalized()
                                                                                 : Vector3d::plusZ() );
            const auto fixedCenter = anchor + unit( generator ) * 40 * axis;
            const auto movingTarget = c.settings.holeMode == HoleMode::Axes
                                        ? anchor + unit( generator ) * 40 * axis
                                        : anchor + ( c.settings.offset + unit( generator ) * 40 ) * normal;
            auto moving = h( inverse( movingTarget ), inverse.A * axis );
            auto fixed = h( fixedCenter, axis );
            moving.kind = iteration % 2 ? HoleKind::Circle : HoleKind::Cylinder;
            fixed.kind = iteration % 3 ? HoleKind::Cylinder : HoleKind::Circle;
            c.holes.push_back( { moving, fixed } );
        }
        // Center mode data must share tangent centers; axial depth is independently arbitrary.
        if ( c.settings.holeMode == HoleMode::Centers )
            for ( int i = 0; i < count; ++i )
            {
                const auto anchor = frame( { i == 0 ? -20. : 30., i == 0 ? 4. : -7., 0 } );
                c.holes[i].fixed.center = anchor + unit( generator ) * 40 * normal;
            }
        for ( const double tolerance : { .1, 30. } )
        {
            c.settings.angularToleranceDeg = tolerance;
            const auto result = solve( c );
            ASSERT_TRUE( result ) << result.error();
            expectProperRigid( result->delta );
            expectFiniteReport( *result );
            EXPECT_TRUE( result->acceptable() );
            EXPECT_NEAR( dot( normal, result->delta( c.movingPlane.center ) - c.fixedPlane.center ), c.settings.offset, 1e-8 );
            expectVectorNear( result->delta.A * c.movingPlane.normal, double( int( c.settings.normals ) ) * normal, 1e-9 );
            EXPECT_EQ( result->remainingDof, count == 1 && c.settings.holeMode == HoleMode::Centers ? 1 : 0 );
            for ( int i = 0; i < count; ++i )
            {
                const auto& pair = c.holes[i];
                Vector3d source = pair.moving.center;
                Vector3d target = pair.fixed.center + c.settings.offset * normal;
                if ( c.settings.holeMode == HoleMode::Axes )
                {
                    source += pair.moving.axis * ( dot( c.movingPlane.normal, c.movingPlane.center - source ) /
                                                   dot( c.movingPlane.normal, pair.moving.axis ) );
                    target = pair.fixed.center + pair.fixed.axis *
                        ( dot( normal, c.fixedPlane.center + c.settings.offset * normal - pair.fixed.center ) /
                          dot( normal, pair.fixed.axis ) );
                }
                const auto difference = result->delta( source ) - target;
                const auto tangent = difference - dot( difference, normal ) * normal;
                EXPECT_NEAR( tangent.length(), 0, 1e-8 );
                EXPECT_NEAR( result->holes[i].position, tangent.length(), 1e-10 );
                EXPECT_NEAR( result->holes[i].rawCenterDistance,
                             ( result->delta( pair.moving.center ) - pair.fixed.center ).length(), 1e-10 );
            }
            if ( count == 2 || c.settings.holeMode == HoleMode::Axes )
                expectVectorNear( result->delta( { 3, -4, 8 } ), desired( { 3, -4, 8 } ), 1e-8 );
        }
    }
}

TEST( MRMesh, FeatureAlignmentSeededAxisOraclesCoverAllLocksDirectionsAndLocatorRanks )
{
    std::mt19937 generator( 83117 );
    for ( int iteration = 0; iteration < 8; ++iteration )
        for ( const auto relation : { AxisDirection::Same, AxisDirection::Opposing, AxisDirection::Closest } )
            for ( const bool coaxial : { false, true } )
                for ( const bool matchCenters : { false, true } )
                    for ( int count : { 0, 1, 2 } )
                        for ( const bool clocked : { false, true } )
                        {
                            SCOPED_TRACE( iteration );
                            SCOPED_TRACE( int( relation ) * 1000 + int( coaxial ) * 100 + int( matchCenters ) * 10 + count * 2 + int( clocked ) );
                            const auto frame = randomRigidFrame( generator );
                            const auto desired = randomRigidFrame( generator );
                            const auto inverse = desired.inverse();
                            const auto n = frame.A * Vector3d::plusZ();
                            const auto u = frame.A * Vector3d::plusX();
                            const auto v = frame.A * Vector3d::plusY();
                            auto c = base();
                            c.settings.mode = PrimaryMode::Axis;
                            c.settings.axisDirection = relation;
                            c.settings.coaxial = coaxial;
                            c.settings.matchAxialCenters = matchCenters;
                            c.settings.offset = matchCenters || count ? ( iteration % 2 ? 3. : -3. ) : 0.;
                            const auto directionSign = relation == AxisDirection::Opposing ? -1. : 1.;
                            const double slide = matchCenters ? c.settings.offset : 6.;
                            c.fixedAxis = { frame.b, n };
                            c.movingAxis = { inverse( frame.b + slide * n ), directionSign * ( inverse.A * n ) };
                            for ( int i = 0; i < count; ++i )
                            {
                                const auto target = frame.b + ( i == 0 ? 13. * u + 2. * n : 9. * v + 7. * n );
                                c.centers.push_back( { inverse( target + c.settings.offset * n ), target } );
                            }
                            if ( clocked )
                                c.clocking = { { inverse.A * u, u } };
                            const auto result = solve( c );
                            ASSERT_TRUE( result ) << result.error();
                            EXPECT_TRUE( result->acceptable() );
                            expectProperRigid( result->delta );
                            expectFiniteReport( *result );
                            const auto transformedAxis = result->delta.A * c.movingAxis.direction;
                            if ( relation == AxisDirection::Closest )
                                EXPECT_NEAR( std::abs( dot( transformedAxis, n ) ), 1, 1e-10 );
                            else
                                expectVectorNear( transformedAxis, directionSign * n, 1e-9 );
                            const auto displacement = result->delta( c.movingAxis.center ) - c.fixedAxis.center;
                            if ( matchCenters )
                                expectVectorNear( displacement, c.settings.offset * n, 1e-8 );
                            else if ( coaxial )
                                EXPECT_NEAR( ( displacement - dot( displacement, n ) * n ).length(), 0, 1e-8 );
                            else if ( count == 0 )
                                expectVectorNear( result->delta( c.movingAxis.center ), c.movingAxis.center, 1e-8 );
                            for ( int i = 0; i < count; ++i )
                            {
                                const auto raw = result->delta( c.centers[i].moving ) - c.centers[i].fixed;
                                EXPECT_NEAR( ( raw - c.settings.offset * n ).length(), 0, 1e-8 );
                                EXPECT_NEAR( result->holes[i].rawCenterDistance, raw.length(), 1e-9 );
                            }
                            if ( clocked )
                                expectVectorNear( result->delta.A * c.clocking->moving, u, 1e-9 );
                            const bool spinLocked = clocked || ( ( coaxial || matchCenters ) ? count >= 1 : count == 2 );
                            const bool positionLocked = matchCenters || count > 0;
                            const int expectedRank = ( spinLocked ? 0 : 1 ) + ( positionLocked ? 0 : coaxial ? 1 : 3 );
                            EXPECT_EQ( result->remainingDof, expectedRank );
                            // Closest axes can admit two discrete placements with one locator; do not invent uniqueness.
                            if ( expectedRank == 0 && relation != AxisDirection::Closest )
                                expectVectorNear( result->delta( { 2, -3, 7 } ), desired( { 2, -3, 7 } ), 1e-8 );
                        }
}

TEST( MRMesh, FeatureAlignmentSeededCenterOraclesPreservePrimaryAndDistinguishCollinearRank )
{
    std::mt19937 generator( 14533 );
    for ( int iteration = 0; iteration < 100; ++iteration )
        for ( int count : { 1, 2, 3 } )
            for ( const bool collinear : { false, true } )
            {
                SCOPED_TRACE( iteration );
                SCOPED_TRACE( count * 2 + int( collinear ) );
                const auto frame = randomRigidFrame( generator );
                const auto desired = randomRigidFrame( generator );
                auto c = base();
                c.settings.mode = PrimaryMode::Centers;
                const std::array<Vector3d, 3> points{ frame( {} ), frame( { 10, 0, 0 } ), frame( collinear ? Vector3d{ 25, 0, 0 } : Vector3d{ 0, 7, 3 } ) };
                for ( int i = 0; i < count; ++i )
                    c.centers.push_back( { points[i], desired( points[i] ) } );
                const auto result = solve( c );
                ASSERT_TRUE( result ) << result.error();
                expectProperRigid( result->delta );
                expectFiniteReport( *result );
                EXPECT_TRUE( result->acceptable() );
                EXPECT_EQ( result->remainingDof, count == 1 ? 3 : count == 2 || collinear ? 1 : 0 );
                for ( const auto& pair : c.centers )
                    expectVectorNear( result->delta( pair.moving ), pair.fixed, 1e-8 );
                if ( count == 1 )
                    EXPECT_NEAR( ( result->delta.A - Matrix3d{} ).norm(), 0, 1e-12 );
                if ( count == 3 && !collinear )
                    expectVectorNear( result->delta( { 4, 3, -2 } ), desired( { 4, 3, -2 } ), 1e-8 );
            }
}

TEST( MRMesh, FeatureAlignmentKnownPlacementsAreEquivariantUnderCommonFrameChanges )
{
    std::mt19937 generator( 16121 );
    for ( int iteration = 0; iteration < 60; ++iteration )
    {
        SCOPED_TRACE( iteration );
        const auto frame = randomRigidFrame( generator );
        const auto desired = randomRigidFrame( generator );
        const auto targetInFrame = frame * desired * frame.inverse();
        for ( const auto mode : { PrimaryMode::Plane, PrimaryMode::Axis, PrimaryMode::Centers } )
        {
            auto c = base();
            c.settings.mode = mode;
            c.settings.axisDirection = AxisDirection::Same;
            if ( mode == PrimaryMode::Plane )
            {
                c.movingPlane = { frame( {} ), frame.A * Vector3d::plusZ() };
                c.fixedPlane = { frame( desired( {} ) ), frame.A * ( desired.A * Vector3d::plusZ() ) };
                for ( const auto point : { Vector3d{ -10, 0, 0 }, Vector3d{ 10, 0, 0 } } )
                    c.holes.push_back( { h( frame( point ), frame.A * Vector3d::plusZ() ), h( frame( desired( point ) ), frame.A * ( desired.A * Vector3d::plusZ() ) ) } );
            }
            else if ( mode == PrimaryMode::Axis )
            {
                c.movingAxis = { frame( {} ), frame.A * Vector3d::plusZ() };
                c.fixedAxis = { frame( desired( {} ) ), frame.A * ( desired.A * Vector3d::plusZ() ) };
                c.centers.push_back( { frame( { 10, 0, 2 } ), frame( desired( { 10, 0, 2 } ) ) } );
            }
            else
                for ( const auto point : { Vector3d{}, Vector3d{ 10, 0, 0 }, Vector3d{ 0, 5, 3 } } )
                    c.centers.push_back( { frame( point ), frame( desired( point ) ) } );
            const auto result = solve( c );
            ASSERT_TRUE( result ) << result.error();
            EXPECT_TRUE( result->acceptable() );
            EXPECT_EQ( result->remainingDof, 0 );
            expectVectorNear( result->delta( { 5, 6, -3 } ), targetInFrame( { 5, 6, -3 } ), 1e-8 );
        }
    }
}

TEST( MRMesh, FeatureAlignmentConflictingLocatorsDoNotRelaxPrimaryLocksWhenToleranceChanges )
{
    for ( const double tolerance : { .01, 2. } )
    {
        auto c = base();
        c.settings.linearTolerance = tolerance;
        c.settings.mode = PrimaryMode::Centers;
        c.centers = { { {}, { 3, 4, 5 } }, { { 10, 0, 0 }, { 3, 15, 5 } } };
        auto result = solve( c );
        ASSERT_TRUE( result ) << result.error();
        expectVectorNear( result->delta( {} ), { 3, 4, 5 }, 1e-12 );
        expectVectorNear( result->delta.A * Vector3d::plusX(), Vector3d::plusY(), 1e-12 );
        EXPECT_NEAR( result->holes[1].position, 1, 1e-12 );
        EXPECT_EQ( result->acceptable(), tolerance >= 1 );
        c = base();
        c.settings.mode = PrimaryMode::Axis;
        c.settings.axisDirection = AxisDirection::Same;
        c.settings.linearTolerance = tolerance;
        c.centers = { { { 5, 0, 3 }, { 6, 0, 4 } } };
        result = solve( c );
        ASSERT_TRUE( result ) << result.error();
        expectVectorNear( result->delta( {} ), {}, 1e-12 );
        expectVectorNear( result->delta.A * Vector3d::plusZ(), Vector3d::plusZ(), 1e-12 );
        EXPECT_NEAR( result->holes[0].position, std::sqrt( 2. ), 1e-12 );
        EXPECT_EQ( result->acceptable(), tolerance >= std::sqrt( 2. ) );
    }
}

TEST( MRMesh, FeatureAlignmentRejectsInvalidEnumsNonfiniteInputsAndUnstableIntersections )
{
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto infinity = std::numeric_limits<double>::infinity();
    for ( const double invalid : { nan, infinity, -infinity } )
    {
        for ( const auto mode : { PrimaryMode::Plane, PrimaryMode::Axis, PrimaryMode::Centers } )
        {
            auto c = base();
            c.settings.mode = mode;
            c.centers = { { {}, {} } };
            c.settings.offset = invalid;
            EXPECT_FALSE( solve( c ) );
            c.settings.offset = 0;
            c.settings.linearTolerance = invalid;
            EXPECT_FALSE( solve( c ) );
            c.settings.linearTolerance = 1;
            c.settings.angularToleranceDeg = invalid;
            EXPECT_FALSE( solve( c ) );
        }
        AffineXf3d transform;
        transform.b.x = invalid;
        EXPECT_FALSE( isRigid( transform ) );
    }
    auto c = base();
    c.settings.normals = NormalRelation( 0 );
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.settings.holeMode = HoleMode( 9 );
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection( 9 );
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.holes = { { h( {}, { 1, 0, 1e-8 } ), h( {} ) } };
    EXPECT_FALSE( solve( c ) );
    c.holes[0].moving.axis = { 1, 0, 1e-4 };
    c.holes[0].fixed.axis = c.holes[0].moving.axis;
    const auto stable = solve( c );
    ASSERT_TRUE( stable ) << stable.error();
    EXPECT_TRUE( stable->acceptable() );
    c.holes[0].moving.kind = HoleKind( 9 );
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.clocking = { { Vector3d::plusZ(), Vector3d::plusX() } };
    EXPECT_FALSE( solve( c ) );
    EXPECT_FALSE( isRigid( {}, -1 ) );
    EXPECT_FALSE( isRigid( {}, nan ) );
}

TEST( MRMesh, FeatureAlignmentOnAxisLocatorsConstrainPositionButNeverInventSpin )
{
    for ( const bool coaxial : { false, true } )
        for ( const bool matchCenters : { false, true } )
            for ( int count : { 0, 1, 2 } )
                for ( const bool clocked : { false, true } )
                {
                    SCOPED_TRACE( int( coaxial ) * 100 + int( matchCenters ) * 10 + count * 2 + int( clocked ) );
                    auto c = base();
                    c.settings.mode = PrimaryMode::Axis;
                    c.settings.axisDirection = AxisDirection::Same;
                    c.settings.coaxial = coaxial;
                    c.settings.matchAxialCenters = matchCenters;
                    for ( int i = 0; i < count; ++i )
                    {
                        const Vector3d point{ 0, 0, i == 0 ? 2. : 5. };
                        c.centers.push_back( { point, point } );
                    }
                    if ( clocked )
                        c.clocking = { { Vector3d::plusX(), Vector3d::plusY() } };
                    const auto result = solve( c );
                    ASSERT_TRUE( result ) << result.error();
                    EXPECT_TRUE( result->acceptable() );
                    // A point on the axis has zero radial lever arm, so only clocking can remove spin.
                    const int freeTranslations = matchCenters || count > 0 ? 0 : coaxial ? 1 : 3;
                    EXPECT_EQ( result->remainingDof, freeTranslations + ( clocked ? 0 : 1 ) );
                    expectVectorNear( result->delta.A * Vector3d::plusZ(), Vector3d::plusZ(), 1e-12 );
                    expectVectorNear( result->delta.A * Vector3d::plusX(), clocked ? Vector3d::plusY() : Vector3d::plusX(), 1e-12 );
                    for ( const auto& pair : c.centers )
                        expectVectorNear( result->delta( pair.moving ), pair.fixed, 1e-12 );
                }
}

TEST( MRMesh, FeatureAlignmentFullyLocatedRoleReversalRecoversInversePlacement )
{
    std::mt19937 generator( 98221 );
    for ( int iteration = 0; iteration < 30; ++iteration )
        for ( const auto mode : { PrimaryMode::Plane, PrimaryMode::Axis, PrimaryMode::Centers } )
        {
            SCOPED_TRACE( iteration );
            SCOPED_TRACE( int( mode ) );
            const auto desired = randomRigidFrame( generator );
            auto c = base();
            c.settings.mode = mode;
            c.settings.axisDirection = AxisDirection::Same;
            const double gap = iteration % 2 ? 5. : -5.;
            if ( mode == PrimaryMode::Plane )
            {
                c.settings.offset = gap;
                c.settings.normals = iteration % 3 ? NormalRelation::Same : NormalRelation::Opposing;
                const auto normal = double( int( c.settings.normals ) ) * ( desired.A * Vector3d::plusZ() );
                c.fixedPlane = { desired( {} ) - gap * normal, normal };
                for ( const auto point : { Vector3d{ -10, 0, 0 }, Vector3d{ 10, 0, 0 } } )
                    c.holes.push_back( { h( point ), h( desired( point ) - gap * normal, desired.A * Vector3d::plusZ() ) } );
            }
            else if ( mode == PrimaryMode::Axis )
            {
                c.settings.offset = gap;
                const auto normal = desired.A * Vector3d::plusZ();
                c.fixedAxis = { desired( {} ) - gap * normal, normal };
                c.centers.push_back( { { 10, 0, 2 }, desired( { 10, 0, 2 } ) - gap * normal } );
            }
            else
                for ( const auto point : { Vector3d{}, Vector3d{ 10, 0, 0 }, Vector3d{ 0, 6, 2 } } )
                    c.centers.push_back( { point, desired( point ) } );
            const auto forward = solve( c );
            ASSERT_TRUE( forward ) << forward.error();
            EXPECT_TRUE( forward->acceptable() );
            expectVectorNear( forward->delta( { 3, 4, 5 } ), desired( { 3, 4, 5 } ), 1e-8 );
            std::swap( c.movingPlane, c.fixedPlane );
            std::swap( c.movingAxis, c.fixedAxis );
            for ( auto& pair : c.holes )
                std::swap( pair.moving, pair.fixed );
            for ( auto& pair : c.centers )
                std::swap( pair.moving, pair.fixed );
            c.settings.offset = mode == PrimaryMode::Plane ? -double( int( c.settings.normals ) ) * gap
                                  : mode == PrimaryMode::Axis ? -gap : 0.;
            const auto reverse = solve( c );
            ASSERT_TRUE( reverse ) << reverse.error();
            EXPECT_TRUE( reverse->acceptable() );
            EXPECT_EQ( reverse->remainingDof, 0 );
            expectVectorNear( reverse->delta( desired( { 3, 4, 5 } ) ), { 3, 4, 5 }, 1e-8 );
            expectVectorNear( reverse->delta( forward->delta( { -2, 9, 7 } ) ), { -2, 9, 7 }, 1e-8 );
        }
}

TEST( MRMesh, FeatureAlignmentHoleAxisConflictsNeverRelaxPlaneTiltAndCentersModeIgnoresTheirAngle )
{
    for ( const auto mode : { HoleMode::Centers, HoleMode::Axes } )
    {
        auto c = base();
        c.settings.holeMode = mode;
        c.holes = { { h( {} ), h( {} ) }, { h( { 10, 0, 0 } ), h( { 10, 0, 0 }, { 1, 0, 1 } ) } };
        const auto result = solve( c );
        ASSERT_TRUE( result ) << result.error();
        expectProperRigid( result->delta );
        expectVectorNear( result->delta.A * Vector3d::plusZ(), Vector3d::plusZ(), 1e-12 );
        EXPECT_NEAR( dot( result->delta( c.movingPlane.center ) - c.fixedPlane.center, c.fixedPlane.normal ), 0, 1e-12 );
        EXPECT_NEAR( result->holes[1].position, 0, 1e-12 );
        EXPECT_NEAR( result->holes[1].axisAngleDeg, 45, 1e-10 );
        EXPECT_EQ( result->acceptable(), mode == HoleMode::Centers );
        EXPECT_EQ( result->holes[1].angleConstrained, mode == HoleMode::Axes );
    }
}

TEST( MRMesh, FeatureAlignmentLargeRepresentableCenterTranslationRemainsFinite )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 1e150, 0, 0 }, { -1e150, 0, 0 } } };
    const auto result = solve( c );
    ASSERT_TRUE( result ) << result.error();
    EXPECT_TRUE( result->acceptable() );
    expectProperRigid( result->delta );
    expectFiniteReport( *result );
    EXPECT_EQ( result->remainingDof, 3 );
    EXPECT_DOUBLE_EQ( result->delta.b.x, -2e150 );
    EXPECT_DOUBLE_EQ( result->delta( c.centers[0].moving ).x, c.centers[0].fixed.x );
    EXPECT_NEAR( result->holes[0].rawCenterDistance, 0, 1e-12 );
}

TEST( MRMesh, FeatureAlignmentNumericalAxisProjectionUsabilityIsIndependentOfTolerance )
{
    struct Case
    {
        double projection;
        int dof;
        double yaw;
    };
    for ( const auto test : { Case{ 0, 1, 0 }, Case{ 1e-12, 1, 0 }, Case{ 1e-8, 0, 90 }, Case{ .5, 0, 90 } } )
        for ( const double tolerance : { .1, 5., 60. } )
        {
            SCOPED_TRACE( test.projection );
            SCOPED_TRACE( tolerance );
            auto c = base();
            c.settings.angularToleranceDeg = tolerance;
            c.holes = { { h( {}, { test.projection, 0, 1 } ), h( {}, { 0, test.projection, 1 } ) } };
            const auto result = solve( c );
            ASSERT_TRUE( result ) << result.error();
            EXPECT_EQ( result->remainingDof, test.dof );
            EXPECT_NEAR( result->yawDeg, test.yaw, 1e-10 );
            expectProperRigid( result->delta );
            expectVectorNear( result->delta.A * Vector3d::plusZ(), Vector3d::plusZ(), 1e-12 );
        }
}

TEST( MRMesh, FeatureAlignmentSmallUsableObliqueAxisStillDeterminesExactQuarterTurn )
{
    for ( const double tiltDeg : { .5, .01, .001 } )
        for ( const double tolerance : { .0001, .1, 30. } )
        {
            SCOPED_TRACE( tiltDeg );
            SCOPED_TRACE( tolerance );
            const double tilt = tiltDeg * PI / 180.;
            auto c = base();
            c.settings.angularToleranceDeg = tolerance;
            c.holes = { { h( {}, { std::sin( tilt ), 0, std::cos( tilt ) } ),
                          h( { 4, 7, 0 }, { 0, std::sin( tilt ), std::cos( tilt ) } ) } };
            const auto result = solve( c );
            ASSERT_TRUE( result ) << result.error();
            EXPECT_EQ( result->remainingDof, 0 );
            EXPECT_NEAR( result->yawDeg, 90, 1e-10 );
            expectVectorNear( result->delta( { 10, 0, 0 } ), { 4, 17, 0 }, 1e-10 );
            expectVectorNear( result->delta.A * c.holes[0].moving.axis, c.holes[0].fixed.axis, 1e-12 );
            EXPECT_TRUE( result->acceptable() );
        }
}

TEST( MRMesh, FeatureAlignmentTinyClockAndAxisAnglesCannotFalselyPassTolerance )
{
    const double angleRad = 1e-9;
    const double angleDeg = angleRad * 180 / PI;
    auto c = base();
    c.settings.angularToleranceDeg = 1e-9;
    c.clocking = { { Vector3d::plusX(), Vector3d::plusX() } };
    const auto clock = measure( c, { Matrix3d::rotation( Vector3d::plusZ(), angleRad ), {} } );
    ASSERT_TRUE( clock ) << clock.error();
    ASSERT_TRUE( clock->clockingErrorDeg );
    EXPECT_NEAR( *clock->clockingErrorDeg / angleDeg, 1, 1e-12 );
    EXPECT_FALSE( clock->acceptable() );
    c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection::Same;
    c.settings.angularToleranceDeg = 1e-9;
    const auto axis = measure( c, { Matrix3d::rotation( Vector3d::plusX(), angleRad ), {} } );
    ASSERT_TRUE( axis ) << axis.error();
    EXPECT_NEAR( axis->primaryAngleErrorDeg / angleDeg, 1, 1e-12 );
    EXPECT_FALSE( axis->acceptable() );
}

TEST( MRMesh, FeatureAlignmentAngularMetricsPreserveRepresentableTinyAnglesAndUnorientedSigns )
{
    for ( const double radians : { 1e-7, 1e-12, 1e-100, 1e-200 } )
    {
        SCOPED_TRACE( radians );
        const double degrees = radians * 180 / PI;
        auto c = base();
        c.settings.angularToleranceDeg = degrees / 2;
        c.clocking = { { Vector3d::plusX(), Vector3d::plusX() } };
        const AffineXf3d clockPlacement{ Matrix3d::rotation( Vector3d::plusZ(), radians ), {} };
        auto result = measure( c, clockPlacement );
        ASSERT_TRUE( result ) << result.error();
        ASSERT_TRUE( result->clockingErrorDeg );
        EXPECT_NEAR( *result->clockingErrorDeg / degrees, 1, 1e-12 );
        EXPECT_FALSE( result->acceptable() );
        c.settings.angularToleranceDeg = degrees * 2;
        result = measure( c, clockPlacement );
        ASSERT_TRUE( result );
        EXPECT_TRUE( result->acceptable() );
        // Construct a near-antiparallel proper rotation directly: no subtraction from pi loses the tiny sine.
        const double sine = std::sin( radians ), cosine = std::cos( radians );
        const AffineXf3d nearOpposite{ Matrix3d{ { 1, 0, 0 }, { 0, -cosine, -sine }, { 0, sine, -cosine } }, {} };
        c = base();
        c.settings.mode = PrimaryMode::Axis;
        c.settings.angularToleranceDeg = degrees / 2;
        c.settings.axisDirection = AxisDirection::Opposing;
        result = measure( c, nearOpposite );
        ASSERT_TRUE( result ) << result.error();
        EXPECT_NEAR( result->primaryAngleErrorDeg / degrees, 1, 1e-12 );
        EXPECT_FALSE( result->acceptable() );
        c.settings.axisDirection = AxisDirection::Closest;
        result = measure( c, nearOpposite );
        ASSERT_TRUE( result );
        EXPECT_NEAR( result->primaryAngleErrorDeg / degrees, 1, 1e-12 );
        EXPECT_FALSE( result->acceptable() );
    }
}

TEST( MRMesh, FeatureAlignmentDirectedAndUnorientedAnglesDistinguishNearPi )
{
    for ( const double radians : { 1e-3, 1e-6, 1e-9 } )
    {
        SCOPED_TRACE( radians );
        const double sine = std::sin( radians ), cosine = std::cos( radians );
        const double degrees = radians * 180 / PI;
        const AffineXf3d placement{ Matrix3d{ { 1, 0, 0 }, { 0, -cosine, -sine }, { 0, sine, -cosine } }, {} };
        auto c = base();
        c.settings.mode = PrimaryMode::Axis;
        c.settings.axisDirection = AxisDirection::Same;
        auto result = measure( c, placement );
        ASSERT_TRUE( result ) << result.error();
        EXPECT_NEAR( result->primaryAngleErrorDeg, 180 - degrees, 1e-11 );
        c.settings.axisDirection = AxisDirection::Opposing;
        result = measure( c, placement );
        ASSERT_TRUE( result );
        EXPECT_NEAR( result->primaryAngleErrorDeg / degrees, 1, 1e-12 );
        c = base();
        c.holes = { { h( {} ), h( {} ) } };
        result = measure( c, placement );
        ASSERT_TRUE( result );
        ASSERT_EQ( result->holes.size(), 1 );
        EXPECT_NEAR( result->normalErrorDeg, 180 - degrees, 1e-11 );
        EXPECT_NEAR( result->holes[0].axisAngleDeg / degrees, 1, 1e-12 );
    }
}

TEST( MRMesh, FeatureAlignmentClosestAxisRanksFiniteScoresWithoutSquaringOverflow )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection::Closest;
    c.settings.linearTolerance = 1e-10;
    c.centers = { { { 1, 0, 1e150 }, { 2, 0, -2e150 } } };
    const auto result = solve( c );
    ASSERT_TRUE( result ) << result.error();
    EXPECT_NEAR( result->maximumPositionError() / 1e150, 1, 1e-12 );
    EXPECT_LT( dot( result->delta.A * c.movingAxis.direction, c.fixedAxis.direction ), -.99 );
    EXPECT_FALSE( result->acceptable() );
    expectProperRigid( result->delta );
    expectFiniteReport( *result );
}

TEST( MRMesh, FeatureAlignmentClosestAxisScoreOrderingIsStableAcrossFiniteMagnitudeAndTolerance )
{
    for ( const double magnitude : { 1e120, 1e140, 1e150, 1e153 } )
        for ( const double tolerance : { 1e-10, .1, 1., 1e100 } )
            for ( const double fixedSign : { -1., 1. } )
            {
                SCOPED_TRACE( magnitude );
                SCOPED_TRACE( tolerance );
                SCOPED_TRACE( fixedSign );
                auto c = base();
                c.settings.mode = PrimaryMode::Axis;
                c.settings.axisDirection = AxisDirection::Closest;
                c.settings.linearTolerance = tolerance;
                c.centers = { { { 1, 0, magnitude }, { 2, 0, fixedSign * 2 * magnitude } } };
                const auto result = solve( c );
                ASSERT_TRUE( result ) << result.error();
                // One sign leaves axial mismatch magnitude; the other leaves 3*magnitude. Primary center stays exact.
                EXPECT_NEAR( result->holes[0].position / magnitude, 1, 1e-12 );
                expectVectorNear( result->delta( {} ), {}, 1e-12 );
                EXPECT_NEAR( dot( result->delta.A * c.movingAxis.direction, c.fixedAxis.direction ), fixedSign, 1e-12 );
                expectProperRigid( result->delta );
                expectFiniteReport( *result );
            }
}

TEST( MRMesh, FeatureAlignmentClosestAxisPlacementIsIndependentOfAcceptanceTolerances )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection::Closest;
    c.centers = { { { 1, 0, 1e-5 }, { 1, 0, -1e-5 } } };
    for ( const bool clocking : { false, true } )
        for ( const double positionTolerance : { 0., 1e-6, .01, 1., 1e100 } )
            for ( const double angleTolerance : { 0., .01, 1., 89. } )
            {
                SCOPED_TRACE( clocking );
                SCOPED_TRACE( positionTolerance );
                SCOPED_TRACE( angleTolerance );
                c.settings.linearTolerance = positionTolerance;
                c.settings.angularToleranceDeg = angleTolerance;
                if ( clocking )
                    c.clocking = { { { 1, .001, 0 }, { 1, .001, 0 } } };
                const auto result = solve( c );
                ASSERT_TRUE( result ) << result.error();
                // A half-turn about X exactly matches the locator. Identity leaves a 2e-5 gap.
                // Even when only identity passes the clocking limit, acceptance must not select it.
                expectVectorNear( result->delta.A * Vector3d::plusX(), { 1, 0, 0 }, 1e-12 );
                expectVectorNear( result->delta.A * Vector3d::plusY(), { 0, -1, 0 }, 1e-12 );
                expectVectorNear( result->delta.A * Vector3d::plusZ(), { 0, 0, -1 }, 1e-12 );
                expectVectorNear( result->delta.b, {}, 1e-12 );
                expectVectorNear( result->delta( c.centers[0].moving ), c.centers[0].fixed, 1e-12 );
                if ( clocking && positionTolerance >= 1e-6 )
                    EXPECT_EQ( result->acceptable(), angleTolerance >= 1. );
            }
}

TEST( MRMesh, FeatureAlignmentClosestAxisUsesClockingThenRotationForPositionTies )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.centers = { { { 1, 0, 0 }, { 1, 0, 0 } } };
    for ( const double angleTolerance : { .01, 1., 89. } )
    {
        c.settings.angularToleranceDeg = angleTolerance;
        c.clocking = { { { 1, .001, 0 }, { 1, -.001, 0 } } };
        const auto clocked = solve( c );
        ASSERT_TRUE( clocked ) << clocked.error();
        // Both signs match the locator, but only the X half-turn matches the directed clocking.
        expectVectorNear( clocked->delta.A * Vector3d::plusZ(), { 0, 0, -1 }, 1e-12 );
        ASSERT_TRUE( clocked->clockingErrorDeg );
        EXPECT_NEAR( *clocked->clockingErrorDeg, 0, 1e-12 );
        c.clocking.reset();
        const auto unclocked = solve( c );
        ASSERT_TRUE( unclocked ) << unclocked.error();
        // With equal geometric residuals, identity is the smaller physical rotation.
        expectVectorNear( unclocked->delta.A * Vector3d::plusX(), { 1, 0, 0 }, 1e-12 );
        expectVectorNear( unclocked->delta.A * Vector3d::plusY(), { 0, 1, 0 }, 1e-12 );
        expectVectorNear( unclocked->delta.A * Vector3d::plusZ(), { 0, 0, 1 }, 1e-12 );
    }
}

namespace
{
void expectDiagnostics( const std::vector<Diagnostic> &actual, std::vector<Diagnostic> expected )
{
    EXPECT_EQ( actual, expected );
}
} // namespace

TEST( MRMesh, FeatureAlignmentTypedDiagnosticDefaultsAndEquality )
{
    const Diagnostic unspecified;
    EXPECT_EQ( unspecified.code, DiagnosticCode::Unspecified );
    EXPECT_EQ( unspecified.index, -1 );
    const Diagnostic hole{ DiagnosticCode::HolePositionMismatch, 1 };
    EXPECT_EQ( hole, ( Diagnostic{ DiagnosticCode::HolePositionMismatch, 1 } ) );
    EXPECT_NE( hole, ( Diagnostic{ DiagnosticCode::HolePositionMismatch, 0 } ) );
    EXPECT_NE( hole, ( Diagnostic{ DiagnosticCode::HoleAxisAngleMismatch, 1 } ) );
    const Solution unpopulated;
    EXPECT_EQ( unpopulated.rotationSource, RotationSource::Unconstrained );
    EXPECT_EQ( unpopulated.freeMotion, FreeMotion::Unspecified );
    EXPECT_TRUE( unpopulated.conflicts.empty() );
    EXPECT_TRUE( unpopulated.warnings.empty() );
}

TEST( MRMesh, FeatureAlignmentTypedPlaneDiagnosticsPreserveInputIndices )
{
    auto c = base();
    c.settings.linearTolerance = .01;
    c.settings.angularToleranceDeg = .1;
    c.holes = { { h( {} ), h( {} ) }, { h( { 10, 0, 0 } ), h( { 11, 0, 0 }, { .2, 0, 1 } ) } };
    c.holes[0].fixed.radius = 2;
    c.holes[1].fixed.radius = 3;
    const auto measured = measure( c, {} );
    ASSERT_TRUE( measured ) << measured.error();
    expectDiagnostics( measured->conflicts, { { DiagnosticCode::HolePositionMismatch, 1 },
                                              { DiagnosticCode::HoleAxisAngleMismatch, 1 } } );
    expectDiagnostics( measured->warnings, { { DiagnosticCode::HoleRadiusMismatch, 0 },
                                             { DiagnosticCode::HoleRadiusMismatch, 1 } } );
    EXPECT_EQ( measured->rotationSource, RotationSource::SecondHole );
    EXPECT_EQ( measured->freeMotion, FreeMotion::FullyConstrained );
    EXPECT_EQ( measured->remainingDof, 0 );
    EXPECT_FALSE( measured->acceptable() );
    ASSERT_EQ( measured->holes.size(), 2 );
    EXPECT_NEAR( measured->holes[1].position, 1, 1e-12 );
    EXPECT_NEAR( measured->holes[1].axisAngleDeg, std::atan( .2 ) * 180. / std::numbers::pi, 1e-12 );

    const auto displaced = measure( c, AffineXf3d::translation( { 0, 0, 2 } ) );
    ASSERT_TRUE( displaced ) << displaced.error();
    expectDiagnostics( displaced->conflicts, { { DiagnosticCode::PlaneSeatingMismatch },
                                               { DiagnosticCode::HolePositionMismatch, 1 },
                                               { DiagnosticCode::HoleAxisAngleMismatch, 1 } } );

    c.settings.holeMode = HoleMode::Centers;
    c.settings.balanceHoles = true;
    const auto balanced = solve( c );
    ASSERT_TRUE( balanced ) << balanced.error();
    expectDiagnostics( balanced->conflicts, { { DiagnosticCode::HolePositionMismatch, 0 },
                                             { DiagnosticCode::HolePositionMismatch, 1 } } );
    expectDiagnostics( balanced->warnings, { { DiagnosticCode::HoleRadiusMismatch, 0 },
                                            { DiagnosticCode::HoleRadiusMismatch, 1 },
                                            { DiagnosticCode::PlaneCentersProjected },
                                            { DiagnosticCode::BalancedHoleFit } } );
    EXPECT_EQ( balanced->freeMotion, FreeMotion::FullyConstrained );
    EXPECT_FALSE( balanced->acceptable() );
    EXPECT_NEAR( balanced->planeOffsetError, 0, 1e-12 );
    EXPECT_NEAR( balanced->holes[0].position, .5, 1e-12 );
    EXPECT_NEAR( balanced->holes[1].position, .5, 1e-12 );
}

TEST( MRMesh, FeatureAlignmentTypedPlaneMotionAndConventionReasons )
{
    auto c = base();
    auto result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::Unconstrained );
    EXPECT_EQ( result->freeMotion, FreeMotion::PlaneTranslationsAndSpin );
    expectDiagnostics( result->warnings, { { DiagnosticCode::Underconstrained } } );
    c.clocking = { { Vector3d::plusX(), Vector3d::plusY() } };
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::DirectedReference );
    EXPECT_EQ( result->freeMotion, FreeMotion::PlaneTranslations );
    expectVectorNear( result->delta.A * Vector3d::plusX(), Vector3d::plusY() );
    c.clocking.reset();
    c.holes = { { h( {} ), h( {} ) } };
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->freeMotion, FreeMotion::PrimarySpin );
    expectDiagnostics( result->warnings, { { DiagnosticCode::Underconstrained } } );
    c = base();
    c.settings.normals = NormalRelation::Opposing;
    result = solve( c );
    ASSERT_TRUE( result );
    expectDiagnostics( result->warnings, { { DiagnosticCode::Underconstrained },
                                           { DiagnosticCode::OppositePlaneNormalsConvention } } );
    c = base();
    c.holes = { { h( {}, { 1, 0, 1 } ), h( {}, { 0, 1, 1 } ) } };
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::ObliquePrimaryAxis );
    EXPECT_EQ( result->freeMotion, FreeMotion::FullyConstrained );
    expectDiagnostics( result->warnings, { { DiagnosticCode::ObliqueHoleAxis } } );
    expectVectorNear( result->delta.A * Vector3d::plusX(), Vector3d::plusY() );

    c = base();
    c.holes = { { h( {} ), h( {} ) }, { h( { 10, 0, 0 } ), h( { 10, 0, 0 } ) } };
    c.clocking = { { Vector3d::plusX(), Vector3d::plusY() } };
    result = solve( c );
    ASSERT_TRUE( result );
    expectDiagnostics( result->conflicts, { { DiagnosticCode::ClockingConflict } } );
    EXPECT_EQ( result->freeMotion, FreeMotion::FullyConstrained );
    EXPECT_FALSE( result->acceptable() );
}

TEST( MRMesh, FeatureAlignmentTypedAxisMotionAndRotationSources )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection::Same;
    auto result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::Unconstrained );
    EXPECT_EQ( result->freeMotion, FreeMotion::PrimarySpin );
    expectDiagnostics( result->warnings, { { DiagnosticCode::Underconstrained },
                                           { DiagnosticCode::FittedAxialCenters } } );
    c.settings.matchAxialCenters = false;
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->freeMotion, FreeMotion::PrimarySlideAndSpin );
    c.clocking = { { Vector3d::plusX(), Vector3d::plusX() } };
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::DirectedReference );
    EXPECT_EQ( result->freeMotion, FreeMotion::PrimarySlide );
    c.settings.coaxial = false;
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->freeMotion, FreeMotion::Translations );
    c.clocking.reset();
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->freeMotion, FreeMotion::TranslationsAndSpin );
    c.centers = { { {}, {} } };
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::Unconstrained );
    EXPECT_EQ( result->freeMotion, FreeMotion::PrimarySpin );
    c.centers.push_back( { { 10, 0, 0 }, { 0, 10, 0 } } );
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::OffAxisCenterLocator );
    EXPECT_EQ( result->freeMotion, FreeMotion::FullyConstrained );
    expectVectorNear( result->delta.A * Vector3d::plusX(), Vector3d::plusY() );
    expectDiagnostics( result->warnings, {} );
}

TEST( MRMesh, FeatureAlignmentTypedAxisConflictsRetainDistinctReasons )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection::Same;
    c.settings.linearTolerance = .01;
    c.settings.angularToleranceDeg = .1;
    c.centers = { { {}, {} }, { { 10, 0, 0 }, { 10, 0, 0 } } };
    auto result = measure( c, AffineXf3d::translation( { 2, 0, 3 } ) );
    ASSERT_TRUE( result );
    expectDiagnostics( result->conflicts, { { DiagnosticCode::PrimaryCenterMismatch },
                                            { DiagnosticCode::CenterPositionMismatch, 0 },
                                            { DiagnosticCode::CenterPositionMismatch, 1 } } );
    EXPECT_EQ( result->freeMotion, FreeMotion::FullyConstrained );
    c.centers.clear();
    c.settings.matchAxialCenters = false;
    result = measure( c, AffineXf3d::translation( { 2, 0, 0 } ) );
    ASSERT_TRUE( result );
    expectDiagnostics( result->conflicts, { { DiagnosticCode::PrimaryAxisNotCoaxial } } );
    const AffineXf3d tilted{ Matrix3d::rotation( Vector3d::plusY(), -std::numbers::pi / 2 ), {} };
    c.clocking = { { Vector3d::plusX(), Vector3d::plusX() } };
    result = measure( c, tilted );
    ASSERT_TRUE( result );
    expectDiagnostics( result->conflicts, { { DiagnosticCode::PrimaryAxisAngleMismatch },
                                            { DiagnosticCode::ClockingParallelToPrimary } } );
    EXPECT_EQ( result->delta, tilted );
    c.clocking = { { Vector3d::plusX(), Vector3d::plusY() } };
    result = measure( c, {} );
    ASSERT_TRUE( result );
    expectDiagnostics( result->conflicts, { { DiagnosticCode::ClockingConflict } } );
}

TEST( MRMesh, FeatureAlignmentTypedAxisAmbiguityWarningRetainsRank )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.matchAxialCenters = false;
    c.centers = { { { 0, 0, 2 }, { 0, 0, 3 } } };
    const auto result = solve( c );
    ASSERT_TRUE( result ) << result.error();
    EXPECT_TRUE( result->acceptable() );
    EXPECT_EQ( result->rotationSource, RotationSource::Unconstrained );
    EXPECT_EQ( result->freeMotion, FreeMotion::PrimarySpin );
    EXPECT_EQ( result->remainingDof, 1 );
    expectDiagnostics( result->warnings, { { DiagnosticCode::Underconstrained },
                                           { DiagnosticCode::AmbiguousAxisDirection } } );
    expectVectorNear( result->delta( { 0, 0, 2 } ), { 0, 0, 3 } );
}

TEST( MRMesh, FeatureAlignmentTypedCenterReasonsAndIndicesIncludePrimary )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Centers;
    c.settings.linearTolerance = .01;
    c.centers = { { {}, {} } };
    auto result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::Unconstrained );
    EXPECT_EQ( result->freeMotion, FreeMotion::Rotations );
    expectDiagnostics( result->warnings, { { DiagnosticCode::Underconstrained } } );
    c.centers.push_back( { { 10, 0, 0 }, { 10, 0, 0 } } );
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::SecondCenter );
    EXPECT_EQ( result->freeMotion, FreeMotion::BaselineSpin );
    c.centers.push_back( { { 20, 0, 0 }, { 20, 0, 0 } } );
    result = solve( c );
    ASSERT_TRUE( result );
    expectDiagnostics( result->warnings, { { DiagnosticCode::Underconstrained },
                                           { DiagnosticCode::CollinearCenters } } );
    c.centers.back() = { { 0, 10, 0 }, { 0, 10, 0 } };
    result = solve( c );
    ASSERT_TRUE( result );
    EXPECT_EQ( result->rotationSource, RotationSource::ThirdCenter );
    EXPECT_EQ( result->freeMotion, FreeMotion::FullyConstrained );
    expectDiagnostics( result->warnings, {} );
    const AffineXf3d displaced = AffineXf3d::translation( { 2, 0, 0 } );
    result = measure( c, displaced );
    ASSERT_TRUE( result );
    expectDiagnostics( result->conflicts, { { DiagnosticCode::CenterPositionMismatch, 0 },
                                            { DiagnosticCode::CenterPositionMismatch, 1 },
                                            { DiagnosticCode::CenterPositionMismatch, 2 } } );
    EXPECT_FALSE( result->acceptable() );
    EXPECT_EQ( result->freeMotion, FreeMotion::FullyConstrained );
    EXPECT_EQ( result->delta, displaced );
    EXPECT_NEAR( result->maximumPositionError(), 2, 1e-12 );
}

TEST( MRMesh, FeatureAlignmentPositionResidualsRetainFiniteExtremeMagnitudes )
{
    for ( const double scale : { 1e-200, 1e200 } )
        for ( const auto mode : { PrimaryMode::Plane, PrimaryMode::Axis, PrimaryMode::Centers } )
        {
            SCOPED_TRACE( ::testing::Message() << "scale=" << scale << " mode=" << int( mode ) );
            auto c = base();
            c.settings.mode = mode;
            c.settings.axisDirection = AxisDirection::Same;
            c.settings.holeMode = HoleMode::Centers;
            c.settings.linearTolerance = 4 * scale;
            if ( mode == PrimaryMode::Plane )
                c.holes = { { h( {} ), h( {} ) } };
            else if ( mode == PrimaryMode::Centers )
                c.centers = { { {}, {} } };
            const auto measured = measure( c, AffineXf3d::translation( { 3 * scale, 4 * scale, 0 } ) );
            ASSERT_TRUE( measured ) << measured.error();
            EXPECT_FALSE( measured->acceptable() );
            EXPECT_NEAR( measured->maximumPositionError() / scale, 5, 1e-12 );
            if ( !measured->holes.empty() )
                EXPECT_NEAR( measured->holes.front().rawCenterDistance / scale, 5, 1e-12 );
            if ( mode == PrimaryMode::Axis )
            {
                c.settings.matchAxialCenters = false;
                const auto radial = measure( c, AffineXf3d::translation( { 3 * scale, 4 * scale, 0 } ) );
                ASSERT_TRUE( radial ) << radial.error();
                EXPECT_FALSE( radial->acceptable() );
                EXPECT_NEAR( radial->primaryPositionError / scale, 5, 1e-12 );
            }
        }
}

TEST( MRMesh, FeatureAlignmentShortestRotationRetainsRepresentableTinyTilt )
{
    constexpr double cTilt = 1e-200;
    for ( const auto mode : { PrimaryMode::Plane, PrimaryMode::Axis, PrimaryMode::Centers } )
    {
        SCOPED_TRACE( int( mode ) );
        auto c = base();
        c.settings.mode = mode;
        c.settings.axisDirection = AxisDirection::Same;
        c.settings.angularToleranceDeg = cTilt * 180 / std::numbers::pi / 2;
        c.settings.linearTolerance = 1e-210;
        if ( mode == PrimaryMode::Plane )
            c.fixedPlane.normal = { cTilt, 0, 1 };
        else if ( mode == PrimaryMode::Axis )
            c.fixedAxis.direction = { cTilt, 0, 1 };
        else
            c.centers = { { {}, {} }, { { 0, 0, 1 }, { cTilt, 0, 1 } } };
        const auto result = solve( c );
        ASSERT_TRUE( result ) << result.error();
        EXPECT_TRUE( result->acceptable() );
        EXPECT_NEAR( ( result->delta.A * Vector3d::plusZ() ).x / cTilt, 1, 1e-12 );
        EXPECT_TRUE( isRigid( result->delta ) );
    }
}

} // namespace MR
