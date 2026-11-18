#include <MRMesh/MRFeatureAlignment.h>
#include <MRMesh/MRPlaneObject.h>
#include <MRMesh/MRCircleObject.h>
#include <MRMesh/MRCylinderObject.h>
#include <MRMesh/MRPointObject.h>
#include <MRMesh/MRSphereObject.h>
#include <MRMesh/MRLineObject.h>
#include <gtest/gtest.h>
#include <algorithm>
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

void expectDiagnostics( const std::vector<Diagnostic> &actual, std::vector<Diagnostic> expected )
{
    EXPECT_EQ( actual, expected );
}
} // namespace

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
            EXPECT_NEAR( r.delta( c.movingPlane.center ).z, gap, 1e-8 );
            expectProperRigid( r.delta );
        }
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
    EXPECT_NEAR( r.delta( c.movingPlane.center ).z, 10, 1e-8 );
    expectProperRigid( r.delta );
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
    expectProperRigid( r.delta );
}

TEST( MRMesh, FeatureAlignmentNoisyPerpendicularFittedHoleLeavesSpin )
{
    FeatureReferences references;
    references.primary = { std::make_shared<PlaneObject>(), std::make_shared<PlaneObject>() };
    const auto moving = std::make_shared<CircleObject>();
    const auto fixed = std::make_shared<CircleObject>();
    const float cTilt = .01f * std::numbers::pi_v<float> / 180;
    moving->setNormal( { std::sin( cTilt ), 0, std::cos( cTilt ) } );
    references.locators[0] = { moving, fixed };
    for ( const float azimuth : { -90.f, 47.f, 90.f, 180.f } )
    {
        SCOPED_TRACE( azimuth );
        const float radians = azimuth * std::numbers::pi_v<float> / 180;
        fixed->setNormal( { std::sin( cTilt ) * std::cos( radians ),
                           std::sin( cTilt ) * std::sin( radians ), std::cos( cTilt ) } );
        for ( const double tolerance : { 1., 5. } )
        {
            auto settings = base().settings;
            settings.angularToleranceDeg = tolerance;
            const auto constraints = readConstraints( references, settings );
            ASSERT_TRUE( constraints ) << constraints.error();
            const auto solved = solve( *constraints );
            ASSERT_TRUE( solved ) << solved.error();
            EXPECT_TRUE( solved->acceptable() );
            EXPECT_EQ( solved->remainingDof, 1 );
            EXPECT_EQ( solved->freeMotion, FreeMotion::PrimarySpin );
            EXPECT_EQ( solved->rotationSource, RotationSource::Unconstrained );
            EXPECT_NEAR( solved->yawDeg, 0, 1e-6 );
            expectVectorNear( solved->delta.A * Vector3d::plusX(), Vector3d::plusX(), 1e-6 );
        }
    }
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
    ASSERT_TRUE( measure( base(), x ) );
    x.A.x.x = 2;
    EXPECT_FALSE( measure( base(), x ) );
    x.A.x.x = -1;
    EXPECT_FALSE( measure( base(), x ) );
    x = {};
    x.A.x.y = .1;
    EXPECT_FALSE( measure( base(), x ) ) << "shear";
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

TEST( MRMesh, FeatureAlignmentCenterOnlyMismatchNeverScales )
{
    auto c = base();
    c.settings.linearTolerance = .05;
    c.settings.mode = PrimaryMode::Centers;
    c.centers = { { { 0, 0, 0 }, { 0, 0, 0 } }, { { 10, 0, 0 }, { 0, 11, 0 } } };
    const auto solved = solve( c );
    ASSERT_TRUE( solved ) << solved.error();
    const auto& r = *solved;
    EXPECT_FALSE( r.acceptable() );
    expectProperRigid( r.delta );
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
    EXPECT_NEAR( pinned->delta( {} ).z, -7, 1e-6 );
    EXPECT_NEAR( pinned->holes[0].position, 0, 1e-9 );
    EXPECT_NEAR( pinned->holes[1].position, 2, 1e-9 );
    EXPECT_FALSE( pinned->acceptable() );
    settings.balanceHoles = true;
    const auto balanced = solveReferences( refs, settings );
    ASSERT_TRUE( balanced ) << balanced.error();
    EXPECT_NEAR( balanced->delta( {} ).z, -7, 1e-6 );
    EXPECT_NEAR( balanced->holes[0].position, 1, 1e-9 );
    EXPECT_NEAR( balanced->holes[1].position, 1, 1e-9 );
    EXPECT_EQ( balanced->remainingDof, 0 );
    expectProperRigid( balanced->delta );
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
    EXPECT_NEAR( one->delta( {} ).z, 0, 1e-6 );
    EXPECT_NEAR( one->holes[0].position, 0, 1e-6 );
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
    expectVectorNear( result->delta( { 1, 2, 3 } ), { 10, 20, 30 }, 1e-5 );
    EXPECT_NEAR( result->primaryAngleErrorDeg, 0, 1e-6 );
    EXPECT_EQ( result->remainingDof, 0 );
    expectProperRigid( result->delta );
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

TEST( MRMesh, FeatureAlignmentFeatureAncestryMustBeRigidButDisplayExtentsDoNotConstrainPlacement )
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
    EXPECT_TRUE( solveReferences( refs, base().settings ) );
    auto plane = fittedPlane();
    plane->setSizeX( 1e-12f );
    ASSERT_GT( plane->xf().A.det(), 0 );
    refs = planeReferences();
    refs.primary.moving = plane;
    EXPECT_TRUE( solveReferences( refs, base().settings ) );
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
        EXPECT_NEAR( result->maximumPositionError(), std::sqrt( 29. ), 1e-8 );
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

TEST( MRMesh, FeatureAlignmentObliqueAxesSetQuarterTurnIndependentlyOfTolerance )
{
    const double cTilt = 10 * std::numbers::pi / 180;
    const Vector3d cMoving{ std::sin( cTilt ), 0, std::cos( cTilt ) };
    const Vector3d cFixed{ 0, std::sin( cTilt ), std::cos( cTilt ) };
    for ( const double sign : { -1., 1. } )
        for ( const double tolerance : { .25, 5., 20., 60. } )
        {
            SCOPED_TRACE( sign );
            SCOPED_TRACE( tolerance );
            auto c = base();
            c.holes = { { h( {}, cMoving ), h( { 4, 7, 0 }, sign * cFixed ) } };
            c.settings.angularToleranceDeg = tolerance;
            const auto result = solve( c );
            ASSERT_TRUE( result ) << result.error();
            EXPECT_EQ( result->remainingDof, 0 );
            EXPECT_NEAR( result->yawDeg, 90, 1e-8 );
            expectVectorNear( result->delta.A * cMoving, cFixed, 1e-8 );
            EXPECT_NEAR( result->delta( {} ).z, 0, 1e-8 );
            EXPECT_TRUE( result->acceptable() );
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
    EXPECT_TRUE( std::any_of( result->conflicts.begin(), result->conflicts.end(), []( const Diagnostic& conflict )
    {
        return conflict == Diagnostic{ DiagnosticCode::ClockingParallelToPrimary };
    } ) );
    EXPECT_TRUE( std::isfinite( result->maximumAngleError() ) );
    EXPECT_EQ( result->delta, supplied );
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
        EXPECT_FALSE( measure( base(), transform ) );
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
    c = base();
    c.settings.mode = PrimaryMode( 9 );
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.clocking = { { Vector3d::plusZ(), Vector3d::plusX() } };
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.movingPlane.normal = {};
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.movingAxis.direction = {};
    EXPECT_FALSE( solve( c ) );
    c = base();
    c.holes = { { h( {} ), h( {} ) } };
    c.holes[0].moving.radius = -1;
    EXPECT_FALSE( solve( c ) );
    c.holes = { { h( {} ), h( {} ) }, { h( { 0, 0, 1 } ), h( {} ) } };
    EXPECT_FALSE( solve( c ) );
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
        EXPECT_NEAR( result->maximumAngleError(), mode == HoleMode::Axes ? 45 : 0, 1e-6 );
        EXPECT_EQ( result->acceptable(), mode == HoleMode::Centers );
    }
}

TEST( MRMesh, FeatureAlignmentMinimumObliqueTiltIsIndependentOfTolerance )
{
    struct Case
    {
        double projection;
        int dof;
        double yaw;
    };
    for ( const auto test : { Case{ 0, 1, 0 }, Case{ 1e-4, 1, 0 }, Case{ .01, 1, 0 }, Case{ .5, 0, 90 } } )
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
            const auto measured = measure( c, result->delta );
            ASSERT_TRUE( measured ) << measured.error();
            EXPECT_EQ( measured->remainingDof, result->remainingDof );
            EXPECT_EQ( measured->freeMotion, result->freeMotion );
            expectProperRigid( result->delta );
            expectVectorNear( result->delta.A * Vector3d::plusZ(), Vector3d::plusZ(), 1e-12 );
        }
}

TEST( MRMesh, FeatureAlignmentCallerDeclaredObliqueAxisDeterminesQuarterTurn )
{
    for ( const double tiltDeg : { .5, 10. } )
        for ( const double tolerance : { .0001, .1, 30. } )
        {
            SCOPED_TRACE( tiltDeg );
            SCOPED_TRACE( tolerance );
            const double tilt = tiltDeg * PI / 180.;
            auto c = base();
            c.settings.angularToleranceDeg = tolerance;
            c.settings.minimumObliqueAngleDeg = .25;
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

TEST( MRMesh, FeatureAlignmentMinimumObliqueTiltRejectsInvalidSettings )
{
    for ( const double minimumTilt : { -1., 90., std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN() } )
    {
        auto c = base();
        c.settings.minimumObliqueAngleDeg = minimumTilt;
        EXPECT_FALSE( solve( c ) );
        EXPECT_FALSE( measure( c, {} ) );
    }
}

TEST( MRMesh, FeatureAlignmentSmallClockAndAxisAnglesRespectTolerance )
{
    const double angleRad = 1e-7;
    const double angleDeg = angleRad * 180 / PI;
    auto c = base();
    c.settings.angularToleranceDeg = 1e-9;
    c.clocking = { { Vector3d::plusX(), Vector3d::plusX() } };
    const auto clock = measure( c, { Matrix3d::rotation( Vector3d::plusZ(), angleRad ), {} } );
    ASSERT_TRUE( clock ) << clock.error();
    EXPECT_NEAR( clock->maximumAngleError(), angleDeg, 1e-8 );
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

TEST( MRMesh, FeatureAlignmentClosestAxisPlacementIsIndependentOfAcceptanceTolerances )
{
    auto c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.settings.axisDirection = AxisDirection::Closest;
    c.centers = { { { 1, 0, 1e-5 }, { 1, 0, -1e-5 } } };
    for ( const bool clocking : { false, true } )
        for ( const double positionTolerance : { 0., 1e-6, .01, 1. } )
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
                {
                    EXPECT_EQ( result->acceptable(), angleTolerance >= 1. );
                }
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
        EXPECT_NEAR( clocked->maximumAngleError(), 0, 1e-8 );
        c.clocking.reset();
        const auto unclocked = solve( c );
        ASSERT_TRUE( unclocked ) << unclocked.error();
        // With equal geometric residuals, identity is the smaller physical rotation.
        expectVectorNear( unclocked->delta.A * Vector3d::plusX(), { 1, 0, 0 }, 1e-12 );
        expectVectorNear( unclocked->delta.A * Vector3d::plusY(), { 0, 1, 0 }, 1e-12 );
        expectVectorNear( unclocked->delta.A * Vector3d::plusZ(), { 0, 0, 1 }, 1e-12 );
    }
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
    EXPECT_NEAR( measured->maximumAngleError(), std::atan( .2 ) * 180. / std::numbers::pi, 1e-8 );

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
    EXPECT_NEAR( balanced->delta( {} ).z, 0, 1e-8 );
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

    c = base();
    c.settings.mode = PrimaryMode::Axis;
    c.fixedAxis.direction = Vector3d::plusX();
    c.clocking = { { Vector3d::plusY(), Vector3d::plusY() } };
    for ( int locatorCount : { 0, 1 } )
    {
        c.centers.resize( locatorCount );
        const auto clocked = solve( c );
        ASSERT_TRUE( clocked ) << clocked.error();
        EXPECT_TRUE( clocked->acceptable() );
        EXPECT_EQ( clocked->remainingDof, 0 );
        expectDiagnostics( clocked->warnings, { { DiagnosticCode::FittedAxialCenters },
                                                { DiagnosticCode::AmbiguousAxisDirection } } );
    }
}

TEST( MRMesh, FeatureAlignmentCollinearFittedCentersLeaveSpinAtDifferentScales )
{
    auto settings = base().settings;
    settings.mode = PrimaryMode::Centers;
    for ( const Vector3f v : { Vector3f{ 1, 3, 9 }, Vector3f{ 1, 2, 2 } } )
        for ( const float scale : { 1.f, 50000.f, 100000.f } )
        {
            SCOPED_TRACE( scale );
            const auto pair = []( Vector3f p ) -> FeaturePair
            {
                return { fittedPoint( p ), fittedPoint( { -p.y, p.x, p.z } ) };
            };
            FeatureReferences refs;
            refs.primary = pair( {} );
            refs.locators[0] = pair( scale * v );
            for ( const bool collinear : { true, false } )
            {
                refs.locators[1] = pair( 2.f * scale * v + ( collinear ? Vector3f{} : Vector3f::plusY() ) );
                const auto c = readConstraints( refs, settings );
                ASSERT_TRUE( c ) << c.error();
                const auto solved = solve( *c );
                ASSERT_TRUE( solved ) << solved.error();
                const auto measured = measure( *c, solved->delta );
                ASSERT_TRUE( measured ) << measured.error();
                for ( const auto *result : { &*solved, &*measured } )
                {
                    EXPECT_TRUE( result->acceptable() );
                    EXPECT_EQ( result->remainingDof, collinear ? 1 : 0 );
                    EXPECT_EQ( result->freeMotion, collinear ? FreeMotion::BaselineSpin : FreeMotion::FullyConstrained );
                }
            }
        }
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

} // namespace MR
