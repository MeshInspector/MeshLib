#include "MRFeatureAlignment.h"
#include "MRPointToPointAligningTransform.h"
#include "MRIntersection.h"
#include "MRTriMath.h"
#include "MRPlaneObject.h"
#include "MRCircleObject.h"
#include "MRCylinderObject.h"
#include "MRPointObject.h"
#include "MRSphereObject.h"
#include "MRLineObject.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <utility>

namespace MR::FeatureAlignment
{
namespace
{
constexpr double cEpsilon = 1e-10, cDegrees = 180. / std::numbers::pi;
bool finite( Vector3d v )
{
    return std::isfinite( v.x ) && std::isfinite( v.y ) && std::isfinite( v.z );
}
double stableLength( Vector3d v )
{
    return std::hypot( v.x, v.y, v.z );
}
bool normalize( Vector3d &v )
{
    const auto size = v.length();
    if ( !finite( v ) || !std::isfinite( size ) || size < cEpsilon )
        return false;
    v /= size;
    return true;
}
Vector3d project( Vector3d v, Vector3d n )
{
    return Plane3d{ n, 0 }.project( v );
}
Matrix3d shortestRotation( Vector3d a, Vector3d b )
{
    const auto axis = cross( a, b );
    const auto sine = stableLength( axis );
    if ( sine == 0 )
        return Matrix3d::rotation( a, b );
    // Normalize before the native axis-angle helper squares the cross product.
    return Matrix3d::rotation( axis / sine, std::atan2( sine, dot( a, b ) ) );
}
double angleDeg( Vector3d a, Vector3d b, bool unoriented = false )
{
    a = a.normalized();
    b = b.normalized();
    const auto perpendicular = cross( a, b );
    // Use the native atan2(cross magnitude, dot) angle formulation, with hypot
    // so squaring a representable tiny cross component does not underflow.
    const auto sine = stableLength( perpendicular );
    const auto cosine = unoriented ? std::abs( dot( a, b ) ) : dot( a, b );
    return std::atan2( sine, cosine ) * cDegrees;
}
Expected<double> spin( Vector3d a, Vector3d b, Vector3d n )
{
    a = project( a, n );
    b = project( b, n );
    if ( !normalize( a ) || !normalize( b ) )
        return unexpected( std::string( _t( "The reference has no usable lever arm about the primary axis." ) ) );
    return dihedralAngle( a, b, n );
}
Expected<void> valid( const Constraints &c )
{
    const auto &s = c.settings;
    if ( !std::isfinite( s.offset ) || !std::isfinite( s.linearTolerance ) || !std::isfinite( s.angularToleranceDeg ) ||
         s.linearTolerance < 0 || s.angularToleranceDeg < 0 || s.angularToleranceDeg >= 90 )
        return unexpected( std::string( _t( "Use finite settings, a nonnegative position tolerance, and an angle tolerance below 90 degrees." ) ) );
    for ( const auto &pair : c.centers )
        if ( !finite( pair.moving ) || !finite( pair.fixed ) )
            return unexpected( std::string( _t( "Feature centers must be finite." ) ) );
    return {};
}
void rigidMetrics( Solution &r )
{
    r.determinant = r.delta.A.det();
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            r.orthogonalityError =
                std::max( r.orthogonalityError,
                          std::abs( dot( r.delta.A.col( i ), r.delta.A.col( j ) ) - ( i == j ? 1. : 0. ) ) );
}
void measureCenterResiduals( const Constraints &c, Solution &r, Vector3d offset = {} )
{
    for ( size_t i = 0; i < c.centers.size(); ++i )
    {
        const auto residual = r.delta( c.centers[i].moving ) - c.centers[i].fixed - offset;
        HoleResidual e;
        e.hasAxis = false;
        e.position = stableLength( residual );
        e.rawCenterDistance = stableLength( r.delta( c.centers[i].moving ) - c.centers[i].fixed );
        r.holes.push_back( e );
        if ( e.position > c.settings.linearTolerance )
            r.conflicts.push_back( { DiagnosticCode::CenterPositionMismatch, int( i ) } );
    }
}

struct Orientation
{
    Matrix3d rotation;
    double yaw = 0;
    int remaining = 3;
    RotationSource source = RotationSource::Unconstrained;
    bool spinLocked = false;
};

Expected<Solution> checkedSolution( Solution result, const Constraints &constraints )
{
    if ( !isRigid( result.delta ) )
        return unexpected( std::string( _t( "Failed to produce or measure a finite rigid transform." ) ) );
    const auto finitePair = [&]( const Vector3d &moving, const Vector3d &fixed )
    {
        const auto transformed = result.delta( moving );
        return finite( transformed ) && finite( transformed - fixed );
    };
    if ( constraints.settings.mode == PrimaryMode::Plane &&
         !finitePair( constraints.movingPlane.center, constraints.fixedPlane.center ) )
        return unexpected( std::string( _t( "Plane placement exceeds the supported numeric range." ) ) );
    if ( constraints.settings.mode == PrimaryMode::Axis &&
         !finitePair( constraints.movingAxis.center, constraints.fixedAxis.center ) )
        return unexpected( std::string( _t( "Axis placement exceeds the supported numeric range." ) ) );
    if ( constraints.settings.mode != PrimaryMode::Plane )
        for ( const auto &center : constraints.centers )
            if ( !finitePair( center.moving, center.fixed ) )
                return unexpected( std::string( _t( "Center placement exceeds the supported numeric range." ) ) );
    if ( constraints.settings.mode == PrimaryMode::Plane )
        for ( const auto &hole : constraints.holes )
            if ( !finitePair( hole.moving.center, hole.fixed.center ) )
                return unexpected( std::string( _t( "Hole placement exceeds the supported numeric range." ) ) );
    for ( const double value :
          { result.primaryPositionError, result.primaryAngleErrorDeg, result.yawDeg, result.signedPlaneGap,
            result.planeOffsetError, result.normalErrorDeg, result.orthogonalityError, result.determinant } )
        if ( !std::isfinite( value ) )
            return unexpected( std::string( _t( "Alignment residuals exceed the supported numeric range." ) ) );
    if ( result.clockingErrorDeg && !std::isfinite( *result.clockingErrorDeg ) )
        return unexpected( std::string( _t( "Clocking residual exceeds the supported numeric range." ) ) );
    for ( const auto &hole : result.holes )
        for ( const double value : { hole.position, hole.inPlane, hole.axial, hole.axisAngleDeg, hole.radiusDifference,
                                     hole.rawCenterDistance } )
            if ( !std::isfinite( value ) )
                return unexpected( std::string( _t( "Locator residuals exceed the supported numeric range." ) ) );
    return result;
}

Expected<Constraints> preparedAxis( Constraints c )
{
    auto good = valid( c );
    if ( !good )
        return unexpected( good.error() );
    if ( !finite( c.movingAxis.center ) || !finite( c.fixedAxis.center ) || !normalize( c.movingAxis.direction ) ||
         !normalize( c.fixedAxis.direction ) )
        return unexpected( std::string( _t( "The primary axes need finite centers and nonzero directions." ) ) );
    if ( c.centers.size() > 2 )
        return unexpected( std::string( _t( "Use at most two additional center locators." ) ) );
    if ( !c.settings.matchAxialCenters && c.centers.empty() && c.settings.offset != 0 )
        return unexpected( std::string( _t( "An axial offset requires matched primary centers or a center locator." ) ) );
    const auto relation = c.settings.axisDirection;
    if ( relation != AxisDirection::Closest && relation != AxisDirection::Same && relation != AxisDirection::Opposing )
        return unexpected( std::string( _t( "Choose closest, same or opposing axis directions." ) ) );
    if ( c.clocking && ( !normalize( c.clocking->moving ) || !normalize( c.clocking->fixed ) ) )
        return unexpected( std::string( _t( "Clocking directions must be nonzero." ) ) );
    return c;
}
Vector3d directedTarget( const Constraints &c )
{
    auto target = c.fixedAxis.direction;
    if ( c.settings.axisDirection == AxisDirection::Opposing ||
         ( c.settings.axisDirection == AxisDirection::Closest && dot( c.movingAxis.direction, target ) < 0 ) )
        target = -target;
    return target;
}
Solution measuredAxis( const Constraints &c, const AffineXf3d &delta )
{
    const auto &s = c.settings;
    const auto n = c.fixedAxis.direction;
    Solution r;
    r.mode = PrimaryMode::Axis;
    r.delta = delta;
    rigidMetrics( r );
    const auto displacement = delta( c.movingAxis.center ) - c.fixedAxis.center;
    r.primaryPositionError = s.matchAxialCenters ? stableLength( displacement - s.offset * n )
                             : s.coaxial         ? stableLength( project( displacement, n ) )
                                                 : 0.;
    r.primaryAngleErrorDeg =
        angleDeg( delta.A * c.movingAxis.direction, directedTarget( c ), s.axisDirection == AxisDirection::Closest );
    if ( r.primaryPositionError > s.linearTolerance )
        r.conflicts.push_back( { s.matchAxialCenters ? DiagnosticCode::PrimaryCenterMismatch
                                                 : DiagnosticCode::PrimaryAxisNotCoaxial } );
    if ( r.primaryAngleErrorDeg > s.angularToleranceDeg )
        r.conflicts.push_back( { DiagnosticCode::PrimaryAxisAngleMismatch } );
    measureCenterResiduals( c, r, s.offset * n );
    if ( c.clocking )
    {
        auto a = project( delta.A * c.clocking->moving, n ), b = project( c.clocking->fixed, n );
        if ( a.length() < cEpsilon || b.length() < cEpsilon )
            r.conflicts.push_back( { DiagnosticCode::ClockingParallelToPrimary } );
        else
        {
            r.clockingErrorDeg = angleDeg( a, b );
            if ( *r.clockingErrorDeg > s.angularToleranceDeg )
                r.conflicts.push_back( { DiagnosticCode::ClockingConflict } );
        }
    }
    return r;
}
Expected<Orientation> orientAxis( const Constraints &c )
{
    const auto &s = c.settings;
    const auto n = c.fixedAxis.direction;
    const auto initial = shortestRotation( c.movingAxis.direction, directedTarget( c ) );
    double yaw = 0;
    bool spinLocked = false;
    RotationSource source = RotationSource::Unconstrained;
    Vector3d sourceOrigin = c.movingAxis.center, targetOrigin = c.fixedAxis.center;
    size_t first = 0;
    if ( !s.coaxial && !s.matchAxialCenters && !c.centers.empty() )
    {
        sourceOrigin = c.centers[0].moving;
        targetOrigin = c.centers[0].fixed;
        first = 1;
    }
    for ( size_t i = first; i < c.centers.size(); ++i )
    {
        const auto a = project( initial * ( c.centers[i].moving - sourceOrigin ), n ),
                   b = project( c.centers[i].fixed - targetOrigin, n );
        if ( !finite( a ) || !finite( b ) || !std::isfinite( a.length() ) || !std::isfinite( b.length() ) )
            return unexpected( std::string( _t( "Axis locator lever arms exceed the supported numeric range." ) ) );
        if ( a.length() > cEpsilon && b.length() > cEpsilon )
        {
            auto angle = spin( a, b, n );
            if ( !angle )
                return unexpected( angle.error() );
            yaw = *angle;
            spinLocked = true;
            source = RotationSource::OffAxisCenterLocator;
            break;
        }
    }
    if ( !spinLocked && c.clocking )
    {
        auto angle = spin( initial * c.clocking->moving, c.clocking->fixed, n );
        if ( !angle )
            return unexpected( angle.error() );
        yaw = *angle;
        spinLocked = true;
        source = RotationSource::DirectedReference;
    }
    const auto rotation = Matrix3d::rotation( n, yaw ) * initial;
    return Orientation{ rotation, yaw, 0, source, spinLocked };
}

void describeAxisMotion( Solution &result, const Constraints &c, const Orientation &orientation )
{
    const auto &s = c.settings;
    const bool positionLocked = s.matchAxialCenters || !c.centers.empty();
    const bool spinLocked = orientation.spinLocked;
    const auto &source = orientation.source;
    result.remainingDof = ( spinLocked ? 0 : 1 ) + ( positionLocked ? 0 : s.coaxial ? 1 : 3 );
    result.rotationSource = source;
    result.freeMotion = positionLocked ? ( spinLocked ? FreeMotion::FullyConstrained : FreeMotion::PrimarySpin )
                        : s.coaxial ? ( spinLocked ? FreeMotion::PrimarySlide : FreeMotion::PrimarySlideAndSpin )
                                    : ( spinLocked ? FreeMotion::Translations : FreeMotion::TranslationsAndSpin );
    if ( result.remainingDof )
        result.warnings.push_back( { DiagnosticCode::Underconstrained } );
    if ( s.matchAxialCenters )
        result.warnings.push_back( { DiagnosticCode::FittedAxialCenters } );
}

Expected<Solution> measureAxis( Constraints c, const AffineXf3d &delta )
{
    if ( !isRigid( delta ) )
        return unexpected( std::string( _t( "The proposed transform is not rigid." ) ) );
    auto value = preparedAxis( std::move( c ) );
    if ( !value )
        return unexpected( value.error() );
    const auto orientation = orientAxis( *value );
    if ( !orientation )
        return unexpected( orientation.error() );
    auto result = measuredAxis( *value, delta );
    describeAxisMotion( result, *value, *orientation );
    return checkedSolution( std::move( result ), *value );
}
Expected<Solution> solveAxis( Constraints c )
{
    auto value = preparedAxis( std::move( c ) );
    if ( !value )
        return unexpected( value.error() );
    c = *value;
    if ( c.settings.axisDirection == AxisDirection::Closest )
    {
        auto same = c, opposite = c;
        same.settings.axisDirection = AxisDirection::Same;
        opposite.settings.axisDirection = AxisDirection::Opposing;
        auto a = solveAxis( std::move( same ) ), b = solveAxis( std::move( opposite ) );
        if ( !a )
            return b;
        if ( !b )
            return a;
        const auto different = []( double x, double y )
        {
            return std::abs( x - y ) > cEpsilon * std::max( { 1., x, y } );
        };
        const auto positionA = a->maximumPositionError(), positionB = b->maximumPositionError();
        const auto angleA = a->maximumAngleError(), angleB = b->maximumAngleError();
        // Compare geometry in separate units; acceptance thresholds never choose the placement.
        bool takeA;
        if ( different( positionA, positionB ) )
            takeA = positionA < positionB;
        else if ( different( angleA, angleB ) )
            takeA = angleA < angleB;
        else
            takeA =
                a->delta.A.x.x + a->delta.A.y.y + a->delta.A.z.z >= b->delta.A.x.x + b->delta.A.y.y + b->delta.A.z.z;
        auto result = takeA ? *a : *b;
        if ( a->acceptable() && b->acceptable() && !c.centers.empty() )
            result.warnings.push_back( { DiagnosticCode::AmbiguousAxisDirection } );
        return result;
    }
    const auto orientation = orientAxis( c );
    if ( !orientation )
        return unexpected( orientation.error() );
    const auto &s = c.settings;
    const auto n = c.fixedAxis.direction;
    const auto &rotation = orientation->rotation;
    Vector3d translation;
    if ( s.coaxial )
    {
        translation = project( c.fixedAxis.center - rotation * c.movingAxis.center, n );
        const auto axial =
            s.matchAxialCenters  ? dot( n, c.fixedAxis.center + s.offset * n - rotation * c.movingAxis.center )
            : !c.centers.empty() ? dot( n, c.centers[0].fixed + s.offset * n - rotation * c.centers[0].moving )
                                 : dot( n, c.movingAxis.center - rotation * c.movingAxis.center );
        translation += axial * n;
    }
    else if ( s.matchAxialCenters )
        translation = c.fixedAxis.center + s.offset * n - rotation * c.movingAxis.center;
    else if ( !c.centers.empty() )
        translation = c.centers[0].fixed + s.offset * n - rotation * c.centers[0].moving;
    else
        translation = c.movingAxis.center - rotation * c.movingAxis.center;
    auto result = measuredAxis( c, { rotation, translation } );
    describeAxisMotion( result, c, *orientation );
    result.yawDeg = orientation->yaw * cDegrees;
    return checkedSolution( std::move( result ), c );
}

Expected<void> validateCenters( const Constraints &c )
{
    auto good = valid( c );
    if ( !good )
        return good;
    if ( c.clocking )
        return unexpected( std::string( _t( "Center-only mode uses additional center pairs. Choose axis/plane mode for a directed reference." ) ) );
    if ( c.centers.empty() || c.centers.size() > 3 )
        return unexpected( std::string( _t( "Choose one primary center pair and at most two additional center pairs." ) ) );
    if ( c.settings.offset != 0 )
        return unexpected( std::string( _t( "Center-only alignment has no offset direction. Use zero offset or choose a plane/axis mode." ) ) );
    if ( c.centers.size() > 1 )
    {
        auto moving = c.centers[1].moving - c.centers[0].moving;
        auto fixed = c.centers[1].fixed - c.centers[0].fixed;
        if ( !normalize( moving ) || !normalize( fixed ) )
            return unexpected( std::string( _t( "The second center needs a distinct, numerically representable baseline on both parts." ) ) );
    }
    return {};
}
Solution measuredCenters( const Constraints &c, const AffineXf3d &delta )
{
    Solution result;
    result.mode = PrimaryMode::Centers;
    result.delta = delta;
    rigidMetrics( result );
    measureCenterResiduals( c, result );
    result.primaryPositionError = result.holes.front().position;
    return result;
}
Expected<Orientation> orientCenters( const Constraints &c )
{
    Matrix3d rotation;
    int remaining = 3;
    RotationSource source = RotationSource::Unconstrained;
    if ( c.centers.size() > 1 )
    {
        const auto moving = ( c.centers[1].moving - c.centers[0].moving ).normalized(),
                   fixed = ( c.centers[1].fixed - c.centers[0].fixed ).normalized();
        rotation = shortestRotation( moving, fixed );
        remaining = 1;
        source = RotationSource::SecondCenter;
        if ( c.centers.size() > 2 )
        {
            const auto a = project( rotation * ( c.centers[2].moving - c.centers[0].moving ), fixed ),
                       b = project( c.centers[2].fixed - c.centers[0].fixed, fixed );
            if ( !finite( a ) || !finite( b ) || !std::isfinite( a.length() ) || !std::isfinite( b.length() ) )
                return unexpected( std::string( _t( "Center locator lever arms exceed the supported numeric range." ) ) );
            if ( a.length() > cEpsilon && b.length() > cEpsilon )
            {
                auto yaw = spin( a, b, fixed );
                if ( !yaw )
                    return unexpected( yaw.error() );
                rotation = Matrix3d::rotation( fixed, *yaw ) * rotation;
                remaining = 0;
                source = RotationSource::ThirdCenter;
            }
        }
    }
    return Orientation{ rotation, 0, remaining, source };
}

void describeCenterMotion( Solution &result, const Constraints &c, const Orientation &orientation )
{
    const int remaining = orientation.remaining;
    const auto &source = orientation.source;
    result.remainingDof = remaining;
    result.rotationSource = source;
    result.freeMotion = remaining == 3 ? FreeMotion::Rotations
                        : remaining == 1 ? FreeMotion::BaselineSpin
                                         : FreeMotion::FullyConstrained;
    if ( remaining )
        result.warnings.push_back( { DiagnosticCode::Underconstrained } );
    if ( c.centers.size() == 3 && remaining )
        result.warnings.push_back( { DiagnosticCode::CollinearCenters } );
}

Expected<Solution> measureCenters( Constraints c, const AffineXf3d &delta )
{
    auto good = validateCenters( c );
    if ( !good )
        return unexpected( good.error() );
    if ( !isRigid( delta ) )
        return unexpected( std::string( _t( "The proposed transform is not rigid." ) ) );
    const auto orientation = orientCenters( c );
    if ( !orientation )
        return unexpected( orientation.error() );
    auto result = measuredCenters( c, delta );
    describeCenterMotion( result, c, *orientation );
    return checkedSolution( std::move( result ), c );
}
Expected<Solution> solveCenters( Constraints c )
{
    auto good = validateCenters( c );
    if ( !good )
        return unexpected( good.error() );
    const auto orientation = orientCenters( c );
    if ( !orientation )
        return unexpected( orientation.error() );
    const auto &rotation = orientation->rotation;
    auto result = measuredCenters( c, { rotation, c.centers[0].fixed - rotation * c.centers[0].moving } );
    describeCenterMotion( result, c, *orientation );
    return checkedSolution( std::move( result ), c );
}

struct PreparedPlane
{
    Constraints c;
    Vector3d seat;
    std::vector<std::pair<Vector3d, Vector3d>> anchors;
};
Expected<PreparedPlane> preparePlane( Constraints c )
{
    auto &s = c.settings;
    if ( s.normals != NormalRelation::Opposing && s.normals != NormalRelation::Same )
        return unexpected( std::string( _t( "Choose same or opposing plane normals." ) ) );
    if ( s.holeMode != HoleMode::Centers && s.holeMode != HoleMode::Axes )
        return unexpected( std::string( _t( "Choose hole centers or axes." ) ) );
    const auto good = valid( c );
    if ( !good )
        return unexpected( good.error() );
    if ( !finite( c.movingPlane.center ) || !finite( c.fixedPlane.center ) || !normalize( c.movingPlane.normal ) ||
         !normalize( c.fixedPlane.normal ) )
        return unexpected( std::string( _t( "The fitted planes need finite centers and nonzero normals." ) ) );
    if ( c.holes.size() > 2 )
        return unexpected( std::string( _t( "Use one primary hole and at most one second hole." ) ) );
    for ( auto &pair : c.holes )
        for ( auto *h : { &pair.moving, &pair.fixed } )
        {
            if ( !finite( h->center ) || !normalize( h->axis ) || !std::isfinite( h->radius ) || h->radius <= cEpsilon )
                return unexpected( std::string( _t( "A hole has an invalid center, axis or radius." ) ) );
            if ( h->kind != HoleKind::Circle && h->kind != HoleKind::Cylinder )
                return unexpected( std::string( _t( "Holes must be fitted circles or cylinders." ) ) );
        }
    if ( c.clocking )
    {
        if ( !normalize( c.clocking->moving ) || !normalize( c.clocking->fixed ) ||
             project( c.clocking->moving, c.movingPlane.normal ).length() < cEpsilon ||
             project( c.clocking->fixed, c.fixedPlane.normal ).length() < cEpsilon )
            return unexpected(
                std::string( _t( "The clocking reference must have a nonzero direction within the mating plane." ) ) );
    }
    PreparedPlane p;
    p.c = std::move( c );
    p.seat = p.c.fixedPlane.center + p.c.settings.offset * p.c.fixedPlane.normal;
    const auto intersectHole = []( const Hole &h, const Vector3d &center, const Vector3d &normal ) -> Expected<Vector3d>
    {
        const auto point = intersection( Plane3d::fromDirAndPt( normal, center ), Line3d{ h.center, h.axis }, 1e-6 );
        if ( !point )
            return unexpected(
                std::string( _t( "A hole axis is nearly parallel to the mating plane; its intersection is unstable." ) ) );
        if ( !finite( *point ) )
            return unexpected( std::string( _t( "Hole-plane intersection exceeds the supported numeric range." ) ) );
        return *point;
    };
    for ( const auto &pair : p.c.holes )
    {
        if ( p.c.settings.holeMode == HoleMode::Centers )
            p.anchors.emplace_back( pair.moving.center,
                                    pair.fixed.center + p.c.settings.offset * p.c.fixedPlane.normal );
        else
        {
            auto a = intersectHole( pair.moving, p.c.movingPlane.center, p.c.movingPlane.normal );
            auto b = intersectHole( pair.fixed, p.seat, p.c.fixedPlane.normal );
            if ( !a )
                return unexpected( a.error() );
            if ( !b )
                return unexpected( b.error() );
            p.anchors.emplace_back( *a, *b );
        }
    }
    return p;
}
Solution measuredPlane( const PreparedPlane &p, const AffineXf3d &delta )
{
    const auto &c = p.c;
    const auto &n = c.fixedPlane.normal;
    const auto &s = c.settings;
    Solution r;
    r.delta = delta;
    rigidMetrics( r );
    r.mode = PrimaryMode::Plane;
    r.signedPlaneGap = dot( n, delta( c.movingPlane.center ) - c.fixedPlane.center );
    r.planeOffsetError = r.signedPlaneGap - s.offset;
    r.normalErrorDeg = angleDeg( delta.A * c.movingPlane.normal, double( int( s.normals ) ) * n );
    r.primaryPositionError = std::abs( r.planeOffsetError );
    r.primaryAngleErrorDeg = r.normalErrorDeg;
    if ( std::abs( r.planeOffsetError ) > s.linearTolerance || r.normalErrorDeg > s.angularToleranceDeg )
        r.conflicts.push_back( { DiagnosticCode::PlaneSeatingMismatch } );
    for ( size_t i = 0; i < c.holes.size(); ++i )
    {
        const auto &pair = c.holes[i];
        const auto difference = delta( p.anchors[i].first ) - p.anchors[i].second;
        HoleResidual e;
        e.inPlane = stableLength( project( difference, n ) );
        // Plane seating owns depth. Hole locators constrain only motion in that plane.
        e.position = e.inPlane;
        e.axial = dot( difference, n );
        e.axisAngleDeg = angleDeg( delta.A * pair.moving.axis, pair.fixed.axis, true );
        e.angleConstrained = s.holeMode == HoleMode::Axes;
        e.radiusDifference = pair.moving.radius - pair.fixed.radius;
        e.rawCenterDistance = stableLength( delta( pair.moving.center ) - pair.fixed.center );
        r.holes.push_back( e );
        if ( e.position > s.linearTolerance )
            r.conflicts.push_back( { DiagnosticCode::HolePositionMismatch, int( i ) } );
        if ( s.holeMode == HoleMode::Axes && e.axisAngleDeg > s.angularToleranceDeg )
            r.conflicts.push_back( { DiagnosticCode::HoleAxisAngleMismatch, int( i ) } );
        if ( std::abs( e.radiusDifference ) > s.linearTolerance )
            r.warnings.push_back( { DiagnosticCode::HoleRadiusMismatch, int( i ) } );
    }
    if ( c.clocking )
    {
        const auto a = project( delta.A * c.clocking->moving, n );
        const auto b = project( c.clocking->fixed, n );
        if ( a.length() < cEpsilon || b.length() < cEpsilon )
            r.conflicts.push_back( { DiagnosticCode::ClockingParallelToPrimary } );
        else
        {
            r.clockingErrorDeg = angleDeg( a, b );
            if ( *r.clockingErrorDeg > s.angularToleranceDeg )
                r.conflicts.push_back( { DiagnosticCode::ClockingConflict } );
        }
    }
    return r;
}
Expected<Orientation> orientPlane( const PreparedPlane &p )
{
    const auto &n = p.c.fixedPlane.normal;
    const auto r0 = shortestRotation( p.c.movingPlane.normal, double( int( p.c.settings.normals ) ) * n );
    double yaw = 0;
    int remaining = 3;
    RotationSource source = RotationSource::Unconstrained;
    if ( p.anchors.size() == 2 )
    {
        auto angle =
            spin( r0 * ( p.anchors[1].first - p.anchors[0].first ), p.anchors[1].second - p.anchors[0].second, n );
        if ( !angle )
            return unexpected( angle.error() );
        yaw = *angle;
        remaining = 0;
        source = RotationSource::SecondHole;
    }
    else if ( p.c.clocking )
    {
        auto angle = spin( r0 * p.c.clocking->moving, p.c.clocking->fixed, n );
        if ( !angle )
            return unexpected( angle.error() );
        yaw = *angle;
        remaining = p.anchors.empty() ? 2 : 0;
        source = RotationSource::DirectedReference;
    }
    else if ( !p.c.holes.empty() && p.c.settings.holeMode == HoleMode::Axes )
    {
        const auto a = r0 * p.c.holes[0].moving.axis;
        auto b = p.c.holes[0].fixed.axis;
        if ( dot( a, n ) * dot( b, n ) < 0 )
            b = -b;
        // Even a small oblique axis determines spin when its projection is numerically usable.
        // Use the same geometric precision floor as spin(), independent of residual tolerance.
        if ( project( a, n ).length() > cEpsilon && project( b, n ).length() > cEpsilon )
        {
            auto angle = spin( a, b, n );
            if ( !angle )
                return unexpected( angle.error() );
            yaw = *angle;
            remaining = 0;
            source = RotationSource::ObliquePrimaryAxis;
        }
        else
            remaining = 1;
    }
    else if ( !p.anchors.empty() )
        remaining = 1;
    const auto rotation = Matrix3d::rotation( n, yaw ) * r0;
    return Orientation{ rotation, yaw, remaining, source };
}

void describePlaneMotion( Solution &r, const PreparedPlane &p, const Orientation &orientation )
{
    const int remaining = orientation.remaining;
    const auto &source = orientation.source;
    const auto &n = p.c.fixedPlane.normal;
    r.remainingDof = remaining;
    r.rotationSource = source;
    r.freeMotion = remaining == 3 ? FreeMotion::PlaneTranslationsAndSpin
                   : remaining == 2 ? FreeMotion::PlaneTranslations
                   : remaining == 1 ? FreeMotion::PrimarySpin
                                    : FreeMotion::FullyConstrained;
    if ( !p.anchors.empty() && p.c.settings.holeMode == HoleMode::Centers )
        r.warnings.push_back( { DiagnosticCode::PlaneCentersProjected } );
    if ( p.c.settings.balanceHoles && p.anchors.size() == 2 )
        r.warnings.push_back( { DiagnosticCode::BalancedHoleFit } );
    if ( source == RotationSource::ObliquePrimaryAxis )
        r.warnings.push_back( { DiagnosticCode::ObliqueHoleAxis } );
    if ( remaining )
        r.warnings.push_back( { DiagnosticCode::Underconstrained } );
    if ( dot( p.c.movingPlane.normal, double( int( p.c.settings.normals ) ) * n ) < -1 + 1e-9 && remaining )
        r.warnings.push_back( { DiagnosticCode::OppositePlaneNormalsConvention } );
}

Expected<Solution> measurePlane( Constraints c, const AffineXf3d &delta )
{
    if ( !isRigid( delta ) )
        return unexpected( std::string( _t( "The proposed transform contains scale, shear or reflection." ) ) );
    auto p = preparePlane( std::move( c ) );
    if ( !p )
        return unexpected( p.error() );
    const auto orientation = orientPlane( *p );
    if ( !orientation )
        return unexpected( orientation.error() );
    auto result = measuredPlane( *p, delta );
    describePlaneMotion( result, *p, *orientation );
    return checkedSolution( std::move( result ), p->c );
}
Expected<Solution> solvePlane( Constraints c )
{
    auto prepared = preparePlane( std::move( c ) );
    if ( !prepared )
        return unexpected( prepared.error() );
    const auto &p = *prepared;
    const auto orientation = orientPlane( p );
    if ( !orientation )
        return unexpected( orientation.error() );
    const auto &n = p.c.fixedPlane.normal;
    const auto &rotation = orientation->rotation;
    auto translation = n * dot( n, p.seat - rotation * p.c.movingPlane.center );
    if ( p.anchors.empty() )
        translation += project( p.c.movingPlane.center - rotation * p.c.movingPlane.center, n );
    else
    {
        auto tangential = project( p.anchors[0].second - rotation * p.anchors[0].first, n );
        if ( p.c.settings.balanceHoles && p.anchors.size() == 2 )
        {
            PointToPointAligningTransform aligning;
            for ( const auto &anchor : p.anchors )
                aligning.add( project( rotation * anchor.first, n ), project( anchor.second, n ) );
            tangential = project( aligning.findBestTranslation(), n );
        }
        translation += tangential;
    }
    Solution r = measuredPlane( p, { rotation, translation } );
    describePlaneMotion( r, p, *orientation );
    r.yawDeg = orientation->yaw * cDegrees;
    return checkedSolution( std::move( r ), p.c );
}

using FeaturePtr = std::shared_ptr<const FeatureObject>;

Expected<AffineXf3d> parentWorld( const FeaturePtr &feature )
{
    // The feature's own scale controls its displayed dimensions, so check only its ancestors.
    // Its getters already include the feature's own local transform.
    auto featureViews = feature->xfsForAllViewports();
    if ( featureViews.reset() )
        return unexpected( std::string( _t( "Per-viewport feature transforms are unsupported." ) ) );
    for ( const auto *current = feature->parent(); current; current = current->parent() )
    {
        if ( !isRigid( AffineXf3d( current->xf() ) ) )
            return unexpected( std::string( _t( "Feature ancestors must have rigid transforms; scale, shear and mirror are unsupported." ) ) );
        auto ancestorViews = current->xfsForAllViewports();
        if ( ancestorViews.reset() )
            return unexpected( std::string( _t( "Per-viewport ancestor transforms are unsupported." ) ) );
    }
    return feature->parent() ? AffineXf3d( feature->parent()->worldXf() ) : AffineXf3d{};
}

Expected<Plane> readPlane( const FeaturePtr &feature )
{
    const auto *plane = dynamic_cast<const PlaneObject *>( feature.get() );
    if ( !plane )
        return unexpected( std::string( _t( "Choose an existing fitted plane on each side." ) ) );
    if ( !std::isfinite( plane->getSizeX() ) || !std::isfinite( plane->getSizeY() ) || plane->getSizeX() <= cEpsilon ||
         plane->getSizeY() <= cEpsilon )
        return unexpected( std::string( _t( "A fitted plane has invalid display extent." ) ) );
    const auto parent = parentWorld( feature );
    if ( !parent )
        return unexpected( parent.error() );
    Plane result{ ( *parent )( Vector3d( plane->getCenter() ) ), parent->A * Vector3d( plane->getNormal() ) };
    if ( !finite( result.center ) || !normalize( result.normal ) )
        return unexpected( std::string( _t( "A fitted plane has an invalid center or normal." ) ) );
    return result;
}

Expected<Hole> readHole( const FeaturePtr &feature )
{
    const auto parent = parentWorld( feature );
    if ( !parent )
        return unexpected( parent.error() );
    Hole result;
    if ( const auto *circle = dynamic_cast<const CircleObject *>( feature.get() ) )
    {
        result.center = ( *parent )( Vector3d( circle->getCenter() ) );
        result.axis = parent->A * Vector3d( circle->getNormal() );
        result.radius = circle->getRadius();
        result.kind = HoleKind::Circle;
    }
    else if ( const auto *cylinder = dynamic_cast<const CylinderObject *>( feature.get() ) )
    {
        if ( !std::isfinite( cylinder->getLength() ) || cylinder->getLength() <= cEpsilon )
            return unexpected( std::string( _t( "A fitted cylinder has invalid length." ) ) );
        result.center = ( *parent )( Vector3d( cylinder->getCenter() ) );
        result.axis = parent->A * Vector3d( cylinder->getDirection() );
        result.radius = cylinder->getRadius();
        result.kind = HoleKind::Cylinder;
    }
    else
        return unexpected( std::string( _t( "Choose an existing fitted circle or cylinder." ) ) );
    if ( !finite( result.center ) || !normalize( result.axis ) || !std::isfinite( result.radius ) ||
         result.radius <= cEpsilon )
        return unexpected( std::string( _t( "A fitted hole has an invalid center, axis or radius." ) ) );
    return result;
}

Expected<Vector3d> readDirection( const FeaturePtr &feature )
{
    if ( dynamic_cast<const PlaneObject *>( feature.get() ) )
    {
        const auto plane = readPlane( feature );
        if ( !plane )
            return unexpected( plane.error() );
        return plane->normal;
    }
    if ( const auto *line = dynamic_cast<const LineObject *>( feature.get() ) )
    {
        const auto parent = parentWorld( feature );
        if ( !parent )
            return unexpected( parent.error() );
        auto direction = parent->A * Vector3d( line->getDirection() );
        if ( !std::isfinite( line->getLength() ) || line->getLength() <= cEpsilon || !normalize( direction ) )
            return unexpected( std::string( _t( "A fitted line has invalid length or direction." ) ) );
        return direction;
    }
    const auto hole = readHole( feature );
    if ( !hole )
        return unexpected( hole.error() );
    return hole->axis;
}

Expected<Vector3d> readCenter( const FeaturePtr &feature )
{
    const auto parent = parentWorld( feature );
    if ( !parent )
        return unexpected( parent.error() );
    Vector3d result;
    if ( const auto *point = dynamic_cast<const PointObject *>( feature.get() ) )
        result = ( *parent )( Vector3d( point->getPoint() ) );
    else if ( const auto *sphere = dynamic_cast<const SphereObject *>( feature.get() ) )
    {
        if ( !std::isfinite( sphere->getRadius() ) || sphere->getRadius() <= cEpsilon )
            return unexpected( std::string( _t( "A fitted sphere has an invalid radius." ) ) );
        result = ( *parent )( Vector3d( sphere->getCenter() ) );
    }
    else
    {
        const auto hole = readHole( feature );
        if ( !hole )
            return unexpected( hole.error() );
        result = hole->center;
    }
    if ( !finite( result ) )
        return unexpected( std::string( _t( "A fitted feature has an invalid center." ) ) );
    return result;
}

Expected<Axis> readAxis( const FeaturePtr &feature )
{
    const auto direction = readDirection( feature );
    if ( !direction )
        return unexpected( direction.error() );
    if ( const auto *line = dynamic_cast<const LineObject *>( feature.get() ) )
    {
        const auto parent = parentWorld( feature );
        if ( !parent )
            return unexpected( parent.error() );
        const auto center = ( *parent )( Vector3d( line->getCenter() ) );
        if ( !finite( center ) )
            return unexpected( std::string( _t( "A fitted line has an invalid center." ) ) );
        return Axis{ center, *direction };
    }
    const auto center = readCenter( feature );
    if ( !center )
        return unexpected( center.error() );
    return Axis{ *center, *direction };
}

Expected<void> validatePair( const FeaturePair &pair, bool required = false )
{
    if ( bool( pair.moving ) != bool( pair.fixed ) || ( required && !pair.moving ) )
        return unexpected( std::string( _t( "Choose both sides of each feature pair, or leave an optional pair empty." ) ) );
    if ( pair.moving && pair.moving == pair.fixed )
        return unexpected( std::string( _t( "Moving and fixed references must be distinct features." ) ) );
    return {};
}
} // namespace

bool isRigid( const AffineXf3d &xf, double tolerance )
{
    if ( !std::isfinite( tolerance ) || tolerance < 0 || !finite( xf.b ) || !finite( xf.A.x ) ||
         !finite( xf.A.y ) || !finite( xf.A.z ) )
        return false;
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            if ( !( std::abs( dot( xf.A.col( i ), xf.A.col( j ) ) - ( i == j ? 1. : 0. ) ) <= tolerance ) )
                return false;
    return std::abs( xf.A.det() - 1 ) <= tolerance;
}
double Solution::maximumPositionError() const
{
    double result = std::max( std::abs( planeOffsetError ), primaryPositionError );
    for ( const auto &hole : holes )
        result = std::max( result, hole.position );
    return result;
}
double Solution::maximumAngleError() const
{
    double result = primaryAngleErrorDeg;
    if ( clockingErrorDeg )
        result = std::max( result, *clockingErrorDeg );
    for ( const auto &hole : holes )
        if ( hole.angleConstrained )
            result = std::max( result, hole.axisAngleDeg );
    return result;
}
Expected<Solution> solve( Constraints c )
{
    switch ( c.settings.mode )
    {
    case PrimaryMode::Plane:
        return solvePlane( std::move( c ) );
    case PrimaryMode::Axis:
        return solveAxis( std::move( c ) );
    case PrimaryMode::Centers:
        return solveCenters( std::move( c ) );
    }
    return unexpected( std::string( _t( "Choose plane, axis or center alignment." ) ) );
}
Expected<Solution> measure( Constraints c, const AffineXf3d &delta )
{
    switch ( c.settings.mode )
    {
    case PrimaryMode::Plane:
        return measurePlane( std::move( c ), delta );
    case PrimaryMode::Axis:
        return measureAxis( std::move( c ), delta );
    case PrimaryMode::Centers:
        return measureCenters( std::move( c ), delta );
    }
    return unexpected( std::string( _t( "Choose plane, axis or center alignment." ) ) );
}

Expected<Constraints> readConstraints( const FeatureReferences &references, const Settings &settings )
{
    for ( const auto *pair :
          { &references.primary, &references.locators[0], &references.locators[1], &references.clocking } )
    {
        const auto valid = validatePair( *pair, pair == &references.primary );
        if ( !valid )
            return unexpected( valid.error() );
    }
    if ( references.locators[1].moving &&
         ( !references.locators[0].moving || references.locators[0].moving == references.locators[1].moving ||
           references.locators[0].fixed == references.locators[1].fixed ) )
        return unexpected( std::string( _t( "The second locator must be a distinct pair after the first locator." ) ) );
    Constraints result;
    result.settings = settings;
    const auto good = valid( result );
    if ( !good )
        return unexpected( good.error() );
    if ( settings.mode == PrimaryMode::Plane )
    {
        const auto moving = readPlane( references.primary.moving );
        const auto fixed = readPlane( references.primary.fixed );
        if ( !moving )
            return unexpected( moving.error() );
        if ( !fixed )
            return unexpected( fixed.error() );
        result.movingPlane = *moving;
        result.fixedPlane = *fixed;
    }
    else if ( settings.mode == PrimaryMode::Axis )
    {
        const auto moving = readAxis( references.primary.moving );
        const auto fixed = readAxis( references.primary.fixed );
        if ( !moving )
            return unexpected( moving.error() );
        if ( !fixed )
            return unexpected( fixed.error() );
        result.movingAxis = *moving;
        result.fixedAxis = *fixed;
    }
    else if ( settings.mode == PrimaryMode::Centers )
    {
        const auto moving = readCenter( references.primary.moving );
        const auto fixed = readCenter( references.primary.fixed );
        if ( !moving )
            return unexpected( moving.error() );
        if ( !fixed )
            return unexpected( fixed.error() );
        result.centers.push_back( { *moving, *fixed } );
    }
    else
        return unexpected( std::string( _t( "Choose plane, axis or center alignment." ) ) );
    for ( const auto &pair : references.locators )
    {
        if ( !pair.moving )
            continue;
        if ( pair.moving == references.primary.moving || pair.fixed == references.primary.fixed )
            return unexpected( std::string( _t( "A locator must be distinct from the primary feature." ) ) );
        if ( settings.mode == PrimaryMode::Plane )
        {
            const auto moving = readHole( pair.moving );
            const auto fixed = readHole( pair.fixed );
            if ( !moving )
                return unexpected( moving.error() );
            if ( !fixed )
                return unexpected( fixed.error() );
            result.holes.push_back( { *moving, *fixed } );
        }
        else
        {
            const auto moving = readCenter( pair.moving );
            const auto fixed = readCenter( pair.fixed );
            if ( !moving )
                return unexpected( moving.error() );
            if ( !fixed )
                return unexpected( fixed.error() );
            result.centers.push_back( { *moving, *fixed } );
        }
    }
    if ( references.clocking.moving )
    {
        if ( settings.mode == PrimaryMode::Centers )
            return unexpected(
                std::string( _t( "Center alignment uses additional center pairs instead of directed references." ) ) );
        const auto moving = readDirection( references.clocking.moving );
        const auto fixed = readDirection( references.clocking.fixed );
        if ( !moving )
            return unexpected( moving.error() );
        if ( !fixed )
            return unexpected( fixed.error() );
        result.clocking = { { *moving, *fixed } };
    }
    return result;
}

} // namespace MR::FeatureAlignment
