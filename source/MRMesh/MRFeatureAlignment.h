#pragma once

#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRFeatureObject.h>
#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <string>
#include <vector>

namespace MR::FeatureAlignment
{

/// \defgroup FeatureAlignmentGroup Feature Alignment
/// \ingroup FeaturesGroup
/// Rigid placement from existing fitted features, with primary constraints held exactly.
/// \{

enum class MRMESH_CLASS NormalRelation
{
    Opposing = -1,
    Same = 1
};
enum class MRMESH_CLASS HoleMode
{
    Centers,
    Axes
};
enum class MRMESH_CLASS HoleKind
{
    Circle,
    Cylinder
};
enum class MRMESH_CLASS PrimaryMode
{
    Plane,
    Axis,
    Centers
};
enum class MRMESH_CLASS AxisDirection
{
    Closest = 0,
    Same = 1,
    Opposing = -1
};
/// Plane datum including its reference center, used to preserve free tangential placement.
struct MRMESH_CLASS Plane
{
    Vector3d center;            ///< Reference point on the plane.
    Vector3d normal{ 0, 0, 1 }; ///< Nonzero normal; normalized by solve/measure.
};
struct MRMESH_CLASS Axis
{
    Vector3d center;               ///< Reference center on the axis, not necessarily a physical rim.
    Vector3d direction{ 0, 0, 1 }; ///< Nonzero axis direction; normalized by solve/measure.
};
struct MRMESH_CLASS CenterPair
{
    Vector3d moving, fixed; ///< Corresponding centers in the common input frame.
};
struct MRMESH_CLASS Hole
{
    Vector3d center;                   ///< Fitted feature center; depth is diagnostic in plane center mode.
    Vector3d axis{ 0, 0, 1 };          ///< Nonzero circle normal or cylinder axis.
    double radius{ 1 };                ///< Positive radius; mismatch is a warning, never a scale request.
    HoleKind kind{ HoleKind::Circle }; ///< Feature identity for validation/reporting.
};
struct MRMESH_CLASS HolePair
{
    Hole moving, fixed; ///< Corresponding circle/cylinder datums.
};
/// Units match the input geometry; offsets are signed, angular tolerances are in degrees.
struct MRMESH_CLASS Settings
{
    PrimaryMode mode{ PrimaryMode::Plane };                ///< Selects the primary hard constraint.
    NormalRelation normals{ NormalRelation::Opposing };    ///< Plane mode: directed normal relationship.
    AxisDirection axisDirection{ AxisDirection::Closest }; ///< Axis mode: sign choice; Closest evaluates both.
    bool coaxial{ true };                                  ///< Axis mode: lock radial displacement to the primary axis.
    bool matchAxialCenters{ true }; ///< Axis mode only: lock fitted primary centers at the axial offset.
    bool balanceHoles{ true };      ///< Plane mode, two holes: share tangent mismatch equally; keep tilt/gap exact.
    /// Plane mode: Centers projects fitted centers tangentially, ignoring their depth;
    /// Axes intersects each hole axis with its mating plane and also measures axis-angle conflicts.
    HoleMode holeMode{ HoleMode::Centers };
    /// Plane gap along the fixed normal, or axis offset along the fixed direction; center mode requires zero.
    double offset{ 0 };
    double linearTolerance{ 1.0 };     ///< Nonnegative residual tolerance in input scene length units.
    double angularToleranceDeg{ 5.0 }; ///< Nonnegative angular tolerance in degrees, strictly below 90.
    bool operator==( const Settings & ) const = default;
};
/// All values must be expressed in the same coordinate frame.
/// Plane mode owns tilt and gap; locators control tangent placement and residual spin only.
/// Axis and center modes preserve primary references before considering later locators.
struct MRMESH_CLASS Constraints
{
    Plane movingPlane, fixedPlane; ///< Plane mode: owns tilt and signed gap/depth.
    Axis movingAxis, fixedAxis;    ///< Axis mode: owns direction and enabled primary position locks.
    /// Axis mode: up to two locators. Center mode: primary followed by up to two locators.
    std::vector<CenterPair> centers;
    std::vector<HolePair> holes; ///< Plane mode: primary hole followed by at most one second hole.
    /// Moving/fixed directed references; projected about the primary normal/axis. Unsupported in center mode.
    std::optional<std::pair<Vector3d, Vector3d>> clocking;
    Settings settings; ///< Constraint policy and residual tolerances.
};
struct MRMESH_CLASS HoleResidual
{
    bool hasAxis{ true };           ///< False for a center-only locator.
    bool angleConstrained{ false }; ///< Whether axisAngleDeg participates in conflicts/maximumAngleError.
    double position{};     ///< Controlled position error: tangent distance for plane holes, 3D distance for centers.
    double inPlane{};      ///< Plane-hole tangential distance; zero for center-only locators.
    double axial{};        ///< Plane-hole signed depth mismatch, diagnostic only.
    double axisAngleDeg{}; ///< Unoriented hole-axis angle in degrees.
    double radiusDifference{};  ///< Moving minus fixed radius in input length units; diagnostic only.
    double rawCenterDistance{}; ///< Full fitted-center distance, independent of projected/intersected locators.
};
/// Placement and measured residuals. An incompatible measurement produces a valid solution
/// with conflicts, while invalid input produces an unexpected error instead.
struct MRMESH_CLASS Solution
{
    AffineXf3d delta;      ///< Maps moving input coordinates into the fixed input frame.
    int remainingDof{ 3 }; ///< Unconstrained rigid degrees of freedom, independent of residual conflicts.
    std::string rotationSource{ "unconstrained" }; ///< Reference determining residual rotation, if any.
    std::string freeMotion;                 ///< Constraint-rank description; populated by both solve and measure.
    PrimaryMode mode{ PrimaryMode::Plane }; ///< Active primary mode.
    double primaryPositionError{};          ///< Controlled primary position residual in input length units.
    double primaryAngleErrorDeg{};          ///< Controlled primary direction residual in degrees.
    double yawDeg{};                        ///< Residual spin selected by solve in degrees; zero for measure.
    double signedPlaneGap{};                ///< Plane mode: measured gap along the fixed normal.
    double planeOffsetError{};              ///< Plane mode: signedPlaneGap minus requested offset.
    double normalErrorDeg{};                ///< Plane mode: directed normal relationship error in degrees.
    double orthogonalityError{};            ///< Maximum absolute error in R-transpose times R versus identity.
    double determinant{ 1 };                ///< Rotation determinant; a successful API result is finite and rigid.
    std::optional<double> clockingErrorDeg; ///< Directed projected-reference residual in degrees.
    std::vector<HoleResidual> holes;        ///< One entry per active hole or center pair, in input order.
    std::vector<std::string> conflicts, warnings; ///< Tolerance failures versus informational diagnostics.
    /// True when all controlled residuals meet tolerance. Invalid numeric results are returned as errors.
    [[nodiscard]] bool acceptable() const
    {
        return conflicts.empty();
    }
    /// Largest controlled position residual; excludes diagnostic-only depth/radius differences.
    [[nodiscard]] MRMESH_API double maximumPositionError() const;
    /// Largest constrained primary/locator/clocking angular residual in degrees.
    [[nodiscard]] MRMESH_API double maximumAngleError() const;
};

/// Calculates a proper rigid transform; never scales, mirrors, fits or edits input geometry.
/// Underconstrained solutions follow deterministic shortest-rotation/free-placement conventions.
[[nodiscard]] MRMESH_API Expected<Solution> solve( Constraints constraints );
/// Retains the supplied rigid placement and reports its residuals without reducing them.
/// DOF/freeMotion describe the input constraint rank, as in solve, independently of placement errors.
/// Inputs whose intermediate arithmetic or reported results exceed the finite numeric range are rejected.
[[nodiscard]] MRMESH_API Expected<Solution> measure( Constraints constraints, const AffineXf3d &delta );
/// Tests finite values, proper rotation determinant and orthogonality within the nonnegative tolerance.
[[nodiscard]] MRMESH_API bool isRigid( const AffineXf3d &xf, double tolerance = 1e-5 );

/// Corresponding fitted features. Both null pointers denote an unused optional pair.
struct MRMESH_CLASS FeaturePair
{
    std::shared_ptr<const FeatureObject> moving, fixed;
};

/// Read-only feature references. No hierarchy ownership or scene mutation is performed.
/// Plane primary: planes; locators: circles/cylinders. Axis primary: lines/circles/cylinders.
/// Center primary/locators: points/spheres/circles/cylinders. Clocking: planes/lines/circles/cylinders.
/// Optional locator pairs are ordered, distinct, and complete; clocking may be empty.
struct MRMESH_CLASS FeatureReferences
{
    FeaturePair primary;
    std::array<FeaturePair, 2> locators;
    FeaturePair clocking;
};

/// Reads fitted feature geometry into a common world frame; rejects nonrigid ancestry.
/// Feature display sizes are not physical scales. Per-viewport transforms are unsupported.
/// References retain their objects, but the result contains copied geometry only.
[[nodiscard]] MRMESH_API Expected<Constraints> readConstraints( const FeatureReferences &references,
                                                                const Settings &settings = {} );
/// Reads and solves without modifying any feature, parent, or scene object.
[[nodiscard]] MRMESH_API Expected<Solution> solve( const FeatureReferences &references, const Settings &settings = {} );
/// \}

} // namespace MR::FeatureAlignment
