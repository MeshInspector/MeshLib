#pragma once
#include "MRVoxelsFwd.h"

#include "MRMesh/MRAxis.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRExpected.h"

namespace MR
{

/// Direction in which the tool traverses each contour
enum class BypassDirection
{
    Clockwise,       ///< Tool moves clockwise around the section
    CounterClockwise ///< Tool moves counter-clockwise around the section
};

/// Parameters shared by all tool path generation functions
struct ToolPathParams
{
    /// radius of the milling tool
    float millRadius = {};
    /// size of voxel needed to offset mesh
    float voxelSize = {};
    /// distance between sections built along Z axis;
    /// in Constant Cusp mode sectionStep should be bigger than voxelSize (x1.2 or more is recommended)
    float sectionStep = {};
    /// if distance to the next section is smaller than it, transition will be performed along the surface;
    /// otherwise transition will be through the safe plane
    float critTransitionLength = {};
    /// when the mill is moving down, it will be slowed down in this distance from mesh
    float plungeLength = {};
    /// when the mill is moving up, it will be slowed down in this distance from mesh
    float retractLength = {};
    /// speed of slow movement down
    float plungeFeed = {};
    /// speed of slow movement up
    float retractFeed = {};
    /// speed of regular milling
    float baseFeed = {};
    /// z-coordinate of plane where tool can move in any direction without touching the object
    float safeZ = {};
    /// which direction isolines or sections should be passed in
    BypassDirection bypassDir = BypassDirection::Clockwise;
    /// mesh can be transformed using xf parameter
    const AffineXf3f* xf = nullptr;
    /// if true then a tool path for a flat milling tool will be generated
    bool flatTool = false;
    /// callback for reporting on progress
    ProgressCallback cb = {};

    /// if > 0 - expand the trajectory creation area and create toolpath to mill excess material to make empty areas;
    /// the area has the shape of a box; lacing specific only
    float toolpathExpansion = 0.f;

    /// optional output, stores isolines without transits
    Contours3f* isolines = nullptr;
    /// optional output, polyline containing start vertices for isolines
    Contours3f* startContours = nullptr;
    /// start vertices on the offset mesh used for calculating isolines
    std::vector<Vector3f>* startVertices = nullptr;

    /// optional pre-computed offset mesh; if null, it will be computed internally
    MeshPart* offsetMesh = nullptr;
};

/// Tool path parameters specific to the constant-cusp strategy
struct ConstantCuspParams : ToolPathParams
{
    /// if true isolines will be processed from center point to the boundary (usually it means from up to down)
    bool fromCenterToBoundary = true;
};

/// Tolerance parameters for linear interpolation of tool path segments
struct LineInterpolationParams
{
    /// maximal deviation from given line
    float eps = {};
    /// maximal length of the line
    float maxLength = {};
    /// callback for reporting on progress
    ProgressCallback cb = {};
};

/// Tolerance parameters for arc interpolation of tool path segments
struct ArcInterpolationParams
{
    /// maximal deviation of arc from given path
    float eps = {};
    /// maximal radius of the arc
    float maxRadius = {};
    /// callback for reporting on progress
    ProgressCallback cb = {};
};

/// G-code move command type
enum class MoveType
{
    None       = -1, ///< No movement
    FastLinear =  0, ///< G0 rapid positioning
    Linear     =  1, ///< G1 linear interpolation at feed rate
    ArcCW      =  2, ///< G2 circular interpolation, clockwise
    ArcCCW     =  3  ///< G3 circular interpolation, counter-clockwise
};

/// G-code arc plane selection
enum class ArcPlane
{
    None = -1, ///< No plane selected
    XY   = 17, ///< G17 — XY plane
    XZ   = 18, ///< G18 — XZ plane
    YZ   = 19  ///< G19 — YZ plane
};

/// A single G-code move command
struct GCommand
{
    /// type of command GX (G0, G1, etc). By default - G1
    MoveType type = MoveType::Linear;
    /// plane of the arc; only relevant when type is ArcCW or ArcCCW
    ArcPlane arcPlane = ArcPlane::None;
    /// feedrate for move
    float feed = std::numeric_limits<float>::quiet_NaN();
    /// coordinates of destination point
    float x = std::numeric_limits<float>::quiet_NaN();
    float y = std::numeric_limits<float>::quiet_NaN();
    float z = std::numeric_limits<float>::quiet_NaN();
    /// if type is ArcCW or ArcCCW center of the arc should be specified
    Vector3f arcCenter = Vector3f::diagonal( std::numeric_limits<float>::quiet_NaN() );
};

/// Result of a tool path computation
struct ToolPathResult
{
    /// mesh after fixing undercuts and offset
    Mesh modifiedMesh;
    /// selected region projected from the original mesh to the offset
    FaceBitSet modifiedRegion;
    /// contains type of movement and its feed
    std::vector<GCommand> commands;
};

/// Computes a Z-level (constant-Z) milling tool path for the given mesh.
/// Sections are built along the Z-axis from top to bottom.
/// The mesh can be transformed using params.xf.
/// \param mp input mesh part to mill
/// \param params milling parameters (offsets, feeds, voxel size, etc.)
/// \return tool path result, or an error string on failure
MRVOXELS_API Expected<ToolPathResult> constantZToolPath( const MeshPart& mp, const ToolPathParams& params );

/// Computes a lacing (raster) milling tool path for the given mesh.
/// This is a traditional lace-roughing toolpath; slices are built along the axis defined by cutDirection.
/// \param mp input mesh part to mill
/// \param params milling parameters (offsets, feeds, voxel size, etc.)
/// \param cutDirection axis along which slices are built (Axis::X or Axis::Y)
/// \return tool path result, or an error string on failure
MRVOXELS_API Expected<ToolPathResult> lacingToolPath( const MeshPart& mp, const ToolPathParams& params, Axis cutDirection );

/// Computes a constant-cusp milling tool path for the given mesh.
/// The toolpath is built from geodesic parallels diverging from a given start point or from the boundaries of selected areas.
/// If neither is specified, the lowest section by XY plane will be used as a start contour.
/// The mesh can be transformed using params.xf.
/// \param mp input mesh part to mill
/// \param params milling and cusp parameters
/// \return tool path result, or an error string on failure
MRVOXELS_API Expected<ToolPathResult> constantCuspToolPath( const MeshPart& mp, const ConstantCuspParams& params );

/// Generates a G-Code object from a list of G-code commands
/// \param commands list of G-code move commands to export
/// \return shared pointer to the resulting G-code object
MRVOXELS_API std::shared_ptr<ObjectGcode> exportToolPathToGCode( const std::vector<GCommand>& commands );

/// Interpolates runs of collinear points with a single linear move
/// \param commands G-code command list to simplify in place
/// \param params line interpolation tolerance parameters
/// \param axis principal axis used for interpolation
/// \return error string on failure
MRVOXELS_API Expected<void> interpolateLines( std::vector<GCommand>& commands, const LineInterpolationParams& params, Axis axis );

/// Interpolates the given path with circular arc moves
/// \param commands G-code command list to simplify in place
/// \param params arc interpolation tolerance parameters
/// \param axis principal axis used for interpolation
/// \return error string on failure
MRVOXELS_API Expected<void> interpolateArcs( std::vector<GCommand>& commands, const ArcInterpolationParams& params, Axis axis );

/// Smooths the given face selection by expanding the boundary outward and then shrinking it back.
/// The input mesh is modified because new edges are cut along the new boundaries.
/// \param mesh input mesh, modified in place
/// \param region face selection to smooth
/// \param expandOffset distance by which the boundary is expanded outward
/// \param shrinkOffset distance by which the boundary is shrunk back inward
/// \return the smoothed face selection
MRVOXELS_API FaceBitSet smoothSelection( Mesh& mesh, const FaceBitSet& region, float expandOffset, float shrinkOffset );

}
