#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMeshFwd.h"
#include "MRAxis.h"
#include "MRMesh.h"
#include "MRPolyline.h"
#include "MRProgressCallback.h"

#include "MRExpected.h"

namespace MR
{

enum class BypassDirection
{
    Clockwise,
    CounterClockwise
};

struct ToolPathParams
{
    // radius of the milling tool
    float millRadius = {};
    // size of voxel needed to offset mesh
    float voxelSize = {};
    // distance between sections built along Z axis
    float sectionStep = {};
    // if distance to the next section is smaller than it, transition will be performed along the surface
    // otherwise transition will be through the safe plane
    float critTransitionLength = {};
    // when the mill is moving down, it will be slowed down in this distance from mesh
    float plungeLength = {};
    // when the mill is moving up, it will be slowed down in this distance from mesh
    float retractLength = {};
    // speed of slow movement down
    float plungeFeed = {};
    // speed of slow movement up
    float retractFeed = {};
    // speed of regular milling
    float baseFeed = {};
    // z-coordinate of plane where tool can move in any direction without touching the object
    float safeZ = {};
    // which direction isolines or sections should be passed in
    BypassDirection bypassDir = BypassDirection::Clockwise;
    // mesh can be transformed using xf parameter
    const AffineXf3f* xf = nullptr;

    bool flatTool = false;
    // callback for reporting on progress
    ProgressCallback cb = {};
};

struct ConstantCuspParams : ToolPathParams
{
    // if true isolines will be processed from center point to the boundary (usually it means from up to down)
    bool fromCenterToBoundary = true;
};

struct LineInterpolationParams
{
    // maximal deviation from given line
    float eps = {};
    // maximal length of the line
    float maxLength = {};
};

struct ArcInterpolationParams
{
    // maximal deviation of arc from given path
    float eps = {};
    // maximal radius of the arc
    float maxRadius = {};
};

enum class MoveType
{
    None = -1,
    FastLinear = 0,
    Linear = 1,
    ArcCW = 2,
    ArcCCW = 3
};

enum class ArcPlane
{
    None = -1,
    XY = 17,
    XZ = 18,
    YZ = 19
};

struct GCommand
{
    // type of command GX (G0, G1, etc). By default - G1
    MoveType type = MoveType::Linear;
    // Place for comment
    ArcPlane arcPlane = ArcPlane::None;
    // feedrate for move
    float feed = std::numeric_limits<float>::quiet_NaN();
    // coordinates of destination point
    float x = std::numeric_limits<float>::quiet_NaN();
    float y = std::numeric_limits<float>::quiet_NaN();
    float z = std::numeric_limits<float>::quiet_NaN();
    // if moveType is ArcCW or ArcCCW center of the arc shoult be specified
    Vector3f arcCenter = Vector3f::diagonal( std::numeric_limits<float>::quiet_NaN() );
};

struct ToolPathResult
{
    // mesh after fixing undercuts and offset
    Mesh modifiedMesh;
    // constains type of movement and its feed
    std::vector<GCommand> commands;
};

// compute path of the milling tool for the given mesh with parameters ( direction of milling is from up to down along Z-direction )
// this toolpath is built from the parallel sections along Z-axis
// mesh can be transformed using xf parameter

MRMESH_API Expected<ToolPathResult, std::string> constantZToolPath( const MeshPart& mp, const ToolPathParams& params );


MRMESH_API Expected<ToolPathResult, std::string>  constantZToolPathWithFlatTool( const MeshPart& mp, const ToolPathParams& params );


// compute path of the milling tool for the given mesh with parameters ( direction of milling is from up to down along Z-direction )
// // this one is traditional lace-roughing toolpath

// Slices are built along the axis defined by cutDirection argument (can be Axis::X or Axis::Y)
MRMESH_API Expected<ToolPathResult, std::string> lacingToolPath( const MeshPart& mp, const ToolPathParams& params, Axis cutDirection );

// compute path of the milling tool for the given mesh with parameters ( direction of milling is from up to down along Z-direction )
// this toolpath is built from geodesic parallels divercing from the given start point or from the bounaries of selected areas
// if neither is specified, the lowest section by XY plane will be used as a start contour
// mesh can be transformed using xf parameter
MRMESH_API Expected<ToolPathResult, std::string> constantCuspToolPath( const MeshPart& mp, const ConstantCuspParams& params );

// generates G-Code for milling tool
MRMESH_API std::shared_ptr<ObjectGcode> exportToolPathToGCode( const std::vector<GCommand>& commands );

// interpolates several points lying on the same straight line with one move
MRMESH_API void interpolateLines( std::vector<GCommand>& commands, const LineInterpolationParams& params, Axis axis );
// interpolates given path with arcs
MRMESH_API void interpolateArcs( std::vector<GCommand>& commands, const ArcInterpolationParams& params, Axis axis );

}
#endif
