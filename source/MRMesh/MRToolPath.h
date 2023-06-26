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
    FastLinear = 0,
    Linear = 1,
    ArcCW = 2,
    ArcCCW = 3,
    PlaneSelectionXY = 17,
    PlaneSelectionXZ = 18,
    PlaneSelectionYZ = 19
};



struct GCommand
{
    // type of command GX (G0, G1, etc). By default - G1
    MoveType type = MoveType::Linear;
    // feedrate for move
    float feed = std::numeric_limits<float>::quiet_NaN();
    // 
    float x = std::numeric_limits<float>::quiet_NaN();
    float y = std::numeric_limits<float>::quiet_NaN();
    float z = std::numeric_limits<float>::quiet_NaN();

    float i = std::numeric_limits<float>::quiet_NaN();
    float j = std::numeric_limits<float>::quiet_NaN();
    float k = std::numeric_limits<float>::quiet_NaN();

    // get coordinate along specified axis
    float coord( Axis axis ) const;
    // get projection to the plane orthogonal to the specified axis
    Vector2f project( Axis axis ) const;
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

MRMESH_API Expected<ToolPathResult, std::string> constantZToolPath( const MeshPart& mp, const ToolPathParams& params, const AffineXf3f* xf = nullptr, ProgressCallback cb = {} );


// compute path of the milling tool for the given mesh with parameters ( direction of milling is from up to down along Z-direction )
// // this one is traditional lace-roughing toolpath
// mesh can be transformed using xf parameter
MRMESH_API Expected<ToolPathResult, std::string> lacingToolPath( const Mesh& mesh, const ToolPathParams& params, const AffineXf3f* xf = nullptr, ProgressCallback cb = {} );

// compute path of the milling tool for the given mesh with parameters ( direction of milling is from up to down along Z-direction )
// this toolpath is built from geodesic parallels divercing from the given start point
// if the start point is not specified, the highest point on the mesh will be used
// mesh can be transformed using xf parameter
MRMESH_API Expected<ToolPathResult, std::string> constantCuspToolPath( const Mesh& mesh, const ToolPathParams& params, VertId startPoint = {}, const AffineXf3f* xf = nullptr, ProgressCallback cb = {} );

// generates G-Code for milling tool
MRMESH_API std::shared_ptr<ObjectGcode> exportToolPathToGCode( const std::vector<GCommand>& commands );

// interpolates several points lying on the same straight line with one move
MRMESH_API void interpolateLines( std::vector<GCommand>& commands, const LineInterpolationParams& params, Axis axis );
// interpolates given path with arcs
MRMESH_API void interpolateArcs( std::vector<GCommand>& commands, const ArcInterpolationParams& params, Axis axis );

}
#endif
