#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRPolyline.h"

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

enum class MoveType
{
    FastLinear = 0,
    Linear = 1,
    ArcCW = 2,
    ArcCCW = 3
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
};

struct ToolPathResult
{
    // mesh after fixing undercuts and offset
    Mesh modifiedMesh;
    // path of the milling tool
    Polyline3 toolPath;
    // constains type of movement and its feed
    std::vector<GCommand> commands;
};

// compute path of the milling tool for the given mesh with parameters ( direction of milling is from up to down )
// mesh can be transformed using xf parameter
MRMESH_API ToolPathResult constantZToolPath( const Mesh& mesh, const ToolPathParams& params, const AffineXf3f* xf );

// generates G-Code for milling tool
MRMESH_API std::string exportToolPathToGCode( const std::vector<GCommand>& commands );

MRMESH_API void interpolateArcs( std::vector<GCommand>& commands, float eps, float maxRadius );

}
#endif
