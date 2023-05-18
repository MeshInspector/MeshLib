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
    // distance between sections
    float sectionStep = {};
    // if distance to the next section is smaller than it, transition will be performed along the surface
    // Otherwise transition will be through the safe plane
    float critTransitionLength = {};
    //When the mill is moving down, it will be slowed down in this distance from mesh
    float plungeLength = {};
    //When the mill is moving up, it will be slowed down in this distance from mesh
    float retractLength = {};
    //Speed of slow movement down
    float plungeFeed = {};
    //Speed of slow movement up
    float retractFeed = {};
};

struct GCommand
{
    // type of command GX (G0, G1, etc). By default - G1
    int type = 1;
    // feedrate for move
    float feed = 0;
};

struct ToolPathResult
{
    // mesh after fixing undercuts and offset
    Mesh modifiedMesh;
    // path of the milling tool
    std::shared_ptr<Polyline3> toolPath;
    // constains type of movement and its feed
    std::vector<GCommand> commands;
};

// compute path of the milling tool for the given mesh with parameters. Direction of milling is from up to down. 
// Mesh can be transformed using xf parameter
MRMESH_API ToolPathResult constantZToolPath( const Mesh& mesh, const AffineXf3f& xf, const ToolPathParams& params );

// generates G-Code for milling tool
MRMESH_API std::string exportToolPathToGCode( const Polyline3& toolPath, const std::vector<GCommand>& commands );

}
#endif
