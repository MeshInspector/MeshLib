#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "exports.h"
#include "MRMeshFwd.h"
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

struct ToolPathResult
{
    // mesh after fixing undercuts and offset
    std::shared_ptr<Mesh> modifiedMesh;
    // path of the milling tool
    std::shared_ptr<Polyline3> toolPath;
    // tool movements exported to G-Code
    std::string gcode;
};

// compute path of the milling tool for the given mesh with parameters. Direction of milling is from up to down. Mesh can be transformed
MRMESH_API ToolPathResult getToolPath( const Mesh& mesh, const AffineXf3f& xf, const ToolPathParams& params );

}
#endif
