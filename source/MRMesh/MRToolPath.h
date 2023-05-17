#pragma once
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "exports.h"
#include "MRMeshFwd.h"
#include "MRPolyline.h"


namespace MR
{

struct ToolPathResult
{
    std::shared_ptr<Polyline3> toolPath;
    std::string gcode;
};

MRMESH_API ToolPathResult getToolPath( Mesh& mesh, float millRadius, float voxelSize, float sectionStep, float critLength = 0.1f,
    float plungeLength = 0.1f, float retractLength = 0.1f,
    float plungeFeed = 500.0f, float retractFeed = 500.0f );

}
#endif
