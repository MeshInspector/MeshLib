#include "MRMesh/MRPython.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRVoxelsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRLinesLoad.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRLog.h"

using namespace MR;

bool pythonSaveMeshToAnyFormat( const Mesh& mesh, const std::string& path )
{
    auto res = MR::MeshSave::toAnySupportedFormat( mesh, path );
    return res.has_value();
}

Mesh pythonLoadMeshFromAnyFormat( const std::string& path )
{
    auto res = MR::MeshLoad::fromAnySupportedFormat( path );
    if ( res.has_value() )
        return *res;
    return {};
}

bool pythonSaveLinesToAnyFormat( const MR::Polyline& lines, const std::string& path )
{
    auto res = MR::LinesSave::toAnySupportedFormat( lines, path );
    return res.has_value();
}

MR::Polyline pythonLoadLinesFromAnyFormat( const std::string& path )
{
    auto res = MR::LinesLoad::fromAnySupportedFormat( path );
    if ( res.has_value() )
        return *res;
    return {};
}

bool pythonSavePointCloudToAnyFormat( const PointCloud& points, const std::string& path )
{
    auto res = MR::PointsSave::toAnySupportedFormat( points, path );
    return res.has_value();
}

PointCloud pythonLoadPointCloudFromAnyFormat( const std::string& path )
{
    auto res = MR::PointsLoad::fromAnySupportedFormat( path );
    if ( res.has_value() )
        return *res;
    return {};
}

MR_ADD_PYTHON_FUNCTION( mrmeshpy, save_mesh, pythonSaveMeshToAnyFormat, "saves mesh in file of known format/extension" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, load_mesh, pythonLoadMeshFromAnyFormat, "load mesh of known format" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, save_lines, pythonSaveLinesToAnyFormat, "saves lines in file of known format/extension" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, load_lines, pythonLoadLinesFromAnyFormat, "load lines of known format" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, save_points, pythonSavePointCloudToAnyFormat, "saves point cloud in file of known format/extension" )
MR_ADD_PYTHON_FUNCTION( mrmeshpy, load_points, pythonLoadPointCloudFromAnyFormat, "load point cloud of known format" )
