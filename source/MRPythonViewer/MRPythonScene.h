#pragma once

#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPolyline.h"
#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRVoxelsVolume.h"
#endif

#include <memory>
#include <string>
#include <vector>

namespace MR::PythonViewer
{

/// Add given mesh to scene tree.
void addMeshToScene( const Mesh& mesh, const std::string& name );

/// Add given point cloud to scene tree.
void addPointCloudToScene( const PointCloud& points, const std::string& name );

/// Add given lines to scene tree.
void addLinesToScene( const Polyline3& lines, const std::string& name );

/// Add given distance map to scene tree.
void addDistanceMapToScene( const DistanceMap& distancemap, const std::string& name, const AffineXf3f& dmap_to_local_xf );

/// Assign a new mesh to the selected mesh object. Exactly one object must be selected.
void modifySelectedMesh( const Mesh& mesh );

/// Get selected face bitsets of the selected mesh objects.
[[nodiscard]] std::vector<FaceBitSet> getSelectedMeshFaces();

/// Get selected edge bitsets of the selected mesh objects.
[[nodiscard]] std::vector<UndirectedEdgeBitSet> getSelectedMeshEdges();

/// Get selected point bitsets of the selected point cloud objects.
[[nodiscard]] std::vector<VertBitSet> getSelectedPointCloudPoints();

/// Set selected face bitsets of the selected mesh objects.
void setSelectedMeshFaces( const std::vector<FaceBitSet>& bitsets );

/// Set selected edge bitsets of the selected mesh objects.
void setSelectedMeshEdges( const std::vector<UndirectedEdgeBitSet>& bitsets );

/// Set selected point bitsets of the selected point cloud objects.
void setSelectedPointCloudPoints( const std::vector<VertBitSet>& bitsets );

/// remove all objects from scene tree
void clearScene();

/// select objects in scene tree with given name, unselect others
void selectByName( const std::string& objectName );

/// string typeName: {"Meshes", "Points", "Voxels"}
/// objects in scene tree with given type, unselect others
void selectByType( const std::string& typeName );

/// unselect all objects in scene tree
void unselectAll();

[[nodiscard]] std::vector<std::shared_ptr<Object>> getSelectedObjects();

/// Get copies of all selected meshes in the scene.
[[nodiscard]] std::vector<Mesh> getSelectedMeshes();

/// Get copies of all selected point clouds in the scene.
[[nodiscard]] std::vector<PointCloud> getSelectedPointClouds();

/// Get copies of all selected polylines in the scene.
[[nodiscard]] std::vector<Polyline3> getSelectedPolylines();

/// Get copies of all selected voxel grids in the scene.
[[nodiscard]] std::vector<DistanceMap> getSelectedDistanceMaps();

#ifndef MESHLIB_NO_VOXELS
/// Add given voxels to scene tree.
void addVoxelsToScene( const VdbVolume& voxels, const std::string& name );

/// Get copies of all selected voxel grids in the scene.
[[nodiscard]] std::vector<VdbVolume> getSelectedVoxels();
#endif

} // namespace MR::PythonViewer
