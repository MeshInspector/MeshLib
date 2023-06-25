#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRExpected.h"
#include <filesystem>

namespace MR
{

/**
 * \brief Class for making mesh from regular distance map
 * \details distance for each lattice of map is defined as 
 * point on ray from surface point cloud to direction point cloud,
 * with length equal to 1/distance from distances file 
 * (if distance in file == 0 ray had no point) 
 * \ingroup PointCloudGroup
 */
class RegularMapMesher
{
public:
    /// Loads surface Point Cloud form file
    MRMESH_API VoidOrErrStr loadSurfacePC( const std::filesystem::path& path );
    /// Sets surface Point Cloud
    MRMESH_API void setSurfacePC( const std::shared_ptr<PointCloud>& surfacePC );
    /// Loads directions Point Cloud from file
    MRMESH_API VoidOrErrStr loadDirectionsPC( const std::filesystem::path& path );
    /// Sets directions Point Cloud
    MRMESH_API void setDirectionsPC( const std::shared_ptr<PointCloud>& directionsPC );
    /// Loads distances form distances file (1/distance)
    MRMESH_API VoidOrErrStr loadDistances( int width, int height, const std::filesystem::path& path );
    /// Sets distances
    MRMESH_API void setDistances( int width, int height, const std::vector<float>& distances );

    /// Creates mesh if all components were successfully loaded
    MRMESH_API Expected<Mesh, std::string> createMesh() const;

private:
    int width_{0};
    int height_{0};

    std::shared_ptr<PointCloud> surfacePC_;
    std::shared_ptr<PointCloud> directionsPC_;
    std::vector<float> distances_;
};
}