#pragma once

#include "MRConstants.h"
#include "MRMesh.h"
#include "MRProgressCallback.h"
#include <optional>

namespace MR
{
/**
 * \defgroup PointCloudTriangulationGroup PointCloud triangulation overview
 * \brief This chapter represents documentation about PointCloud triangulation
 */

/**
 * \brief Parameters of point cloud triangulation
 * \ingroup PointCloudTriangulationGroup
 *
 * \sa \ref triangulatePointCloud
 */
struct TriangulationParameters
{
    /**
     * \brief The number of nearest neighbor points to use for building of local triangulation
     * \note Too small value can make not optimal triangulation and additional holes\n
     * Too big value increases difficulty of optimization and decreases performance
     *
        <table border=0>
            <caption id="TriangulationParameters::numNeighbours_examples"></caption>
            <tr>
                <td> \image html triangulate/triangulate_3.png "Good" width = 350cm </td>
                <td> \image html triangulate/triangulate_2.png "Too small value" width = 350cm </td>
            </tr>
        </table>
     */
    int numNeighbours = 16;

    /**
     * Radius of neighborhood around each point to consider for building local triangulation.
     * This is an alternative to numNeighbours parameter.
     * Please set to positive value only one of them.
     */
    float radius = 0;

    /**
     * \brief Critical angle of triangles in local triangulation (angle between triangles in fan should be less then this value)
     *
        <table border=0>
            <caption id="TriangulationParameters::critAngle_examples"></caption>
            <tr>
                <td> \image html triangulate/triangulate_3.png "Good" width = 350cm </td>
                <td> \image html triangulate/triangulate_4.png "Too small value" width = 350cm </td>
            </tr>
        </table>
     */
    float critAngle = PI2_F;

    /// the vertex is considered as boundary if its neighbor ring has angle more than this value
    float boundaryAngle = 0.9f * PI_F;

    /**
     * \brief Critical length of hole (all holes with length less then this value will be filled)
     * \details If value is subzero it is set automaticly to 0.7*bbox.diagonal()
     */
    float critHoleLength = -FLT_MAX;

    /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
    bool automaticRadiusIncrease = true;

    /// optional: if provided this cloud will be used for searching of neighbors (so it must have same validPoints)
    const PointCloud * searchNeighbors = nullptr;
};

/**
 * \brief Creates mesh from given point cloud according params
 * Returns empty optional if was interrupted by progress bar
 * \ingroup PointCloudTriangulationGroup
    <table border=0>
        <caption id="triangulatePointCloud_examples"></caption>
        <tr>
            <td> \image html triangulate/triangulate_0.png "Before" width = 350cm </td>
            <td> \image html triangulate/triangulate_3.png "After" width = 350cm </td>
        </tr>
    </table>
 */
MRMESH_API std::optional<Mesh> triangulatePointCloud( const PointCloud& pointCloud, const TriangulationParameters& params = {},
    ProgressCallback progressCb = {} );

} //namespace MR
