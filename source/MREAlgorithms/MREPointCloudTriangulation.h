#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRConstants.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRProgressCallback.h"

namespace MRE
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
     * \brief Average number of neighbors for optimazed local triangulation
     * \details The triangulation calculates the radius at which the average
     * number of neighboring points is closest to this parameter.
     * This radius is used to determine the local triangulation zone.
     * \note Too small value can make not optimal triangulation and additional holes\n
     * Too big value increases difficulty of optimization and can make local optimum of local triangulation
     *
        <table border=0>
            <caption id="TriangulationParameters::avgNumNeighbours_examples"></caption>
            <tr>
                <td> \image html triangulate/triangulate_3.png "Good" width = 350cm </td>
                <td> \image html triangulate/triangulate_2.png "Too small value" width = 350cm </td>
            </tr>
        </table>
     */
    int avgNumNeighbours{40};
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
    float critAngle{MR::PI2_F};
    /**
     * \brief Critical length of hole (all holes with length less then this value will be filled)
     * \details If value is subzero it is set automaticly to 0.7*bbox.diagonal()
     */
    float critHoleLength{-FLT_MAX};
};

/**
 * \brief Creates mesh from given point cloud according params
 * \ingroup PointCloudTriangulationGroup
    <table border=0>
        <caption id="triangulatePointCloud_examples"></caption>
        <tr>
            <td> \image html triangulate/triangulate_0.png "Before" width = 350cm </td>
            <td> \image html triangulate/triangulate_3.png "After" width = 350cm </td>
        </tr>
    </table>
 */
MREALGORITHMS_API MR::Mesh triangulatePointCloud( const MR::PointCloud& pointCloud, const TriangulationParameters& params = {},
    MR::SimpleProgressCallback progressCb = {} );
}