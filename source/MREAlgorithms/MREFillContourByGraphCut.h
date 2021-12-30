#pragma once

#include "exports.h"
#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MREdgePaths.h>
#include <vector>

namespace MRE
{
/**
 * \defgroup MeshSegmentationGroup Mesh Segmentation overwiev
 * \brief A set of functions for selecting segments on a mesh
 * 
 * \subsection MeshSegmentationGroup_HowTo How To
 * 1. Make Contour(-s)\n
 * First step to get a mesh segment by two (/three) points, is getting the contour that bounds the segment.\n
 * To do this, you can use \ref surroundingContour
 * 2. Get Segment\n
 * For getting segment of mesh by contour(-s) use \ref fillContourLeftByGraphCut.
 *
 * <table border=0> <caption id="fillContourLeftByGraphCut_examples"></caption>
 * <tr> <td> \image html segmentation/mesh_segmentation_0.png "Before" width = 350cm </td>
 *      <td> \image html segmentation/mesh_segmentation_1.png "After" width = 350cm </td> </tr>
 * </table>
 */

/**
 * \brief Fill region located to the left from given contour, by minimizing the sum of metric over the boundary
 * \ingroup MeshSegmentationGroup
 */
MREALGORITHMS_API MR::FaceBitSet fillContourLeftByGraphCut( const MR::MeshTopology & topology, const MR::EdgePath & contour,
    const MR::EdgeMetric & metric );

/**
 * \brief Fill region located to the left from given contours, by minimizing the sum of metric over the boundary
 * \ingroup MeshSegmentationGroup
 */
MREALGORITHMS_API MR::FaceBitSet fillContourLeftByGraphCut( const MR::MeshTopology & topology, const std::vector<MR::EdgePath> & contours,
    const MR::EdgeMetric & metric );

} //namespace MRE
