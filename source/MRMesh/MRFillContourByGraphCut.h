#pragma once

#include "MREdgePaths.h"
#include <vector>

namespace MR
{
/**
 * \defgroup MeshSegmentationGroup Mesh Segmentation overwiev
 * \brief A set of functions for selecting segments on a mesh
 * 
 * \section MeshSegmentationGroup_HowTo How To
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
 * \brief Fills region located to the left from given contour, by minimizing the sum of metric over the boundary
 * If the computations are terminated by \p progress, then returns the best approximation found by the moment of termination
 * \ingroup MeshSegmentationGroup
 */
MRMESH_API FaceBitSet fillContourLeftByGraphCut( const MeshTopology & topology, const EdgePath & contour,
    const EdgeMetric & metric, const ProgressCallback& progress = {} );

/**
 * \brief Fills region located to the left from given contours, by minimizing the sum of metric over the boundary
 * If the computations are terminated by \p progress, then returns the best approximation found by the moment of termination
 * \ingroup MeshSegmentationGroup
 */
MRMESH_API FaceBitSet fillContourLeftByGraphCut( const MeshTopology & topology, const std::vector<EdgePath> & contours,
    const EdgeMetric & metric, const ProgressCallback& progress = {} );

/**
 * \brief Finds segment that divide mesh on source and sink (source included, sink excluded), by minimizing the sum of metric over the boundary
 * If the computations are terminated by \p progress, then returns the best approximation found by the moment of termination
 * \ingroup MeshSegmentationGroup
 */
MRMESH_API FaceBitSet segmentByGraphCut( const MeshTopology& topology, const FaceBitSet& source,
    const FaceBitSet& sink, const EdgeMetric& metric, const ProgressCallback& progress = {} );

} //namespace MR
