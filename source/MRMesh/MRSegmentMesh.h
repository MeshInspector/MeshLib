#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRFaceFace.h"

namespace MR
{

using GroupOrder = std::vector<FaceFace>;

MRMESH_API Expected<GroupOrder> segmentMesh( const Mesh& mesh,
    const EdgeMetric& metric ); ///< sum of this metric to be minimized for the boundaries of segments

} //namespace MR
