#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRBitSet.h"

namespace MR
{

// Sample vertices, removing ones that are too close
MRMESH_API VertBitSet pointUniformSampling( const PointCloud& pointCloud, float distance );

}