#pragma once

#include "MRMesh/MRMeshFwd.h"

namespace MR
{

// Feature objects should call this in the constructor to configure some default visualization parameters.
MRMESH_API void setDefaultFeatureObjectParams( VisualObject& object );

}
