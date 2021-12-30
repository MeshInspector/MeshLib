#pragma once
#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{
	// creates hollow arrow from the 'base' to the 'vert'. Number of points on the circle 'qual' is between 3 and 256
	MRMESH_API Mesh makeArrow(const Vector3f& base, const Vector3f& vert, const float& thickness = 0.05f, const float& coneRadius = 0.1f, const float coneSize = 0.2f, const int qual = 32);
	// creates the mesh with 3 axis arrows
	MRMESH_API Mesh makeBasisAxes(const float& size = 1.0f, const float& thickness = 0.05f, const float& coneRadius = 0.1f, const float coneSize = 0.2f, const int qual = 32);
}
