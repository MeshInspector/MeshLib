#pragma once
#include "MRMeshFwd.h"

namespace MR
{
	// Z is polar axis of this UVSphere
	MRMESH_API MR::Mesh makeUVSphere(float radius = 1.0, int horisontalResolution = 16, int verticalResolution = 16);
}