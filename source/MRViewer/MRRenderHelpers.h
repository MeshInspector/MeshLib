#pragma once
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector2.h"

namespace MR
{

// calc texture resolution, to fit MAX_TEXTURE_SIZE, and have minimal empty pixels
Vector2i calcTextureRes( int bufferSize, int maxTextWidth );

}
