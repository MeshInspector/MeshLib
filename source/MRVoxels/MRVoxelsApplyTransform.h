#pragma once

#include "MRVoxelsFwd.h"
#include <MRMesh/MRBox.h>


namespace MR
{

/// Transform volume
/// @param volume Volume to transform
/// @param xf The transformation
/// @param fixBox If true, and if \p box is valid and represents the bounding box of the \p volume, then
///               the result will be shifted so that no data has negative coordinate by any of dimensions
MRVOXELS_API VdbVolume
    transformVdbVolume( const VdbVolume& volume, const AffineXf3f& xf, bool fixBox = false, const Box3f& box = {} );

/// Same as above but for the SceneObject
/// @return true, if \p fixBox is true and the box was "fixed" (see parameter `fixBox` of \ref transformVdbVolume)
MRVOXELS_API void voxelsApplyTransform( ObjectVoxels& obj, const AffineXf3f& xf, bool fixBox );

}

