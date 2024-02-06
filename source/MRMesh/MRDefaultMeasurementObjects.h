#pragma once

#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRMeshFwd.h"

namespace MR
{

enum class AttachMeasurementsFlags
{
    // If the measurements already exist, remove them and attach new ones.
    // Otherwise the existing measurements are left alone, even if they have some incorrect parameters.
    overwrite = 1 << 0,
};
MR_MAKE_FLAG_OPERATORS( AttachMeasurementsFlags )

// Given a feature object, attach all necessary measurement subobjects to it.
// They then automatically track the object dimensions.
MRMESH_API void attachDefaultMeasurementsToObject( Object& object, AttachMeasurementsFlags flags = {} );

}
