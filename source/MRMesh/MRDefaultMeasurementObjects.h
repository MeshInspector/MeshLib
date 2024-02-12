#pragma once

#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRMeshFwd.h"

namespace MR
{

enum class AttachDefaultMeasurementsFlags
{
    // If the measurements already exist, remove them and attach new ones.
    // Otherwise the existing measurements are left alone, even if they have some incorrect parameters.
    Overwrite = 1 << 0,
    Radiuses = 1 << 1,
    Distances = 1 << 2,
    Angles = 1 << 3,
    AllKinds = Radiuses | Distances | Angles,
};
MR_MAKE_FLAG_OPERATORS( AttachDefaultMeasurementsFlags )

// Given a feature object, attach all necessary measurement subobjects to it.
// They then automatically track the object dimensions.
MRMESH_API void attachDefaultMeasurementsToObject( Object& object, AttachDefaultMeasurementsFlags flags = AttachDefaultMeasurementsFlags::AllKinds );

}
