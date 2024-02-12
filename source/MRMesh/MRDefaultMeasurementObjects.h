#pragma once

#include "MRMesh/MRFlagOperators.h"
#include "MRMesh/MRMeshFwd.h"

namespace MR
{

enum class DefaultMeasurementKinds
{
    Radiuses = 1 << 0,
    Distances = 1 << 1,
    Angles = 1 << 2,
    All = Radiuses | Distances | Angles,
};
MR_MAKE_FLAG_OPERATORS( DefaultMeasurementKinds )

struct AttachDefaultMeasurementsParams
{
    // If the measurements already exist, remove them and attach new ones.
    // Otherwise the existing measurements are left alone, even if they have some incorrect parameters.
    bool overwrite = false;

    // If a measurement kind is not in this list, it's left alone.
    DefaultMeasurementKinds enabledKinds = DefaultMeasurementKinds::All;
    // If a measurement kind is not in this list, it's created invisible by default.
    DefaultMeasurementKinds defaultVisibleKinds = DefaultMeasurementKinds::All;
};

// Given a feature object, attach all necessary measurement subobjects to it.
// They then automatically track the object dimensions.
MRMESH_API void attachDefaultMeasurementsToObject( Object& object, const AttachDefaultMeasurementsParams& params = {} );

}
