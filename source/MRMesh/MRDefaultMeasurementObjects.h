#pragma once

#include "MRMesh/MRMeshFwd.h"

namespace MR
{

struct AttachDefaultMeasurementsParams
{
    // If the measurements already exist, remove them and attach new ones.
    // Otherwise the existing measurements are left alone, even if they have some incorrect parameters.
    bool overwrite = false;
};

// Given a feature object, attach all necessary measurement subobjects to it.
// They then automatically track the object dimensions.
MRMESH_API void attachDefaultMeasurementsToObject( Object& object, const AttachDefaultMeasurementsParams& params = {} );

}
