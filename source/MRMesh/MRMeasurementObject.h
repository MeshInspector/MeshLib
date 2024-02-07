#pragma once

#include "MRMesh/MRObject.h"

namespace MR
{

// A common base class for measurement objects.
// Can't be constructed directly.
class MeasurementObject : public Object
{
protected:
    MeasurementObject() = default;
};

}
