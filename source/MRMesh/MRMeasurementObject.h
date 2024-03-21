#pragma once

#include "MRMesh/MRVisualObject.h"

namespace MR
{

// A common base class for measurement objects.
// Can't be constructed directly.
class MeasurementObject : public VisualObject
{
protected:
    MRMESH_API MeasurementObject();
};

}
