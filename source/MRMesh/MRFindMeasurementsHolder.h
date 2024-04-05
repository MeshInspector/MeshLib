#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRSceneRoot.h"

namespace MR
{

// Those functions let you determine where the measurements of an object should be placed.
// The current behavior is to create a "Measurements" subobject under each individual object.

// Finds the data model object that should hold the measurements, or creates it if it doesn't exit.
[[nodiscard]] MRMESH_API std::shared_ptr<Object> findOrCreateMeasurementsHolder( Object& object );
// Finds the data model object that should hold the measurements, or returns zero.
[[nodiscard]] MRMESH_API std::shared_ptr<Object> findMeasurementsHolderOpt( Object& object );

}
