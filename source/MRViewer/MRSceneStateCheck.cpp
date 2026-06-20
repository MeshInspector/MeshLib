#include "MRSceneStateCheck.h"

#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"

// Explicit instantiations for the templates declared `extern template` in MRSceneStateCheck.h.
// Defining them here once keeps every consuming TU from re-generating this code.

namespace MR
{

template std::string getNObjectsLine<ObjectMesh>( unsigned );
template std::string getNObjectsLine<ObjectPoints>( unsigned );
template std::string getNObjectsLine<ObjectLines>( unsigned );

template std::string sceneSelectedExactly<ObjectMesh, true, true>( const std::vector<std::shared_ptr<const Object>>&, unsigned );
template std::string sceneSelectedExactly<ObjectPoints, true, true>( const std::vector<std::shared_ptr<const Object>>&, unsigned );
template std::string sceneSelectedExactly<ObjectLines, true, true>( const std::vector<std::shared_ptr<const Object>>&, unsigned );

} //namespace MR
