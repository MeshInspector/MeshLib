#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRMeshFwd.h"
#include <memory>
#include <vector>
#include <string>

namespace MR
{

// Interface for checking scene state, to determine availability, also can return string with requirements 
class ISceneStateCheck
{
public:
    virtual ~ISceneStateCheck() = default;
    // return empty string if all requirements are satisfied, otherwise return first unsatisfied requirement
    virtual std::string isAvailable( const std::vector<std::shared_ptr<const Object>>& ) const { return {}; }
};

} //namespace MR
