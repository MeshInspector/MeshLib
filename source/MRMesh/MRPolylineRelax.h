#pragma once
#include "MRMeshFwd.h"
#include "MRMeshRelax.h"

namespace MR
{

/// \addtogroup PolylineGroup
/// \{

struct PolylineRelaxParams : RelaxParams
{
    //
};

/// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relax( Polyline3& polyline, const PolylineRelaxParams& params = {}, ProgressCallback cb = {} );

/// applies given number of relaxation iterations to the whole pointCloud ( or some region if it is specified )
/// do not really keeps volume but tries hard
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relaxKeepVolume( Polyline3& polyline, const PolylineRelaxParams& params = {}, ProgressCallback cb = {} );

/// \}

} // namespace MR
