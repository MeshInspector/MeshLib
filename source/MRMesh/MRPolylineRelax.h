#pragma once
#include "MRMeshFwd.h"
#include "MRMeshRelax.h"

namespace MR
{

/// \addtogroup PolylineGroup
/// \{

/// applies given number of relaxation iterations to the whole polyline ( or some region if it is specified )
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relax( Polyline2& polyline, const RelaxParams& params = {}, ProgressCallback cb = {} );
MRMESH_API bool relax( Polyline3& polyline, const RelaxParams& params = {}, ProgressCallback cb = {} );

/// applies given number of relaxation iterations to the whole polyline ( or some region if it is specified )
/// do not really keeps area but tries hard
/// \return true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relaxKeepArea( Polyline2& polyline, const RelaxParams& params = {}, ProgressCallback cb = {} );
MRMESH_API bool relaxKeepArea( Polyline3& polyline, const RelaxParams& params = {}, ProgressCallback cb = {} );

/// \}

} // namespace MR
