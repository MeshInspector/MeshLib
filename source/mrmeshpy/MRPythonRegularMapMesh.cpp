#include "MRMesh/MRPython.h"
#include "MRMesh/MRRegularMapMesher.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRLog.h"

using namespace MR;

MR::Mesh meshRegularMap( const char* surfacePCPath, const char* directionsPCPath,
                     const char* distancesPath, int width, int height )
{
    RegularMapMesher mesher;
    {
        auto res = mesher.loadSurfacePC( surfacePCPath );
        if ( !res.has_value() )
        {
            spdlog::error( res.error() );
            return {};
        }
    }
    {
        auto res = mesher.loadDirectionsPC( directionsPCPath );
        if ( !res.has_value() )
        {
            spdlog::error( res.error() );
            return {};
        }
    }
    {
        auto res = mesher.loadDistances( width, height, distancesPath );
        if ( !res.has_value() )
        {
            spdlog::error( res.error() );
            return {};
        }
    }
    auto resMesh = mesher.createMesh();
    if ( !resMesh.has_value() )
    {
        spdlog::error( resMesh.error() );
        return {};
    }

    return resMesh.value();
}


MR_ADD_PYTHON_FUNCTION( mrmeshpy, mesh_regular_map, meshRegularMap, "creates and select mesh by regular map\n"
                     "params:\n"
                     " path to surface PointCloud\n"
                     " path to directions PointCloud\n"
                     " path to distances binnary file (1/distance per point)\n"
                     " map width\n"
                     " map height\n")