#include "MRPython/MRPython.h"
#include "MRMesh/MRRegularMapMesher.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRLog.h"

using namespace MR;

MR::Mesh meshRegularMap( const char* surfacePCPath, const char* directionsPCPath,
                     const char* distancesPath, int width, int height )
{
    RegularMapMesher mesher;
    {
        auto surfacePC = PointsLoad::fromAnySupportedFormat( surfacePCPath );
        if ( !surfacePC.has_value() )
        {
            spdlog::error( surfacePC.error() );
            return {};
        }

        mesher.setSurfacePC( std::make_shared<PointCloud>( std::move( surfacePC.value() ) ) );
    }
    {
        auto directionsPC = PointsLoad::fromAnySupportedFormat( directionsPCPath );
        if ( !directionsPC.has_value() )
        {
            spdlog::error( directionsPC.error() );
            return {};
        }

        mesher.setDirectionsPC( std::make_shared<PointCloud>( std::move( directionsPC.value() ) ) );
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

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, RegularMapMeshing, [] ( pybind11::module_& m )
{
    m.def( "meshRegularMap", &meshRegularMap,
        pybind11::arg( "surfacePCPath" ), pybind11::arg( "directionsPCPath" ), pybind11::arg( "distancesPath" ), pybind11::arg( "width" ), pybind11::arg( "height" ),
        "creates and select mesh by regular map\n"
        "\tsurfacePCPath - path to surface PointCloud\n"
        "\tdirectionsPCPath - path to directions PointCloud\n"
        "\tdistancesPath - path to distances binnary file (1/distance per point)\n"
        "\twidth - map width\n"
        "\theight - map height" );
} )
