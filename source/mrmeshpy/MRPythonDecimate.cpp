#include "mrmeshpy/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshDecimate.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Decimate, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::DecimateSettings>( m, "DecimateSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "maxError", &MR::DecimateSettings::maxError ).
        def_readwrite( "maxDeletedFaces", &MR::DecimateSettings::maxDeletedFaces ).
        def_readwrite( "maxDeletedVertices", &MR::DecimateSettings::maxDeletedVertices ).
        def_readwrite( "maxTriangleAspectRatio", &MR::DecimateSettings::maxTriangleAspectRatio ).
        def_readwrite( "stabilizer", &MR::DecimateSettings::stabilizer ).
        def_readwrite( "region", &MR::DecimateSettings::region );

    pybind11::class_<MR::DecimateResult>( m, "DecimateResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "vertsDeleted", &MR::DecimateResult::vertsDeleted ).
        def_readwrite( "facesDeleted", &MR::DecimateResult::facesDeleted ).
        def_readwrite( "errorIntroduced", &MR::DecimateResult::errorIntroduced );

    m.def( "decimate", MR::decimateMesh, "simplifies mesh by collapsing edges" );
} )
