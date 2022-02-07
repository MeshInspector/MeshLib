#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MREAlgorithms/MREMeshDecimate.h"

MR_ADD_PYTHON_CUSTOM_DEF( mrealgorithmspy, Decimate, [] ( pybind11::module_& m )
{
    pybind11::class_<MRE::DecimateSettings>( m, "DecimateSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "maxError", &MRE::DecimateSettings::maxError ).
        def_readwrite( "maxDeletedFaces", &MRE::DecimateSettings::maxDeletedFaces ).
        def_readwrite( "maxDeletedVertices", &MRE::DecimateSettings::maxDeletedVertices ).
        def_readwrite( "maxTriangleAspectRatio", &MRE::DecimateSettings::maxTriangleAspectRatio ).
        def_readwrite( "stabilizer", &MRE::DecimateSettings::stabilizer ).
        def_readwrite( "region", &MRE::DecimateSettings::region );

    pybind11::class_<MRE::DecimateResult>( m, "DecimateResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "vertsDeleted", &MRE::DecimateResult::vertsDeleted ).
        def_readwrite( "facesDeleted", &MRE::DecimateResult::facesDeleted ).
        def_readwrite( "errorIntroduced", &MRE::DecimateResult::errorIntroduced );

    m.def( "decimate", MRE::decimateMesh, "simplifies mesh by collapsing edges" );
} )
