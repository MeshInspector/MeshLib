#include "MRMesh/MREmbeddedPython.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRMesh.h"
#include "MREAlgorithms/MREContoursCut.h"

MR_INIT_PYTHON_MODULE_PRECALL( mrealgorithmspy, [] ()
{
    pybind11::module_::import( "mrmeshpy" );
} )

MR_ADD_PYTHON_FUNCTION( mrealgorithmspy, cut_mesh_with_plane, MRE::cutMeshWithPlane, "cuts mesh with given plane" )
