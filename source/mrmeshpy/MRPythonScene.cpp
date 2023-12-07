#include <MRMesh/MRPython.h>
#include <MRMesh/MRObject.h>

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SceneObject, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::Object, std::shared_ptr<MR::Object>>( m, "SceneObject" );
} )
