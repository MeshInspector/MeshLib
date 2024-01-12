#include <MRMesh/MRPython.h>
#include <MRMesh/MRObject.h>

MR_ADD_PYTHON_CUSTOM_CLASS_DECL_1( mrmeshpy, SceneObject, MR::Object, std::shared_ptr<MR::Object> )
MR_ADD_PYTHON_CUSTOM_CLASS_INST_0( mrmeshpy, SceneObject )
