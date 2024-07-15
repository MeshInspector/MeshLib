#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObject.h"
#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRObjectPointsHolder.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPython.h"

namespace
{

template <typename T, auto M>
auto extractModel( const MR::Object& object ) -> std::unique_ptr<std::remove_cvref_t<decltype(*(std::declval<T>().*M)())>>
{
    if ( auto base = dynamic_cast<const T*>( &object ) )
    {
        if ( auto ptr = (base->*M)() )
            return std::make_unique<std::remove_cvref_t<decltype( *ptr )>>( *ptr );
    }

    return nullptr;
}

} // namespace

MR_ADD_PYTHON_CUSTOM_CLASS_DECL( mrmeshpy, SceneObject, MR::Object, std::shared_ptr<MR::Object> )
MR_ADD_PYTHON_CUSTOM_CLASS_INST( mrmeshpy, SceneObject )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SceneObject, []( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( SceneObject )
        .def( "extractMesh", extractModel<MR::ObjectMeshHolder, &MR::ObjectMeshHolder::mesh> )
        .def( "extractPoints", extractModel<MR::ObjectPointsHolder, &MR::ObjectPointsHolder::pointCloud> )
        .def( "extractLines", extractModel<MR::ObjectLinesHolder, &MR::ObjectLinesHolder::polyline> )
    ;
} )
