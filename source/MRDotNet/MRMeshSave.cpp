#include "MRMeshSave.h"
#include "MRMesh.h"
#include "MRAffineXf.h"

#include <msclr/marshal_cppstd.h>

#pragma managed( push, off )
#include <MRMesh/MRMeshSaveObj.h>
#include <filesystem>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

void MeshSave::SceneToObj( List<NamedMeshXf>^ meshes, System::String^ path )
{
    if ( !path )
        throw gcnew System::ArgumentNullException( "path" );

    if ( !meshes )
        throw gcnew System::ArgumentNullException( "meshes" );

    std::filesystem::path nativePath( msclr::interop::marshal_as<std::string>( path ) );
    std::vector<MR::MeshSave::NamedXfMesh> objects;

    for ( int i = 0; i < meshes->Count; ++i )
    {
        objects.emplace_back();
        auto& nativeMesh = objects.back();
        nativeMesh.name = msclr::interop::marshal_as<std::string>( meshes[i].name );
        nativeMesh.mesh = std::shared_ptr<const MR::Mesh>( new MR::Mesh( *meshes[i].mesh->getMesh() ) );
        nativeMesh.toWorld = MR::AffineXf3f( *meshes[i].toWorld->xf() );
    }

    auto resOrErr = MR::MeshSave::sceneToObj( objects, nativePath );
    if ( !resOrErr )
        throw gcnew System::SystemException( gcnew System::String( resOrErr.error().c_str() ) );
}

MR_DOTNET_NAMESPACE_END