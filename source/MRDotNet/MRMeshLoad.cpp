#include "MRMeshLoad.h"
#include "MRMesh.h"
#include "MRAffineXf.h"

#include <msclr/marshal_cppstd.h>

#pragma managed( push, off )
#include <MRMesh/MRMeshLoadObj.h>
#include <filesystem>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

List<NamedMesh>^ MeshLoad::FromSceneObjFile( System::String^ path, bool combineAllObjects, ObjLoadSettings settings )
{
    if ( !path )
        throw gcnew System::ArgumentNullException( "path" );

    std::filesystem::path nativePath( msclr::interop::marshal_as<std::string>( path ) );
    MR::MeshLoad::ObjLoadSettings nativeSettings;
    nativeSettings.countSkippedFaces = settings.countSkippedFaces;
    nativeSettings.customXf = settings.customXf;

    auto resOrErr = MR::MeshLoad::fromSceneObjFile( nativePath, combineAllObjects, nativeSettings );
    if ( !resOrErr )
        throw gcnew System::SystemException( gcnew System::String( resOrErr.error().c_str() ) );

    auto res = gcnew List<NamedMesh>( int( resOrErr->size() ) );

    for ( auto& nativeNamedMesh : *resOrErr )
    {
        NamedMesh namedMesh;
        namedMesh.name = msclr::interop::marshal_as<System::String^>( nativeNamedMesh.name );
        namedMesh.mesh = gcnew Mesh( new MR::Mesh( std::move( nativeNamedMesh.mesh ) ) );
        if ( settings.customXf )
            namedMesh.xf = gcnew AffineXf3f( new MR::AffineXf3f( std::move( nativeNamedMesh.xf ) ) );

        if ( settings.countSkippedFaces )
        {
            namedMesh.skippedFaceCount = nativeNamedMesh.skippedFaceCount;
            namedMesh.duplicatedVertexCount = nativeNamedMesh.duplicatedVertexCount;
        }

        res->Add( namedMesh );
    }

    return res;
}

MR_DOTNET_NAMESPACE_END