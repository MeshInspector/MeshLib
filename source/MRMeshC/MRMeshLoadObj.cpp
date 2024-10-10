#include "MRMeshLoadObj.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshLoadObj.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST2( std::string, MRString )
REGISTER_AUTO_CAST2( std::vector<MeshLoad::NamedMesh>, MRVectorMeshLoadNamedMesh )

MRMeshLoadObjLoadSettings mrMeshLoadObjLoadSettingsNew( void )
{
    static const MeshLoad::ObjLoadSettings def;
    return {
        .customXf = def.customXf,
        .countSkippedFaces = def.countSkippedFaces,
        .callback = nullptr,
    };
}

const MRMeshLoadNamedMesh mrVectorMeshLoadNamedMeshGet( const MRVectorMeshLoadNamedMesh* vector_, size_t index )
{
    ARG( vector );
    const auto& result = vector[index];
    return {
        .name = auto_cast( &result.name ),
        .mesh = auto_cast( &result.mesh ),
        // TODO: uvCoords
        // TODO: colors
        // TODO: textureFiles
        // TODO: texturePerFace
        // TODO: diffuseColor
        .xf = auto_cast( result.xf ),
        .skippedFaceCount = result.skippedFaceCount,
        .duplicatedVertexCount = result.duplicatedVertexCount,
    };
}

size_t mrVectorMeshLoadNamedMeshSize( const MRVectorMeshLoadNamedMesh* vector_ )
{
    ARG( vector );
    return vector.size();
}

void mrVectorMeshLoadNamedMeshFree( MRVectorMeshLoadNamedMesh* vector_ )
{
    ARG_PTR( vector );
    delete vector;
}

MRVectorMeshLoadNamedMesh* mrMeshLoadFromSceneObjFile( const char* file, bool combineAllObjects, const MRMeshLoadObjLoadSettings* settings_, MRString** errorString )
{
    MeshLoad::ObjLoadSettings settings;
    if ( settings_ )
    {
        settings = {
            .customXf = settings_->customXf,
            .countSkippedFaces = settings_->countSkippedFaces,
            .callback = settings_->callback,
        };
    }

    auto result = MeshLoad::fromSceneObjFile( file, combineAllObjects, settings );
    if ( !result.has_value() && errorString )
    {
        *errorString = auto_cast( new_from( std::move( result.error() ) ) );
    }
    if ( result )
        RETURN_NEW( std::move( *result ) );
    else
        return nullptr;
}
