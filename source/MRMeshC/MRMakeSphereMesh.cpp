#include "MRMakeSphereMesh.h"

#include "MRMesh/MRMakeSphereMesh.h"

using namespace MR;

static_assert( sizeof( MRSphereParams ) == sizeof( SphereParams ) );

MRSphereParams mrSphereParamsNew( void )
{
    SphereParams result;
    return reinterpret_cast<const MRSphereParams&>( result );
}

MRMesh* mrMakeSphere( const MRSphereParams* params_ )
{
    const auto& params = *reinterpret_cast<const SphereParams*>( params_ );

    auto result = makeSphere( params );

    return reinterpret_cast<MRMesh*>( new Mesh( std::move( result ) ) );
}

MRMakeUVSphereParameters mrMakeUvSphereParametersNew( void )
{
    return {
        .radius = 1.0,
        .horizontalResolution = 16,
        .verticalResolution = 16,
    };
}

MRMesh* mrMakeUVSphere( const MRMakeUVSphereParameters* params )
{
    auto result = makeUVSphere( params->radius, params->horizontalResolution, params->verticalResolution );

    return reinterpret_cast<MRMesh*>( new Mesh( std::move( result ) ) );
}
