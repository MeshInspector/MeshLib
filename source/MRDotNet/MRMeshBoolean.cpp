#include "MRMeshBoolean.h"
#include "MRMesh.h"
#include "MRAffineXf.h"

#pragma managed( push, off )
#include <MRMesh/MRMeshBoolean.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRAffineXf3.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

BooleanResult MeshBoolean::Boolean( Mesh^ meshA, Mesh^ meshB, BooleanOperation op )
{
    BooleanParameters params;
    return Boolean( meshA, meshB, op, params );
}

BooleanResult MeshBoolean::Boolean( Mesh^ meshA, Mesh^ meshB, BooleanOperation op, BooleanParameters params )
{
    if ( !meshA )
        throw gcnew System::ArgumentNullException( "meshA" );
    if ( !meshB )
        throw gcnew System::ArgumentNullException( "meshB" );

    MR::AffineXf3f xf;
    if ( params.rigidB2A )
        xf =  *params.rigidB2A->xf();

    MR::BooleanParameters nativeParams;
    nativeParams.rigidB2A = &xf;
    nativeParams.mergeAllNonIntersectingComponents = params.mergeAllNonIntersectingComponents;

    auto nativeRes = MR::boolean( *meshA->getMesh(), *meshB->getMesh(), MR::BooleanOperation( op ), nativeParams );
    if ( !nativeRes.errorString.empty() )
        throw gcnew System::Exception( gcnew System::String( nativeRes.errorString.c_str() ) );

    BooleanResult res;
    res.mesh = gcnew Mesh( new MR::Mesh( std::move( nativeRes.mesh ) ) );
    
    return res;
}

MR_DOTNET_NAMESPACE_END

