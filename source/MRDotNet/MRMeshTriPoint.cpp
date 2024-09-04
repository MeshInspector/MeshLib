#include "MRMeshTriPoint.h"
#include "MRVector3.h"

#pragma managed( push, off )
#include <MRMesh/MRMeshTriPoint.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

Vector3f^ TriPoint::Interpolate( Vector3f^ v0, Vector3f^ v1, Vector3f^ v2 )
{
    if ( !v0 )
        throw gcnew System::ArgumentNullException( "v0" );
    if ( !v1 )
        throw gcnew System::ArgumentNullException( "v1" );
    if ( !v2 )
        throw gcnew System::ArgumentNullException( "v2" );

    return  v0 * ( 1 - a - b ) + a * v1 + b * v2;
}

MeshTriPoint::MeshTriPoint( MR::MeshTriPoint* mtp )
{
    if ( !mtp )
        throw gcnew System::ArgumentNullException( "mtp" );

    mtp_ = mtp;
    e = EdgeId( mtp_->e );
    bary.a = mtp_->bary.a;
    bary.b = mtp_->bary.b;
}

MeshTriPoint::~MeshTriPoint()
{
    delete mtp_;
}

MR_DOTNET_NAMESPACE_END
