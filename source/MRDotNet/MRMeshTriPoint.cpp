#include "MRMeshTriPoint.h"

#pragma managed( push, off )
#include <MRMesh/MRMeshTriPoint.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

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
