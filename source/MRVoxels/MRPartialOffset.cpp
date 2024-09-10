#include "MRPartialOffset.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshBoolean.h"

namespace MR
{

Expected<Mesh> partialOffsetMesh( const MeshPart& mp, float offset, const GeneralOffsetParameters& params /*= {} */ )
{
    auto realParams = params;
    realParams.signDetectionMode = SignDetectionMode::Unsigned; // for now only shell can be in partial offset
    realParams.callBack = subprogress( params.callBack, 0.0f, 0.5f );
    auto offsetPart = generalOffsetMesh( mp, offset, realParams );
    if ( params.callBack && !params.callBack( 0.5f ) )
        return unexpectedOperationCanceled();
    if ( !offsetPart.has_value() )
        return offsetPart;
    auto res = boolean( mp.mesh, *offsetPart, BooleanOperation::Union, nullptr, nullptr, subprogress( params.callBack, 0.5f, 1.0f ) );
    if ( res.errorString == stringOperationCanceled() )
        return unexpectedOperationCanceled();
    if ( !res.valid() )
        return unexpected("Partial offset failed: " + res.errorString );
    return std::move( res.mesh );
}

} //namespace MR
