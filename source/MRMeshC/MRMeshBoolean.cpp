#include "MRMeshBoolean.h"

#include "MRMesh/MRMeshBoolean.h"

using namespace MR;

MRBooleanResult mrBoolean( const MRMesh* meshA, const MRMesh* meshB, MRBooleanOperation operation, const MRBooleanParameters* params_ )
{
    BooleanParameters params;
    if ( params_ )
    {
        params = {
            .rigidB2A = reinterpret_cast<const AffineXf3f*>( params_->rigidB2A ),
            .mergeAllNonIntersectingComponents = params_->mergeAllNonIntersectingComponents,
            .cb = params_->cb,
        };
    }
    auto res = MR::boolean(
        *reinterpret_cast<const Mesh*>( meshA ),
        *reinterpret_cast<const Mesh*>( meshB ),
        static_cast<BooleanOperation>(operation),
        params
    );
    return {
        .mesh = reinterpret_cast<MRMesh*>( new Mesh( std::move( res.mesh ) ) ),
        .errorString = reinterpret_cast<MRString*>( new std::string( std::move( res.errorString ) ) ),
    };
}
