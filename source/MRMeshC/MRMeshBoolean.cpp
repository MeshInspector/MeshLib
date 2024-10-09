#include "MRMeshBoolean.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshBoolean.h"

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( BooleanOperation )
REGISTER_AUTO_CAST( BooleanResultMapper )
REGISTER_AUTO_CAST2( std::string, MRString )

MRBooleanParameters mrBooleanParametersNew( void )
{
    static const BooleanParameters def;
    return {
        .rigidB2A = auto_cast( def.rigidB2A ),
        .mapper = auto_cast( def.mapper ),
        .mergeAllNonIntersectingComponents = def.mergeAllNonIntersectingComponents,
        .cb = nullptr,
    };
}

MRBooleanResult mrBoolean( const MRMesh* meshA_, const MRMesh* meshB_, MRBooleanOperation operation_, const MRBooleanParameters* params_ )
{
    ARG( meshA ); ARG( meshB ); ARG_VAL( operation );

    BooleanParameters params;
    if ( params_ )
    {
        params = {
            .rigidB2A = auto_cast( params_->rigidB2A ),
            .mapper = auto_cast( params_->mapper ),
            .mergeAllNonIntersectingComponents = params_->mergeAllNonIntersectingComponents,
            .cb = params_->cb,
        };
    }
    auto res = MR::boolean( meshA, meshB, operation, params );
    return {
        .mesh = auto_cast( new_from( std::move( res.mesh ) ) ),
        .errorString = auto_cast( new_from( std::move( res.errorString ) ) ),
    };
}
