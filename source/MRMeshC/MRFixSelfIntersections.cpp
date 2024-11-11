#include "MRFixSelfIntersections.h"
#include "MRBitSet.h"
#include "MRMesh.h"

#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRFixSelfIntersections.h"

#include "detail/TypeCast.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST2( std::string, MRString )

MRFixSelfIntersectionsSettings mrFixSelfIntersectionsSettingsNew( void )
{
    MRFixSelfIntersectionsSettings res;
    res.method = MRFixSelfIntersectionsMethodRelax;
    res.relaxIterations = 5;
    res.maxExpand = 3;
    res.subdivideEdgeLen = 0.0f;
    res.cb = nullptr;

    return res;
}

MRFaceBitSet* mrFixSelfIntersectionsGetFaces( const MRMesh* mesh_, MRProgressCallback cb, MRString** errorString )
{
    ARG( mesh );
    auto resOrErr = SelfIntersections::getFaces( mesh, cb );

    MRFaceBitSet* res = nullptr;
    if ( resOrErr )
    {
        res = auto_cast( new_from( std::move( resOrErr.value() ) ) );
    }

    if ( errorString && !resOrErr )
        *errorString = auto_cast( new_from( std::move( resOrErr.error() ) ) );

    return res;
}

void mrFixSelfIntersectionsFix( MRMesh* mesh_, const MRFixSelfIntersectionsSettings* settings, MRString** errorString )
{
    ARG( mesh );
    auto resOrErr = SelfIntersections::fix( mesh, {
        .method = SelfIntersections::Settings::Method( settings->method ),
        .relaxIterations = settings->relaxIterations,
        .maxExpand = settings->maxExpand,
        .subdivideEdgeLen = settings->subdivideEdgeLen,
        .callback = settings->cb } );

    if ( errorString && !resOrErr )
        *errorString = auto_cast( new_from( std::move( resOrErr.error() ) ) );
}