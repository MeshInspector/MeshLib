#include "MRMeshFillHole.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRId.h"
#include "MRMesh/MRMeshFillHole.h"

#include <span>

using namespace MR;

REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( FillHoleMetric )
REGISTER_AUTO_CAST2( FillHoleParams::MultipleEdgesResolveMode, MRFillHoleMetricMultipleEdgesResolveMode )
REGISTER_AUTO_CAST( Mesh )

MRFillHoleParams mrFillHoleParamsNew( void )
{
    static const FillHoleParams def;
    return {
        .metric = auto_cast( &def.metric ),
        .smoothBd = def.smoothBd,
        .outNewFaces = auto_cast( def.outNewFaces ),
        .multipleEdgesResolveMode = auto_cast( def.multipleEdgesResolveMode ),
        .makeDegenerateBand = def.makeDegenerateBand,
        .maxPolygonSubdivisions = def.maxPolygonSubdivisions,
        .stopBeforeBadTriangulation = def.stopBeforeBadTriangulation,
    };
}

void mrFillHole( MRMesh* mesh_, MREdgeId a_, const MRFillHoleParams* params_ )
{
    ARG( mesh ); ARG_VAL( a );

    FillHoleParams params;
    if ( params_ )
    {
        params = {
            .metric = params_->metric ? *auto_cast( params_->metric ) : FillHoleMetric {},
            .smoothBd = params_->smoothBd,
            .outNewFaces = auto_cast( params_->outNewFaces ),
            .multipleEdgesResolveMode = auto_cast( params_->multipleEdgesResolveMode ),
            .makeDegenerateBand = params_->makeDegenerateBand,
            .maxPolygonSubdivisions = params_->maxPolygonSubdivisions,
            .stopBeforeBadTriangulation = params_->stopBeforeBadTriangulation,
        };
    }

    fillHole( mesh, a, params );
}

void mrFillHoles( MRMesh* mesh_, const MREdgeId* as_, size_t asNum, const MRFillHoleParams* params_ )
{
    ARG( mesh );

    std::span as { auto_cast( as_ ), asNum };

    // TODO: cast instead of copying
    std::vector<EdgeId> asVec( as.begin(), as.end() );

    FillHoleParams params;
    if ( params_ )
    {
        params = {
            .metric = params_->metric ? *auto_cast( params_->metric ) : FillHoleMetric {},
            .outNewFaces = auto_cast( params_->outNewFaces ),
            .multipleEdgesResolveMode = auto_cast( params_->multipleEdgesResolveMode ),
            .makeDegenerateBand = params_->makeDegenerateBand,
            .maxPolygonSubdivisions = params_->maxPolygonSubdivisions,
            .stopBeforeBadTriangulation = params_->stopBeforeBadTriangulation,
        };
    }

    fillHoles( mesh, asVec, params );
}

void mrBuildCylinderBetweenTwoHoles( MRMesh* mesh_, MREdgeId a_, MREdgeId b_, const MRStitchHolesParams* params_ )
{
    ARG( mesh ); ARG_VAL( a ); ARG_VAL( b );

    StitchHolesParams params;
    if ( params_ )
    {
        if ( params_->metric )
            params.metric = auto_cast( *params_->metric );
        params.outNewFaces = auto_cast( params_->outNewFaces );
    }

    buildCylinderBetweenTwoHoles( mesh, a, b, params );
}
