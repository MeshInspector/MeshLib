#include "MRFillHoleNicely.h"
#include "detail/TypeCast.h"

#include "MRMesh/MRId.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRFillHoleNicely.h"

using namespace MR;

REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( FillHoleMetric )
REGISTER_AUTO_CAST2( FillHoleParams::MultipleEdgesResolveMode, MRFillHoleMetricMultipleEdgesResolveMode )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )

MRFillHoleNicelyParams mrFillHoleNicelyParamsNew( void )
{
    MRFillHoleNicelyParams params;
    FillHoleNicelySettings defaultParams;
    params.triangulateParams = mrFillHoleParamsNew();
    params.notFlippable = auto_cast( defaultParams.notFlippable );
    params.triangulateOnly = defaultParams.triangulateOnly;
    params.maxEdgeLen = defaultParams.maxEdgeLen;
    params.maxEdgeSplits = defaultParams.maxEdgeSplits;
    params.maxAngleChangeAfterFlip = defaultParams.maxAngleChangeAfterFlip;
    params.smoothCurvature = defaultParams.smoothCurvature;
    params.naturalSmooth = defaultParams.naturalSmooth;
    params.edgeWeights = (MREdgeWeights)defaultParams.edgeWeights;
    params.vmass = (MRVertexMass)defaultParams.vmass;

    return params;
}

MRFaceBitSet* mrFillHoleNicely( MRMesh* mesh_, MREdgeId holeEdge_, const MRFillHoleNicelyParams* params_ )
{
    ARG( mesh ); ARG_VAL( holeEdge );

    FillHoleNicelySettings params;
    if ( params_ )
    {
        params.triangulateParams = MR::FillHoleParams{
            .metric = params_->triangulateParams.metric ? *auto_cast( params_->triangulateParams.metric ) : FillHoleMetric {},
            .outNewFaces = auto_cast( params_->triangulateParams.outNewFaces ),
            .multipleEdgesResolveMode = auto_cast( params_->triangulateParams.multipleEdgesResolveMode ),
            .makeDegenerateBand = params_->triangulateParams.makeDegenerateBand,
            .maxPolygonSubdivisions = params_->triangulateParams.maxPolygonSubdivisions,
            .stopBeforeBadTriangulation = params_->triangulateParams.stopBeforeBadTriangulation,
        };

        params.triangulateOnly = params_->triangulateOnly;
        params.notFlippable = auto_cast( params_->notFlippable );
        params.maxEdgeLen = params_->maxEdgeLen;
        params.maxEdgeSplits = params_->maxEdgeSplits;
        params.maxAngleChangeAfterFlip = params_->maxAngleChangeAfterFlip;
        params.smoothCurvature = params_->smoothCurvature;
        params.naturalSmooth = params_->naturalSmooth;
        params.edgeWeights = (MR::EdgeWeights)params_->edgeWeights;
        params.vmass = (MR::VertexMass)params_->vmass;
    }
    RETURN_NEW(fillHoleNicely( mesh, holeEdge, params ) );
}