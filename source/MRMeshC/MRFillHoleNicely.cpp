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

MRFillHoleNicelyParams mrFillHoleNicelyParamsNew( void )
{
    MRFillHoleNicelyParams params;
    FillHoleNicelySettings defaultParams;
    params.triangulateParams = mrFillHoleParamsNew();
    params.triangulateOnly = defaultParams.triangulateOnly;    
    params.maxEdgeLen = defaultParams.maxEdgeLen;
    params.maxEdgeSplits = defaultParams.maxEdgeSplits;
    params.maxAngleChangeAfterFlip = defaultParams.maxAngleChangeAfterFlip;
    params.smoothCurvature = defaultParams.smoothCurvature;
    params.naturalSmooth = defaultParams.naturalSmooth;
    params.edgeWeights = (MREdgeWeights)defaultParams.edgeWeights;

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
        params.maxEdgeLen = params_->maxEdgeLen;
        params.maxEdgeSplits = params_->maxEdgeSplits;
        params.maxAngleChangeAfterFlip = params_->maxAngleChangeAfterFlip;
        params.smoothCurvature = params_->smoothCurvature;
        params.naturalSmooth = params_->naturalSmooth;
        params.edgeWeights = (MR::EdgeWeights)params_->edgeWeights;
    }
    RETURN_NEW(fillHoleNicely( mesh, holeEdge, params ) );
}