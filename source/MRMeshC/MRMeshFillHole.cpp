#include "MRMeshFillHole.h"

#include "MRMesh/MRId.h"
#include "MRMesh/MRMeshFillHole.h"

#include <span>

using namespace MR;

MRFillHoleParams mrFillHoleParamsNew()
{
    static const FillHoleParams def;
    return {
        .metric = reinterpret_cast<const MRFillHoleMetric*>( &def.metric ),
        .outNewFaces = reinterpret_cast<MRFaceBitSet*>( def.outNewFaces ),
        .multipleEdgesResolveMode = static_cast<MRFillHoleParamsMultipleEdgesResolveMode>( def.multipleEdgesResolveMode ),
        .makeDegenerateBand = def.makeDegenerateBand,
        .maxPolygonSubdivisions = def.maxPolygonSubdivisions,
        .stopBeforeBadTriangulation = def.stopBeforeBadTriangulation,
    };
}

void mrFillHole( MRMesh* mesh_, MREdgeId a_, const MRFillHoleParams* params_ )
{
    auto& mesh = *reinterpret_cast<Mesh*>( mesh_ );
    const auto& a = reinterpret_cast<EdgeId&>( a_ );

    FillHoleParams params;
    if ( params_ )
    {
        params = {
            .metric = params_->metric ? *reinterpret_cast<const FillHoleMetric*>( params_->metric ) : FillHoleMetric {},
            .outNewFaces = reinterpret_cast<FaceBitSet*>( params_->outNewFaces ),
            .multipleEdgesResolveMode = static_cast<FillHoleParams::MultipleEdgesResolveMode>( params_->multipleEdgesResolveMode ),
            .makeDegenerateBand = params_->makeDegenerateBand,
            .maxPolygonSubdivisions = params_->maxPolygonSubdivisions,
            .stopBeforeBadTriangulation = params_->stopBeforeBadTriangulation,
        };
    }

    fillHole( mesh, a, params );
}

void mrFillHoles( MRMesh* mesh_, const MREdgeId* as_, size_t asNum, const MRFillHoleParams* params_ )
{
    auto& mesh = *reinterpret_cast<Mesh*>( mesh_ );

    std::span as { reinterpret_cast<const EdgeId*>( as_ ), asNum };

    // TODO: cast instead of copying
    std::vector<EdgeId> asVec( as.begin(), as.end() );

    FillHoleParams params;
    if ( params_ )
    {
        params = {
            .metric = params_->metric ? *reinterpret_cast<const FillHoleMetric*>( params_->metric ) : FillHoleMetric {},
            .outNewFaces = reinterpret_cast<FaceBitSet*>( params_->outNewFaces ),
            .multipleEdgesResolveMode = static_cast<FillHoleParams::MultipleEdgesResolveMode>( params_->multipleEdgesResolveMode ),
            .makeDegenerateBand = params_->makeDegenerateBand,
            .maxPolygonSubdivisions = params_->maxPolygonSubdivisions,
            .stopBeforeBadTriangulation = params_->stopBeforeBadTriangulation,
        };
    }

    fillHoles( mesh, asVec, params );
}
