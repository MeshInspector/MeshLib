#include "MRMeshSubdivide.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshSubdivide.h"

using namespace MR;

REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )
REGISTER_AUTO_CAST( VertBitSet )

#define COPY_FROM( obj, field ) . field = auto_cast( ( obj ). field ),

MRSubdivideSettings mrSubdivideSettingsNew()
{
    static const SubdivideSettings def {};
    return {
        COPY_FROM( def, maxEdgeLen )
        COPY_FROM( def, maxEdgeSplits )
        COPY_FROM( def, maxDeviationAfterFlip )
        COPY_FROM( def, maxAngleChangeAfterFlip )
        COPY_FROM( def, criticalAspectRatioFlip )
        COPY_FROM( def, region )
        COPY_FROM( def, notFlippable )
        COPY_FROM( def, newVerts )
        COPY_FROM( def, subdivideBorder )
        COPY_FROM( def, maxTriAspectRatio )
        COPY_FROM( def, maxSplittableTriAspectRatio )
        COPY_FROM( def, smoothMode )
        COPY_FROM( def, minSharpDihedralAngle )
        COPY_FROM( def, projectOnOriginalMesh )
        .progressCallback = NULL,
    };
}

int mrSubdivideMesh( MRMesh* mesh_, const MRSubdivideSettings* settings_ )
{
    ARG( mesh );
    SubdivideSettings settings;
    if ( settings_ )
    {
        settings = SubdivideSettings {
            COPY_FROM( *settings_, maxEdgeLen )
            COPY_FROM( *settings_, maxEdgeSplits )
            COPY_FROM( *settings_, maxDeviationAfterFlip )
            COPY_FROM( *settings_, maxAngleChangeAfterFlip )
            COPY_FROM( *settings_, criticalAspectRatioFlip )
            .region = const_cast<FaceBitSet*>( auto_cast( settings_->region ) ),
            COPY_FROM( *settings_, notFlippable )
            COPY_FROM( *settings_, newVerts )
            COPY_FROM( *settings_, subdivideBorder )
            COPY_FROM( *settings_, maxTriAspectRatio )
            COPY_FROM( *settings_, maxSplittableTriAspectRatio )
            COPY_FROM( *settings_, smoothMode )
            COPY_FROM( *settings_, minSharpDihedralAngle )
            COPY_FROM( *settings_, projectOnOriginalMesh )
            .progressCallback = settings_->progressCallback,
        };
    }
    return subdivideMesh( mesh, settings );
}
