#include "MRMeshDecimate.h"

#include "MRPch/MRSuppressWarning.h"

MR_SUPPRESS_WARNING_PUSH
MR_SUPPRESS_WARNING( "-Wdeprecated-declarations", 4996 )
#include "MRMesh/MRMeshDecimate.h"
MR_SUPPRESS_WARNING_POP

using namespace MR;

#define COPY_FROM( obj, field ) . field = ( obj ). field ,

MRDecimateSettings mrDecimateSettingsDefault()
{
    static const DecimateSettings def;
    return {
        .strategy = static_cast<MRDecimateStrategy>( def.strategy ),
        COPY_FROM( def, maxError )
        COPY_FROM( def, maxEdgeLen )
        COPY_FROM( def, maxBdShift )
        COPY_FROM( def, maxTriangleAspectRatio )
        COPY_FROM( def, criticalTriAspectRatio )
        COPY_FROM( def, tinyEdgeLength )
        COPY_FROM( def, stabilizer )
        COPY_FROM( def, optimizeVertexPos )
        COPY_FROM( def, maxDeletedVertices )
        COPY_FROM( def, maxDeletedFaces )
        .region = nullptr,
        // TODO: notFlippable
        // TODO: edgesToCollapse
        // TODO: touchBdVertices
        // TODO: bdVerts
        COPY_FROM( def, maxAngleChange )
        // TODO: preCollapse
        // TODO: adjustCollapse
        // TODO: onEdgeDel
        // TODO: vertForms
        COPY_FROM( def, packMesh )
        .progressCallback = nullptr,
        COPY_FROM( def, subdivideParts )
        COPY_FROM( def, decimateBetweenParts )
        // TODO: partFaces
        COPY_FROM( def, minFacesInPart )
    };
#undef COPY
}

MRDecimateResult mrDecimateMesh( MRMesh* mesh, const MRDecimateSettings* settings_ )
{
    DecimateSettings settings;
    if ( settings_ )
    {
        auto& src = *settings_;
        settings = {
            .strategy = static_cast<DecimateStrategy>( settings_->strategy ),
            COPY_FROM( src, maxError )
            COPY_FROM( src, maxEdgeLen )
            COPY_FROM( src, maxBdShift )
            COPY_FROM( src, maxTriangleAspectRatio )
            COPY_FROM( src, criticalTriAspectRatio )
            COPY_FROM( src, tinyEdgeLength )
            COPY_FROM( src, stabilizer )
            COPY_FROM( src, optimizeVertexPos )
            COPY_FROM( src, maxDeletedVertices )
            COPY_FROM( src, maxDeletedFaces )
            .region = reinterpret_cast<FaceBitSet*>( src.region ),
            // TODO: notFlippable
            // TODO: edgesToCollapse
            // TODO: touchBdVertices
            // TODO: bdVerts
            COPY_FROM( src, maxAngleChange )
            // TODO: preCollapse
            // TODO: adjustCollapse
            // TODO: onEdgeDel
            // TODO: vertForms
            COPY_FROM( src, packMesh )
            COPY_FROM( src, progressCallback )
            COPY_FROM( src, subdivideParts )
            COPY_FROM( src, decimateBetweenParts )
            // TODO: partFaces
            COPY_FROM( src, minFacesInPart )
        };
    }

    const auto res = decimateMesh(
        *reinterpret_cast<Mesh*>( mesh ),
        settings
    );
    // TODO: reinterpret_cast?
    // NOTE: C bool != C++ bool
    return {
        COPY_FROM( res, vertsDeleted )
        COPY_FROM( res, facesDeleted )
        COPY_FROM( res, errorIntroduced )
        COPY_FROM( res, cancelled )
    };
}

MRResolveMeshDegenSettings mrResolveMeshDegenSettingsDefault()
{
    static const ResolveMeshDegenSettings def;
    return {
        COPY_FROM( def, maxDeviation )
        COPY_FROM( def, tinyEdgeLength )
        COPY_FROM( def, maxAngleChange )
        COPY_FROM( def, criticalAspectRatio )
        COPY_FROM( def, stabilizer )
        .region = nullptr,
    };
}

bool mrResolveMeshDegenerations( MRMesh* mesh, const MRResolveMeshDegenSettings* settings_ )
{
    ResolveMeshDegenSettings settings;
    if ( settings_ )
    {
        auto& src = *settings_;
        settings = {
            COPY_FROM( src, maxDeviation )
            COPY_FROM( src, tinyEdgeLength )
            COPY_FROM( src, maxAngleChange )
            COPY_FROM( src, criticalAspectRatio )
            COPY_FROM( src, stabilizer )
            .region = reinterpret_cast<FaceBitSet*>( src.region ),
        };
    }

    return resolveMeshDegenerations(
        *reinterpret_cast<Mesh*>( mesh ),
        settings
    );
}

MRRemeshSettings mrRemeshSettingsDefault()
{
    static const RemeshSettings def;
    return {
        COPY_FROM( def, targetEdgeLen )
        COPY_FROM( def, maxEdgeSplits )
        COPY_FROM( def, maxAngleChangeAfterFlip )
        COPY_FROM( def, maxBdShift )
        COPY_FROM( def, useCurvature )
        COPY_FROM( def, finalRelaxIters )
        COPY_FROM( def, finalRelaxNoShrinkage )
        .region = nullptr,
        // TODO: notFlippable
        COPY_FROM( def, packMesh )
        COPY_FROM( def, projectOnOriginalMesh )
        // TODO: onEdgeSplit
        // TODO: onEdgeDel
        // TODO: preCollapse
        .progressCallback = nullptr,
    };
}

bool mrRemesh( MRMesh* mesh, const MRRemeshSettings* settings_ )
{
    RemeshSettings settings;
    if ( settings_ )
    {
        const auto& src = *settings_;
        settings = {
            COPY_FROM( src, targetEdgeLen )
            COPY_FROM( src, maxEdgeSplits )
            COPY_FROM( src, maxAngleChangeAfterFlip )
            COPY_FROM( src, maxBdShift )
            COPY_FROM( src, useCurvature )
            COPY_FROM( src, finalRelaxIters )
            COPY_FROM( src, finalRelaxNoShrinkage )
            .region = reinterpret_cast<FaceBitSet*>( src.region ),
            // TODO: notFlippable
            COPY_FROM( src, packMesh )
            COPY_FROM( src, projectOnOriginalMesh )
            // TODO: onEdgeSplit
            // TODO: onEdgeDel
            // TODO: preCollapse
            COPY_FROM( src, progressCallback )
        };
    }

    return remesh(
        *reinterpret_cast<Mesh*>( mesh ),
        settings
    );
}
