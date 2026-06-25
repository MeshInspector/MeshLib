#pragma once

#include "MRMesh/MRFlagOperators.h"

namespace MR
{

/// struct to determine transparent rendering mode
struct TransparencyMode
{
public:
    TransparencyMode() = default;
    TransparencyMode( bool alphaSort ) :alphaSort_{ alphaSort }
    {
    }
    TransparencyMode( unsigned bgDepthTexId, unsigned fgColorTexId, unsigned fgDepthTexId ) :
        bgDepthTexId_{ bgDepthTexId },
        fgColorTexId_{ fgColorTexId },
        fgDepthTexId_{ fgDepthTexId }
    {
    }
    bool isAlphaSortEnabled() const { return alphaSort_; }
    bool isDepthPeelingEnabled() const { return !alphaSort_ && bgDepthTexId_ != 0 && fgColorTexId_ != 0 && fgDepthTexId_ != 0; }
    unsigned getBGDepthPeelingDepthTextureId() const { return bgDepthTexId_; }
    unsigned getFGDepthPeelingColorTextureId() const { return fgColorTexId_; }
    unsigned getFGDepthPeelingDepthTextureId() const { return fgDepthTexId_; }

private:
    bool alphaSort_{ false };
    unsigned bgDepthTexId_ = 0;
    unsigned fgColorTexId_ = 0;
    unsigned fgDepthTexId_ = 0;
};

/// Various passes of the 3D rendering.
enum class RenderModelPassMask
{
    Opaque = 1 << 0,
    Transparent = 1 << 1,
    VolumeRendering = 1 << 2,
    NoDepthTest = 1 << 3,

    All =
        Opaque | Transparent | NoDepthTest | VolumeRendering
};
MR_MAKE_FLAG_OPERATORS( RenderModelPassMask )

}
