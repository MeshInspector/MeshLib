#pragma once
#include <MRMesh/MRVector2.h>
#include <algorithm>

namespace MR::FeatureAlignmentTool
{
struct PanelBounds
{
    Vector2f position, size;
};
inline PanelBounds fitPanelBounds( Vector2f position, Vector2f size, Vector2f workPosition, Vector2f workSize,
                                   float margin, Vector2f minimum = { 380, 320 } )
{
    for ( int i = 0; i < 2; ++i )
    {
        const auto inset = std::clamp( margin, 0.f, std::max( 0.f, ( workSize[i] - 1.f ) / 2.f ) );
        const auto maximum = std::max( 1.f, workSize[i] - 2.f * inset );
        size[i] = std::clamp( size[i], std::clamp( minimum[i], 1.f, maximum ), maximum );
        position[i] =
            std::clamp( position[i], workPosition[i] + inset, workPosition[i] + workSize[i] - inset - size[i] );
    }
    return { position, size };
}
struct PanelDiagnostics
{
    PanelBounds window, viewport;
    bool actionsInsideClip{};
};
} // namespace MR::FeatureAlignmentTool
