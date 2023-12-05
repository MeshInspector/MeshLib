#include "MRSelectCurvaturePreference.h"
#include "MRUIStyle.h"

namespace MR
{

float SelectCurvaturePreference( PathPreference* pp, float menuScaling )
{
    static constexpr float factors[3] = { 0.0f, -2.0f, 2.0f };

    if ( !pp )
        return 0.0f;
    
    UI::combo( "Curvature Preference", ( int* )pp, { "Geodesic", "Convex", "Concave" }, true, { "Select the shortest boundary", "Select longer boundary but going in convex regions", "Select longer path but going in concave regions" } );
    UI::setTooltipIfHovered( "Select to prefer in selection convex/concave angles or neither", menuScaling );
    return factors[int( *pp )];
}

}