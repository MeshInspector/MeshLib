#include "MRSelectCurvaturePreference.h"
#include "MRUIStyle.h"

namespace MR
{

float SelectCurvaturePreference( CurvaturePreferenceMode* cp, float menuScaling )
{
    static constexpr float factors[3] = { 0.0f, -2.0f, 2.0f };

    if ( !cp )
        return 0.0f;
    
    UI::combo( "Curvature Preference", ( int* )cp, { "Geodesic", "Convex", "Concave" }, true, { "Select the shortest boundary", "Select longer boundary but going in more and more convex regions", "Select longer path but going in more and more concave regions" } );
    UI::setTooltipIfHovered( "Select to prefer in selection convex/concave angles or neither", menuScaling );
    return factors[int( *cp )];
}

}