#include "MRSelectCurvaturePreference.h"
#include "MRUIStyle.h"
#include "MRI18n.h"

namespace MR
{

float SelectCurvaturePreference( PathPreference* pp )
{
    static constexpr float factors[3] = { 0.0f, -2.0f, 2.0f };

    if ( !pp )
        return 0.0f;

    UI::combo( _tr( "Curvature Preference" ), ( int* )pp, { _tr( "Geodesic" ), _tr( "Convex" ), _tr( "Concave" ) }, true, { _tr( "Select the shortest boundary" ), _tr( "Select longer boundary but going in convex regions" ), _tr( "Select longer path but going in concave regions" ) } );
    UI::setTooltipIfHovered( _tr( "Select to prefer in selection convex/concave angles or neither" ) );
    return factors[int( *pp )];
}

}
