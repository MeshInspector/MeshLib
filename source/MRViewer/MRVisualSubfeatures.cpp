#include "MRVisualSubfeatures.h"

namespace MR::Features
{

void forEachVisualSubfeature( const Features::Primitives::Variant& feature, const Features::SubfeatureFunc& func )
{
    Features::forEachSubfeature( feature, func );

    // Cap centers.
    if ( const auto* cone = std::get_if<Features::Primitives::ConeSegment>( &feature ) )
    {
        if ( !cone->isCircle() )
        {
            for ( bool negativeCap : { false, true } )
            {
                float sideLength = negativeCap ? cone->negativeLength : cone->positiveLength;
                float sideRadius = negativeCap ? cone->negativeSideRadius : cone->positiveSideRadius;
                float otherSideRadius = negativeCap ? cone->positiveSideRadius : cone->negativeSideRadius;
                if ( std::isfinite( sideLength ) && sideRadius > 0 )
                {
                    func( {
                        .name = otherSideRadius <= 0 ? "Base circle center" : negativeCap ? "Base circle center (negative side)" : "Base circle center (positive side)",
                        .isInfinite = false,
                        .create = [&] { return cone->basePoint( negativeCap ); },
                    } );
                }
            }
        }
    }
}

} // namespace MR::Features
