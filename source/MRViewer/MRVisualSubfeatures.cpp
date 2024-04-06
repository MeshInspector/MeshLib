#include "MRVisualSubfeatures.h"

namespace MR::Features
{

void forEachVisualSubfeature( const Features::Primitives::Variant& feature, const Features::SubfeatureFunc& func )
{
    Features::forEachSubfeature( feature, func );

    // cap centers
    if ( const auto* cone = std::get_if<Features::Primitives::ConeSegment>( &feature ) )
    {
        if ( !cone->isCircle() )
        {
            for ( bool negativeCap : { false, true } )
            {
                const auto length = negativeCap ? cone->negativeLength : cone->positiveLength;
                const auto sideRadius = negativeCap ? cone->negativeSideRadius : cone->positiveSideRadius;
                if ( std::isfinite( length ) && sideRadius > 0 )
                {
                    func( {
                        .name = negativeCap ? "Base circle center (negative side)" : "Base circle center (positive side)",
                        .isInfinite = false,
                        .create = [&] { return cone->basePoint( negativeCap ); },
                    } );
                }
            }
        }
    }
}

} // namespace MR::Features
