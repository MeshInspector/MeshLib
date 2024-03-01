#include "MRSubfeatures.h"

#include "MRPch/MRFmt.h"

namespace MR
{

void extractSubfeatures( const Features::Primitives::Variant& feature, const OfferSubfeatureFunc& offerSubFeature )
{
    std::string featureKindString = std::visit( []( const auto& p ) { return Features::name( p ); }, feature );

    std::visit( overloaded{
        [&]( const Features::Primitives::Sphere& sphere )
        {
            if ( sphere.radius > 0 )
                offerSubFeature( "Center point", {}, [&]{ auto ret = sphere; ret.radius = 0; return ret; } );
        },
        [&]( const Features::Primitives::ConeSegment& cone )
        {
            // Center point.
            offerSubFeature( "Center point", {}, [&] { return cone.centerPoint(); } );

            // Central axis.
            if ( cone.positiveSideRadius > 0 || cone.negativeSideRadius > 0 )
            {
                bool infinite = cone.isCircle();

                offerSubFeature( "Axis", OfferedSubfeatureFlags::visualizationIsIntrusive * infinite, [&]
                {
                    auto ret = cone.axis();
                    if ( infinite )
                        ret = ret.extendToInfinity();
                    return ret;
                } );
            }

            if ( cone.isCircle() )
            {
                // Plane.
                offerSubFeature( "Plane", OfferedSubfeatureFlags::visualizationIsIntrusive, [&]{ return cone.basePlane( false ); } );
            }
            else
            {
                // Caps.
                for ( bool negativeCap : { false, true } )
                {
                    float forwardLength = negativeCap ? -cone.negativeLength : cone.positiveLength;

                    float forwardRadius = negativeCap ? cone.negativeSideRadius : cone.positiveSideRadius;
                    float backwardRadius = negativeCap ? cone.positiveSideRadius : cone.negativeSideRadius;

                    if ( std::isfinite( forwardLength ) )
                    {
                        if ( forwardRadius == 0 )
                        {
                            offerSubFeature( backwardRadius == 0 ? fmt::format( "End point ({})", negativeCap ? "negative side" : "positive side" ).c_str() : "Apex", {}, [&]
                            {
                                return cone.basePoint( negativeCap );
                            } );
                        }
                        else
                        {
                            offerSubFeature( backwardRadius == 0 ? "Base circle" : fmt::format( "Base circle ({})", negativeCap ? "negative side" : "positive side" ).c_str(), {}, [&]
                            {
                                return cone.baseCircle( negativeCap );
                            } );

                            offerSubFeature(
                                backwardRadius == 0 ? "Center of base circle" : fmt::format( "Center of base circle ({})", negativeCap ? "negative side" : "positive side" ).c_str(),
                                OfferedSubfeatureFlags::indirect,
                                [&]
                                {
                                    return cone.basePoint( negativeCap );
                                }
                            );
                        }
                    }
                }
            }

            // Extend to infinity.
            if ( cone.positiveSideRadius == cone.negativeSideRadius && ( std::isfinite( cone.positiveLength ) || std::isfinite( cone.negativeLength ) ) )
            {
                bool hasPositiveRadius = cone.positiveSideRadius > 0 || cone.negativeSideRadius > 0;

                OfferedSubfeatureFlags extensionFlags = OfferedSubfeatureFlags::visualizationIsIntrusive * hasPositiveRadius;

                if ( std::isfinite( cone.positiveLength ) && std::isfinite( cone.negativeLength ) )
                {
                    offerSubFeature( hasPositiveRadius ? "Infinite cylinder" : "Infinite line", extensionFlags, [&] { return cone.extendToInfinity(); } );
                }
                #if 0 // One-way extensions.
                if ( std::isfinite( cone.positiveLength ) )
                {
                    offerSubFeature( fmt::format( "{} extended to pos. infinity", featureKindString ), extensionFlags, [&] { return cone.extendToInfinity( false ); } );
                }
                if ( std::isfinite( cone.negativeLength ) )
                {
                    offerSubFeature( fmt::format( "{} extended to neg. infinity", featureKindString ), extensionFlags, [&] { return cone.extendToInfinity( true ); } );
                }
                #endif
            }

            // Untruncate to a full cone.
            if ( cone.positiveSideRadius != cone.negativeSideRadius && cone.positiveSideRadius > 0 && cone.negativeSideRadius > 0 )
                offerSubFeature( "Untruncated cone", OfferedSubfeatureFlags::visualizationIsIntrusive, [&]{ return cone.untruncateCone(); } );
        },
        [&]( const Features::Primitives::Plane& plane )
        {
            offerSubFeature( "Center point", {}, [&]{ return Features::toPrimitive( plane.center ); } );
        },
    }, feature );
}

}
