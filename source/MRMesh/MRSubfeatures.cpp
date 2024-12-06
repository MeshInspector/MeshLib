#include "MRSubfeatures.h"

#include "MRPch/MRFmt.h"

namespace MR::Features
{

void forEachSubfeature( const Features::Primitives::Variant& feature, const SubfeatureFunc& func )
{
    std::visit( overloaded{
        [&]( const Features::Primitives::Sphere& sphere )
        {
            if ( sphere.radius > 0 )
            {
                func( { .name = "Center point", .isInfinite = false, .create = []( const Features::Primitives::Variant& f )
                {
                    auto ret = std::get<Features::Primitives::Sphere>( f );
                    ret.radius = 0;
                    return ret;
                } } );
            }
        },
        [&]( const Features::Primitives::ConeSegment& cone )
        {
            // Center point.
            func( { .name = "Center point", .isInfinite = false, .create = []( const Features::Primitives::Variant& f )
            {
                return std::get<Features::Primitives::ConeSegment>( f ).centerPoint();
            } } );

            // Central axis.
            if ( cone.positiveSideRadius > 0 || cone.negativeSideRadius > 0 )
            {
                bool infinite = cone.isCircle();

                func( { .name = "Axis", .isInfinite = infinite, .create = [infinite]( const Features::Primitives::Variant& f )
                {
                    auto ret = std::get<Features::Primitives::ConeSegment>( f ).axis();
                    if ( infinite )
                        ret = ret.extendToInfinity();
                    return ret;
                } } );
            }

            if ( cone.isCircle() )
            {
                // Plane.
                func( { .name = "Plane", .isInfinite = true, .create = []( const Features::Primitives::Variant& f )
                {
                    return std::get<Features::Primitives::ConeSegment>( f ).basePlane( false );
                } } );
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
                            func( { .name = backwardRadius == 0 ? fmt::format( "End point ({})", negativeCap ? "negative side" : "positive side" ).c_str() : "Apex", .isInfinite = false, .create = [negativeCap]( const Features::Primitives::Variant& f )
                            {
                                return std::get<Features::Primitives::ConeSegment>( f ).basePoint( negativeCap );
                            } } );
                        }
                        else
                        {
                            func( { .name = backwardRadius == 0 ? "Base circle" : fmt::format( "Base circle ({})", negativeCap ? "negative side" : "positive side" ).c_str(), .isInfinite = false, .create = [negativeCap]( const Features::Primitives::Variant& f )
                            {
                                return std::get<Features::Primitives::ConeSegment>( f ).baseCircle( negativeCap );
                            } } );
                        }
                    }
                }
            }

            // Extend to infinity.
            if ( cone.positiveSideRadius == cone.negativeSideRadius && ( std::isfinite( cone.positiveLength ) || std::isfinite( cone.negativeLength ) ) )
            {
                bool hasPositiveRadius = cone.positiveSideRadius > 0 || cone.negativeSideRadius > 0;

                if ( std::isfinite( cone.positiveLength ) && std::isfinite( cone.negativeLength ) )
                {
                    func( { .name = hasPositiveRadius ? "Infinite cylinder" : "Infinite line", .isInfinite = true, .create = []( const Features::Primitives::Variant& f )
                    {
                        return std::get<Features::Primitives::ConeSegment>( f ).extendToInfinity();
                    } } );
                }

                // We could have one-way extensions here, but they seem unnecessary.
            }

            // Untruncate to a full cone.
            if ( cone.positiveSideRadius != cone.negativeSideRadius && cone.positiveSideRadius > 0 && cone.negativeSideRadius > 0 )
                func( { .name = "Untruncated cone", .isInfinite = false, .create = []( const Features::Primitives::Variant& f )
            {
                return std::get<Features::Primitives::ConeSegment>( f ).untruncateCone();
            } } );
        },
        [&]( const Features::Primitives::Plane& )
        {
            func( { .name = "Center point", .isInfinite = false, .create = []( const Features::Primitives::Variant& f )
            {
                return Features::toPrimitive( std::get<Features::Primitives::Plane>( f ).center );
            } } );
        },
    }, feature );
}

}
