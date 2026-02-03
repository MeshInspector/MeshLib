#include "MRSvg.h"
#ifndef MRIOEXTRAS_NO_XML

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRIOParsing.h"
#include "MRMesh/MRMatrix2.h"
#include "MRMesh/MRStringConvert.h"

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/spirit/home/x3.hpp>

#include <tinyxml2.h>

// required for parsing lists of points
BOOST_FUSION_ADAPT_STRUCT( MR::Vector2f, x, y )

namespace MR::LinesLoad
{

namespace
{

using namespace tinyxml2;

namespace parser
{

using namespace boost::spirit::x3;

constexpr auto point = rule<class point, MR::Vector2f>{ "point" }
                     = double_ >> -lit( ',' ) >> double_;
constexpr auto points = point % -lit( ',' );

Expected<std::vector<Vector2f>> parsePoints( std::string_view str )
{
    std::vector<Vector2f> results;
    if ( !phrase_parse( str.begin(), str.end(), points, space, results ) )
        return unexpected( "Failed to parse points" );
    return results;
}

Expected<AffineXf2f> parseTransform( std::string_view str )
{
#define _(...) ( [] ( auto& ctx ) { [[maybe_unused]] auto& xf = _val( ctx ); [[maybe_unused]] auto& A = xf.A; [[maybe_unused]] auto& b = xf.b; [[maybe_unused]] auto& val = _attr( ctx ); __VA_ARGS__; } )

    constexpr auto matrix
        = rule<class matrix, AffineXf2f>{ "matrix" }
        = lit( "matrix" ) >> '('
            >> double_[_( A.x.x = val )] >> -lit( ',' )
            >> double_[_( A.y.x = val )] >> -lit( ',' )
            >> double_[_( A.x.y = val )] >> -lit( ',' )
            >> double_[_( A.y.y = val )] >> -lit( ',' )
            >> double_[_( b.x = val )] >> -lit( ',' )
            >> double_[_( b.y = val )]
          >> ')';
    constexpr auto translate
        = rule<class translate, AffineXf2f>{ "translate" }
        = lit( "translate" ) >> '('
            >> double_[_( b.x = val )] >> -lit( ',' )
            >> -double_[_( b.y = val )]
          >> ')';
    constexpr auto scale
        = rule<class scale, AffineXf2f>{ "scale" }
        = lit( "scale" ) >> '('
            // uniform scaling by default
            >> double_[_( A.x.x = val, A.y.y = val )] >> -lit( ',' )
            >> -double_[_( A.y.y = val )]
          >> ')';
    constexpr auto rotate
        = rule<class rotate, AffineXf2f>{ "rotate" }
        = lit( "rotate" ) >> '('
            >> double_[_( A = Matrix2f::rotation( val * PI_F / 180.f ) )] >> -lit( ',' )
            // optional translation
            >> -( point[_( xf = AffineXf2f::translation( val ) * xf * AffineXf2f::translation( -val ) )])
          >> ')';
    constexpr auto skewX
        = rule<class skewX, AffineXf2f>{ "skewX" }
        = lit( "skewX" ) >> '('
            >> double_[_( A.x.y = std::tan( val * PI_F / 180.f ) )]
          >> ')';
    constexpr auto skewY
        = rule<class skewY, AffineXf2f>{ "skewY" }
        = lit( "skewY" ) >> '('
            >> double_[_( A.y.x = std::tan( val * PI_F / 180.f ) )]
          >> ')';

#undef _

    constexpr auto transform = ( matrix | translate | scale | rotate | skewX | skewY ) % -lit( ',' );

    std::vector<AffineXf2f> results;
    if ( !phrase_parse( str.begin(), str.end(), transform, space, results ) )
        return unexpected( "Failed to parse points" );

    AffineXf2f result;
    for ( auto xf : results )
        result = result * xf;

    return result;
}

} // namespace parser

struct EllipseParams
{
    float cx = 0.f;
    float cy = 0.f;
    float rx = 1.f;
    float ry = 1.f;
    float a0 = 0.f;
    float a1 = 2.f * PI;
    int resolution = 32;
};

Contour2f getEllipsePoints( const EllipseParams params = {} )
{
    assert( params.a0 < params.a1 );
    Contour2f results;
    for ( auto i = 0; i <= params.resolution; ++i )
    {
        const auto a = params.a0 + ( params.a1 - params.a0 ) * (float)i / (float)params.resolution;
        const auto x = std::cos( a ), y = std::sin( a );
        results.push_back( {
            x * params.rx + params.cx,
            y * params.ry + params.cy,
        } );
    }
    return results;
}

void close( Contour2f& contour )
{
    if ( contour.back() != contour.front() )
        contour.emplace_back( contour.front() );
}

class SvgLoader
{
public:
    explicit SvgLoader( const std::filesystem::path& path )
    {
        const auto pathStr = utf8string( path );
        doc_.LoadFile( pathStr.c_str() );
    }
    explicit SvgLoader( const char* data, size_t size )
    {
        doc_.Parse( data, size );
    }

    explicit operator bool () const { return !doc_.Error(); }
    std::string error() const { return doc_.ErrorStr(); }

    Expected<Polyline2> parse()
    {
        auto* svg = doc_.RootElement();
        if ( !svg || svg->Name() != std::string_view{ "svg" } )
            return unexpected( "Not an SVG document" );

        if ( auto contours = parseChildren_( svg ) )
        {
            Polyline2 result{ *contours };
            // flip Y axis
            for ( auto& p : result.points )
                p.y *= -1.f;
            return result;
        }
        else
        {
            return unexpected( std::move( contours.error() ) );
        }
    }

private:
    Expected<Contours2f> parseChildren_( XMLElement* group ) const
    {
        Contours2f results;
        for ( auto* child = group->FirstChildElement(); child != nullptr; child = child->NextSiblingElement() )
        {
            auto contours = parseElement_( child );
            if ( !contours )
                return contours;

            // https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/transform
            if ( const auto* transform = child->FindAttribute( "transform" ) )
            {
                const auto xf = parser::parseTransform( transform->Value() );
                if ( !xf )
                    return unexpected( std::move( xf.error() ) );

                for ( auto& contour : *contours )
                    for ( auto& p : contour )
                        p = (*xf)( p );
            }

            for ( auto&& contour : *contours )
                results.emplace_back( std::move( contour ) );
        }
        return results;
    }

    Expected<Contours2f> parseElement_( XMLElement* elem ) const
    {
        const std::string_view name = elem->Name();

        // group
        if ( name == "g" )
            return parseChildren_( elem );

        // shape
        if ( name == "circle" )
            return parseCircle_( elem );
        if ( name == "ellipse" )
            return parseEllipse_( elem );
        if ( name == "line" )
            return parseLine_( elem );
        if ( name == "polygon" )
            return parsePolygon_( elem );
        if ( name == "polyline" )
            return parsePolyline_( elem );
        if ( name == "rect" )
            return parseRect_( elem );

        // omitting unknown element
        return {};
    }

    Expected<Contours2f> parseCircle_( XMLElement* elem ) const
    {
        const auto cx = elem->FloatAttribute( "cx", 0.f );
        const auto cy = elem->FloatAttribute( "cy", 0.f );
        const auto r = elem->FloatAttribute( "r", 0.f );
        if ( r == 0.f )
            return {};
        auto points = getEllipsePoints( {
            .cx = cx,
            .cy = cy,
            .rx = r,
            .ry = r,
        } );
        return { { std::move( points ) } };
    }

    Expected<Contours2f> parseEllipse_( XMLElement* elem ) const
    {
        const auto cx = elem->FloatAttribute( "cx", 0.f );
        const auto cy = elem->FloatAttribute( "cy", 0.f );
        const auto rx = elem->FloatAttribute( "rx", 0.f );
        const auto ry = elem->FloatAttribute( "ry", 0.f );
        if ( rx == 0.f || ry == 0.f )
            return {};
        auto points = getEllipsePoints( {
            .cx = cx,
            .cy = cy,
            .rx = rx,
            .ry = ry,
        } );
        return { { std::move( points ) } };
    }

    Expected<Contours2f> parseLine_( XMLElement* elem ) const
    {
        const auto x1 = elem->FloatAttribute( "x1", 0.f );
        const auto y1 = elem->FloatAttribute( "y1", 0.f );
        const auto x2 = elem->FloatAttribute( "x2", 0.f );
        const auto y2 = elem->FloatAttribute( "y2", 0.f );
        Contour2 line {
            Vector2f { x1, y1 },
            Vector2f { x2, y2 },
        };
        return { { std::move( line ) } };
    }

    Expected<Contours2f> parsePolygon_( XMLElement* elem ) const
    {
        const std::string pointsStr = elem->Attribute( "points" );
        auto points = parser::parsePoints( pointsStr );
        if ( !points )
            return unexpected( std::move( points.error() ) );
        close( *points );
        return { { std::move( *points ) } };
    }

    Expected<Contours2f> parsePolyline_( XMLElement* elem ) const
    {
        const std::string pointsStr = elem->Attribute( "points" );
        auto points = parser::parsePoints( pointsStr );
        if ( !points )
            return unexpected( std::move( points.error() ) );
        return { { std::move( *points ) } };
    }

    Expected<Contours2f> parseRect_( XMLElement* elem ) const
    {
        const auto x = elem->FloatAttribute( "x", 0.f );
        const auto y = elem->FloatAttribute( "y", 0.f );
        const auto width = elem->FloatAttribute( "width", 0.f );
        const auto height = elem->FloatAttribute( "height", 0.f );
        auto rx = elem->FloatAttribute( "rx", 0.f );
        auto ry = elem->FloatAttribute( "ry", 0.f );
        if ( width == 0.f || height == 0.f )
            return {};
        if ( rx == 0.f && ry == 0.f )
        {
            Contour2f rect {
                { x, y },
                { x, y + height },
                { x + width, y + height },
                { x + width, y },
                { x, y },
            };
            return { { std::move( rect ) } };
        }
        else
        {
            // SVG requirements
            if ( rx == 0.f )
                rx = ry;
            else if ( ry == 0.f )
                ry = rx;
            if ( width / 2.f < rx )
                rx = width / 2.f;
            if ( height / 2.f < ry )
                ry = height / 2.f;

            Contour2f points;
            for ( auto p : getEllipsePoints( {
                .cx = x + width - rx,
                .cy = y + ry,
                .rx = rx,
                .ry = ry,
                .a0 = -PI2_F,
                .a1 = 0.f,
            } ) )
                points.emplace_back( p );
            for ( auto p : getEllipsePoints( {
                .cx = x + width - rx,
                .cy = y + height - ry,
                .rx = rx,
                .ry = ry,
                .a0 = 0.f,
                .a1 = PI2_F,
            } ) )
                points.emplace_back( p );
            for ( auto p : getEllipsePoints( {
                .cx = x + rx,
                .cy = y + height - ry,
                .rx = rx,
                .ry = ry,
                .a0 = PI2_F,
                .a1 = PI_F,
            } ) )
                points.emplace_back( p );
            for ( auto p : getEllipsePoints( {
                .cx = x + rx,
                .cy = y + ry,
                .rx = rx,
                .ry = ry,
                .a0 = -PI_F,
                .a1 = -PI2_F,
            } ) )
                points.emplace_back( p );
            close( points );
            return { { std::move( points ) } };
        }
    }

private:
    XMLDocument doc_;
};

} // namespace

Expected<Polyline3> fromSvg( const std::filesystem::path& file, const LinesLoadSettings& )
{
    SvgLoader loader( file );
    if ( !loader )
        return unexpected( loader.error() );

    return loader.parse().transform( [] ( Polyline2&& polyline2 )
    {
        return polyline2.toPolyline<Vector3f>();
    } );
}

Expected<Polyline3> fromSvg( std::istream& in, const LinesLoadSettings& settings )
{
    return readCharBuffer( in ).and_then( [&] ( Buffer<char>&& buf )
    {
        return fromSvg( buf.data(), buf.size(), settings );
    } );
}

Expected<Polyline3> fromSvg( const char* data, size_t size, const LinesLoadSettings& )
{
    SvgLoader loader( data, size );
    if ( !loader )
        return unexpected( loader.error() );

    return loader.parse().transform( [] ( Polyline2&& polyline2 )
    {
        return polyline2.toPolyline<Vector3f>();
    } );
}

MR_ADD_LINES_LOADER(IOFilter( "SVG (.svg)", "*.svg" ), fromSvg)

} // namespace MR::LinesLoad
#endif
