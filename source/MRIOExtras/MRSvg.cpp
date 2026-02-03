#include "MRSvg.h"
#ifndef MRIOEXTRAS_NO_XML

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRIOParsing.h"
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

const auto point = rule<class point, MR::Vector2f>{ "point" }
                 = double_ >> -lit( ',' ) >> double_;
const auto points = point % -lit( ',' );

Expected<std::vector<Vector2f>> parsePoints( std::string_view str )
{
    std::vector<Vector2f> results;
    if ( !phrase_parse( str.begin(), str.end(), points, space, results ) )
        return unexpected( "Failed to parse points" );
    return results;
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

        Polyline2 result;
        if ( auto res = parseChildren_( svg, result ); !res )
            return unexpected( std::move( res.error() ) );
        // flip Y axis
        for ( auto& p : result.points )
            p.y *= -1.f;
        return result;
    }

private:
    Expected<void> parseChildren_( XMLElement* group, Polyline2& polyline )
    {
        for ( auto* child = group->FirstChildElement(); child != nullptr; child = child->NextSiblingElement() )
        {
            if ( child->Name() == std::string_view{ "g" } )
            {
                // TODO: update transform chain
                // https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/transform
                if ( auto res = parseChildren_( child, polyline ); !res )
                    return res;
            }
            else
            {
                if ( auto res = parseElement_( child ) )
                {
                    for ( const auto& contour : *res )
                    {
                        // TODO: apply transform
                        polyline.addFromPoints( contour.data(), contour.size() );
                    }
                }
                else
                {
                    return unexpected( std::move( res.error() ) );
                }
            }
        }
        return {};
    }

    Expected<Contours2f> parseElement_( XMLElement* elem ) const
    {
        const std::string_view name = elem->Name();
        if ( name == "circle" )
            return parseCircle_( elem );
        else if ( name == "ellipse" )
            return parseEllipse_( elem );
        else if ( name == "line" )
            return parseLine_( elem );
        else if ( name == "polygon" )
            return parsePolygon_( elem );
        else if ( name == "polyline" )
            return parsePolyline_( elem );
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
        points->emplace_back( points->front() );
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
