#include "MRSvg.h"
#ifndef MRIOEXTRAS_NO_XML

#include <tinyxml2.h>

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRIOParsing.h"
#include "MRMesh/MRStringConvert.h"

namespace MR::LinesLoad
{

namespace
{

using namespace tinyxml2;

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
        if ( name == "line" )
            return parseLine_( elem );
        return {};
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
